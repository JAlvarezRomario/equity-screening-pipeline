"""Auxiliary, opt-in Scrapy crawler for auditing public index constituent tables against `fq.lists`.

Never imported by the scoring core, fetch, report or main pipeline; runs only when explicitly invoked
via `fq indices` (or `python -m fq.crawlers`). Parses public constituent tables, checks that table headers
and row counts match expected bounds (source-layout observability), normalises symbols to Yahoo Finance
tickers, and reports constituent additions and removals without modifying packaged list files."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import scrapy
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse, Response

from fq import universe


class LayoutChangeError(ValueError):
    """Raised when a crawled constituent page drops expected table columns or breaches row-count bounds."""


@dataclass(frozen=True)
class IndexSpec:
    """Configuration and validation rules for one public index constituent table."""

    name: str
    url: str
    min_rows: int
    max_rows: int
    suffix: str = ""
    dot_to_dash: bool = False
    ignored_tickers: frozenset[str] = frozenset()
    ticker_aliases: tuple[tuple[str, str], ...] = ()

    def alias_map(self) -> dict[str, str]:
        return dict(self.ticker_aliases)


@dataclass(frozen=True)
class ConstituentRecord:
    """One constituent row extracted and normalised from a public index table."""

    index: str
    raw_symbol: str
    ticker: str
    company: str
    sector: str = ""


@dataclass(frozen=True)
class IndexDriftReport:
    """Comparison between crawled constituents and the packaged `src/fq/lists/<index>.txt` file."""

    index: str
    url: str
    crawled_count: int
    list_count: int
    added: list[str]
    removed: list[str]
    layout_ok: bool
    error: str | None = None


INDEX_SPECS: dict[str, IndexSpec] = {
    "sp500": IndexSpec(
        name="sp500",
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        min_rows=495,
        max_rows=510,
        dot_to_dash=True,
        # Secondary share classes excluded in `lists/sp500.txt` (one class per company).
        ignored_tickers=frozenset({"GOOG", "FOX", "NWS"}),
    ),
    "dax": IndexSpec(
        name="dax",
        url="https://en.wikipedia.org/wiki/DAX",
        min_rows=38,
        max_rows=42,
        suffix=".DE",
    ),
    "eurostoxx50": IndexSpec(
        name="eurostoxx50",
        url="https://en.wikipedia.org/wiki/EURO_STOXX_50",
        min_rows=48,
        max_rows=52,
    ),
    "ftse100": IndexSpec(
        name="ftse100",
        url="https://en.wikipedia.org/wiki/FTSE_100_Index",
        min_rows=95,
        max_rows=105,
        suffix=".L",
        dot_to_dash=True,
        ignored_tickers=frozenset({"CCEP.L"}),
    ),
    "ibex35": IndexSpec(
        name="ibex35",
        url="https://en.wikipedia.org/wiki/IBEX_35",
        min_rows=33,
        max_rows=37,
        suffix=".MC",
        # Primary listing MT.AS is held in msci_world.txt instead of MTS.MC.
        ignored_tickers=frozenset({"MTS.MC"}),
    ),
    "omxs30": IndexSpec(
        name="omxs30",
        url="https://en.wikipedia.org/wiki/OMX_Stockholm_30",
        min_rows=28,
        max_rows=32,
        suffix=".ST",
        dot_to_dash=True,
        ignored_tickers=frozenset({"AZN.ST", "NDA-SE.ST", "ABB.ST", "ATCO-B.ST"}),
    ),
}

_TICKER_HEADERS = ("symbol", "ticker", "ticker symbol", "epic", "code")
_NAME_HEADERS = ("security", "company", "name", "issuer", "constituent")
_SECTOR_HEADERS = ("gics sector", "sector", "industry", "icb sector", "subsector")


def normalise_ticker(raw: str, spec: IndexSpec) -> str:
    """Turn a raw table symbol into a canonical Yahoo Finance ticker for `spec`."""
    cleaned = re.sub(r"\[.*?\]", "", raw).strip().upper()
    cleaned = cleaned.split()[0] if cleaned else ""
    if not cleaned:
        return ""
    aliases = spec.alias_map()
    if cleaned in aliases:
        return aliases[cleaned]
    if spec.suffix and cleaned.endswith(spec.suffix.upper()):
        base = cleaned[: -len(spec.suffix)]
        suffix = spec.suffix.upper()
    elif "." in cleaned and not spec.dot_to_dash and not spec.suffix:
        return cleaned
    else:
        base = cleaned
        suffix = spec.suffix.upper()
    if spec.dot_to_dash:
        base = base.rstrip(".").replace(".", "-")
    return f"{base}{suffix}"


def _match_column(headers: list[str], candidates: tuple[str, ...]) -> int | None:
    for idx, header in enumerate(headers):
        norm = re.sub(r"[^a-z0-9 ]", "", header.lower()).strip()
        if any(norm == c or norm.startswith(c + " ") or norm.endswith(" " + c) for c in candidates):
            return idx
    return None


def parse_constituent_table(html: str, spec: IndexSpec) -> list[ConstituentRecord]:
    """Extract and validate constituent rows from an HTML page according to `spec`.

    Raises `LayoutChangeError` when no table carries the required ticker/company headers or
    when the number of extracted constituents falls outside `[spec.min_rows, spec.max_rows]`.
    """
    soup = BeautifulSoup(html, "html.parser")
    matched_table = None
    ticker_col = name_col = sector_col = None

    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if first_row is None:
            continue
        headers = [cell.get_text(" ", strip=True) for cell in first_row.find_all(["th", "td"])]
        t_idx = _match_column(headers, _TICKER_HEADERS)
        n_idx = _match_column(headers, _NAME_HEADERS)
        if t_idx is not None and n_idx is not None:
            matched_table = table
            ticker_col = t_idx
            name_col = n_idx
            sector_col = _match_column(headers, _SECTOR_HEADERS)
            break

    if matched_table is None or ticker_col is None or name_col is None:
        raise LayoutChangeError(f"{spec.name}: no HTML table with required ticker/company columns found at {spec.url}")

    records: list[ConstituentRecord] = []
    seen: set[str] = set()
    for row in matched_table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(ticker_col, name_col):
            continue
        raw_symbol = cells[ticker_col].get_text(" ", strip=True)
        company = re.sub(r"\[.*?\]", "", cells[name_col].get_text(" ", strip=True)).strip()
        sector = (
            re.sub(r"\[.*?\]", "", cells[sector_col].get_text(" ", strip=True)).strip()
            if sector_col is not None and len(cells) > sector_col
            else ""
        )
        ticker = normalise_ticker(raw_symbol, spec)
        if not ticker or not company:
            continue
        if ticker in spec.ignored_tickers or ticker in seen:
            continue
        seen.add(ticker)
        records.append(
            ConstituentRecord(
                index=spec.name,
                raw_symbol=raw_symbol,
                ticker=ticker,
                company=company,
                sector=sector,
            )
        )

    if not (spec.min_rows <= len(records) <= spec.max_rows):
        raise LayoutChangeError(
            f"{spec.name}: extracted {len(records)} unique constituents, outside expected bounds "
            f"[{spec.min_rows}, {spec.max_rows}]"
        )
    return records


def compare_with_list(
    spec: IndexSpec,
    records: list[ConstituentRecord],
    packaged_tickers: set[str] | None = None,
) -> IndexDriftReport:
    """Compare validated `records` against `src/fq/lists/<spec.name>.txt`."""
    current = (
        set(universe.benchmark_part(spec.name))
        if packaged_tickers is None
        else {t.strip().upper() for t in packaged_tickers if t.strip()}
    )
    crawled = {r.ticker for r in records} - set(spec.ignored_tickers)
    return IndexDriftReport(
        index=spec.name,
        url=spec.url,
        crawled_count=len(crawled),
        list_count=len(current),
        added=sorted(crawled - current),
        removed=sorted(current - crawled),
        layout_ok=True,
        error=None,
    )


class IndexConstituentsSpider(scrapy.Spider):
    """Scrapy spider that crawls public index pages and emits `IndexDriftReport` dictionaries."""

    name = "fq_index_constituents"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        "DOWNLOAD_DELAY": 0.5,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 2,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
        "USER_AGENT": "fq-index-audit/0.1 (+https://github.com/JAlvarezRomario/equity-screening-pipeline)",
        "LOG_LEVEL": "WARNING",
    }

    def __init__(
        self,
        indices: list[str] | None = None,
        results: list[IndexDriftReport] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        selected = indices or list(INDEX_SPECS)
        unknown = [i for i in selected if i not in INDEX_SPECS]
        if unknown:
            raise ValueError(f"Unknown index spec(s): {', '.join(unknown)}. Valid: {', '.join(INDEX_SPECS)}")
        self.specs = [INDEX_SPECS[i] for i in selected]
        self.results = results if results is not None else []

    def start_requests(self):
        for spec in self.specs:
            yield scrapy.Request(url=spec.url, callback=self.parse_index, cb_kwargs={"spec": spec})

    def parse_index(self, response: Response, spec: IndexSpec):
        report = self.evaluate_response(response, spec)
        self.results.append(report)
        yield asdict(report)

    @staticmethod
    def evaluate_response(
        response: Response | HtmlResponse,
        spec: IndexSpec,
        packaged_tickers: set[str] | None = None,
    ) -> IndexDriftReport:
        """Validate a Scrapy `Response` against `spec` and return its `IndexDriftReport`."""
        try:
            records = parse_constituent_table(response.text, spec)
            return compare_with_list(spec, records, packaged_tickers=packaged_tickers)
        except LayoutChangeError as exc:
            current_len = (
                len(packaged_tickers) if packaged_tickers is not None else len(universe.benchmark_part(spec.name))
            )
            return IndexDriftReport(
                index=spec.name,
                url=spec.url,
                crawled_count=0,
                list_count=current_len,
                added=[],
                removed=[],
                layout_ok=False,
                error=str(exc),
            )


def run_index_audit(indices: list[str] | None = None, json_out: Path | None = None) -> list[IndexDriftReport]:
    """Run `IndexConstituentsSpider` across `indices` and optionally write the JSON report."""
    reports: list[IndexDriftReport] = []
    process = CrawlerProcess(settings=IndexConstituentsSpider.custom_settings)
    process.crawl(IndexConstituentsSpider, indices=indices, results=reports)
    process.start()
    reports.sort(key=lambda r: r.index)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    return reports
