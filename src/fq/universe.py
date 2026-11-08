"""Ticker universe and industry exclusions, loaded from packaged text files."""

import re
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from fq.settings import FINANCIAL_MODELS


def _read_lines(name: str) -> list[str]:
    text = resources.files("fq.lists").joinpath(name).read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


SELECTORS = ("test", "benchmark", "portfolio")  # named sets; own output folders
LIST_SELECTORS = ("test", "bist", "india")  # one packaged list file each
# The parts of `benchmark`, the one global set every run is scored against (one list file each).
BENCHMARK_SETS = (
    "sp500", "ibex35", "dax", "eurostoxx50", "ftse100", "sti", "hsi_hk", "klci", "obx", "omxs30", "wig20", "asx50", "msci_world", "msci_em",
    "ftse250", "asx_mid_small", "st_smallcap", "spain_continuo", "mdax_sdax", "russell_small", "nasdaq_mid", "nyse_mid",
    "msci_eafe_small", "msci_europe_small", "omx_smallcap", "ibovespa", "russell_2000_rest", "nasdaq_nextgen", "offshore_leaders", "tsx_composite", "ftse_smallcap", "singapore_smallcap", "spain_100m", "germany_100m", "singapore_100m", "uk_100m", "hk_100m", "saudi_100m", "japan_100m", "hk_smallcap", "spain_smallcap", "germany_smallcap", "bist", "india", "nl_midsmall", "pl_midsmall", "fr_midsmall", "ca_midsmall", "my_midsmall", "vietnam", "cayman_adrs", "thailand_100m", "brazil_100m", "mexico_100m", "gulf_100m", "greece_100m", "csi800", "others",
)  # fmt: skip
PORTFOLIO_FILE = Path("portfolio.txt")  # resolved against the working directory (the project root)


@dataclass(frozen=True)
class Selection:
    """A resolved run set: `label` names the output folder and cache file."""

    label: str
    tickers: list[str]


def _read_ticker_file(path: Path) -> list[str]:
    lines = [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln.upper() for ln in lines if ln]


def _expand(token: str) -> list[str] | None:
    """A selector word or file expanded to tickers; None when the token is a plain symbol."""
    if token in LIST_SELECTORS:
        return sorted(set(_read_lines(f"{token}.txt")))
    if token == "benchmark":
        return sorted({t for name in BENCHMARK_SETS for t in benchmark_part(name)})
    if token == "portfolio":
        if not PORTFOLIO_FILE.is_file():
            raise ValueError(f"{PORTFOLIO_FILE.resolve()} not found; create it with one ticker per line")
        return portfolio_tickers()
    path = Path(token)
    if path.suffix and path.is_file():
        return _read_ticker_file(path)
    return None


def portfolio_tickers() -> list[str]:
    """The holdings in portfolio.txt; empty when the file does not exist."""
    return _read_ticker_file(PORTFOLIO_FILE) if PORTFOLIO_FILE.is_file() else []


def benchmark_part(name: str) -> list[str]:
    """Tickers of one part of the benchmark (`BENCHMARK_SETS`); `others` less `excluded_tickers.txt`
    (pruned from the old full universe; the index lists are taken whole), plus every holding in no
    index list: holdings are always in the benchmark, excluded or not (user's choice, Oct 2026)."""
    tickers = set(_read_lines(f"{name}.txt"))
    if name != "others":
        return sorted(tickers)
    indexed = {t for part in BENCHMARK_SETS if part != "others" for t in _read_lines(f"{part}.txt")}
    held = set(portfolio_tickers()) - indexed
    return sorted((tickers - set(_read_lines("excluded_tickers.txt"))) | held)


def resolve_selection(spec: str) -> Selection:
    """Turn a run spec into tickers.

    `spec` is a comma-separated mix of `benchmark`, `test`, `portfolio`, ticker-file paths and plain
    symbols, e.g. `portfolio,DECK` or `AAPL,MSFT,ASML.AS`. Order is kept, duplicates dropped.
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise ValueError("Empty ticker selection")
    tickers: list[str] = []
    for tok in tokens:
        expanded = _expand(tok.lower() if tok.lower() in SELECTORS else tok)
        tickers += expanded if expanded is not None else [tok.upper()]
    tickers = list(dict.fromkeys(tickers))
    label = tokens[0].lower() if len(tokens) == 1 and tokens[0].lower() in SELECTORS else "custom"
    return Selection(label, tickers)


def load_universe(name: str = "benchmark") -> list[str]:
    """Convenience wrapper: the tickers of a selection spec."""
    return resolve_selection(name).tickers


def industry_key(industry: str) -> str:
    """Normalise an industry label (Yahoo mixes dashes and mojibake)."""
    return re.sub(r"[^a-z0-9]", "", str(industry).lower())


@cache
def excluded_industry_keys() -> frozenset[str]:
    return frozenset(industry_key(i) for i in _read_lines("excluded_industries.txt"))


def _ticker_list(name: str) -> frozenset[str]:
    return frozenset(line.split("#")[0].strip().upper() for line in _read_lines(name))


@cache
def model_overrides() -> dict[str, str]:
    """Ticker -> business model from `lists/model_overrides.txt` (where the Yahoo industry misleads)."""
    out = {}
    for line in _read_lines("model_overrides.txt"):
        ticker, model = line.split("#")[0].split()[:2]
        out[ticker.upper()] = model
    return out


@cache
def industry_models() -> dict[str, str]:
    """Industry key -> business model from `lists/business_models.txt`."""
    out = {}
    for line in _read_lines("business_models.txt"):
        model, industry = line.split("#")[0].split(":", 1)
        out[industry_key(industry.strip())] = model.strip()
    return out


def business_model(ticker: str, industry) -> str:
    """`operating`, `client_cash`, `float`, `asset_heavy`, `bank`, `insurer` or `holding` (see
    `settings.BUSINESS_MODELS`): the ticker override, else the industry's model, else operating."""
    override = model_overrides().get(str(ticker).upper())
    if override:
        return override
    return industry_models().get(industry_key(industry), "operating") if isinstance(industry, str) else "operating"


def is_excluded(ticker: str, industry) -> bool:
    """Out of scope for ranking: an excluded industry (no usable cash flow), unless a ticker override
    gives the company a business model."""
    in_list = isinstance(industry, str) and industry_key(industry) in excluded_industry_keys()
    return in_list and str(ticker).upper() not in model_overrides()


def is_financial(ticker: str, industry, sector=None) -> bool:
    """Measured in financial mode: a bank, insurer or holding company (`settings.FINANCIAL_MODELS`)."""
    return business_model(ticker, industry) in FINANCIAL_MODELS


@cache
def _dividend_wht_table() -> tuple[dict[str, float], dict[str, float]]:
    """(country key -> %, ticker -> %) from `lists/dividend_wht.txt`."""
    countries, tickers = {}, {}
    for line in _read_lines("dividend_wht.txt"):
        entry = line.split("#")[0].strip()
        if ":" in entry:
            country, rate = entry.split(":", 1)
            countries[industry_key(country)] = float(rate)
        elif entry:
            ticker, rate = entry.split()[:2]
            tickers[ticker.upper()] = float(rate)
    return countries, tickers


def dividend_wht(ticker: str, country) -> float:
    """Statutory dividend withholding (%) on a non-resident individual: the ticker override, else Yahoo's country,
    else NaN. Shown only, never scored."""
    countries, tickers = _dividend_wht_table()
    rate = tickers.get(str(ticker).upper())
    if rate is None and isinstance(country, str):
        rate = countries.get(industry_key(country))
    return float("nan") if rate is None else rate


@cache
def _sovereign_spread_table() -> dict[str, float]:
    """country key -> sovereign dollar default spread (%) from `lists/sovereign_spread.txt`."""
    table = {}
    for line in _read_lines("sovereign_spread.txt"):
        entry = line.split("#")[0].strip()
        if ":" in entry:
            country, spread = entry.split(":", 1)
            table[industry_key(country)] = float(spread)
    return table


def sovereign_spread(country) -> float:
    """How much more than the US Yahoo's `country`'s government pays to borrow dollars, % (its default spread in
    `lists/sovereign_spread.txt` less the United States', at least 0): 0 for a country not listed or unknown, the
    Treasury already holding the US's own. The Boring companies bond hurdle only (user's choice, Nov 7, 2026); never
    scored."""
    table = _sovereign_spread_table()
    spread = table.get(industry_key(country)) if isinstance(country, str) else None
    return 0.0 if spread is None else max(0.0, spread - table.get(industry_key("United States"), 0.0))


def share_classes() -> dict[str, tuple[str, float]]:
    """Linked share classes from `lists/share_classes.txt`: ticker -> (sister ticker, sister/own ratio)."""
    links = {}
    for line in _read_lines("share_classes.txt"):
        ticker, sister, ratio = line.split("#")[0].split()[:3]
        links[ticker.upper()] = (sister.upper(), float(ratio))
    return links


@cache
def accounting_overrides() -> dict[str, str]:
    """Ticker -> `ifrs` / `us_gaap` from `lists/accounting_standards.txt`."""
    out = {}
    for line in _read_lines("accounting_standards.txt"):
        ticker, standard = line.split("#")[0].split()[:2]
        out[ticker.upper()] = standard
    return out


def accounting_standard(ticker: str) -> str:
    """`us_gaap` or `ifrs` (IFRS or a local standard): the override, else IFRS for a ticker with an
    exchange suffix (listed at home outside the US). Yahoo does not say which standard a company uses."""
    ticker = str(ticker).upper()
    return accounting_overrides().get(ticker, "ifrs" if "." in ticker else "us_gaap")


@cache
def inflation_basis() -> dict[str, str]:
    """Ticker -> `restated` / `nominal` from `lists/inflation_basis.txt`."""
    out = {}
    for line in _read_lines("inflation_basis.txt"):
        ticker, basis = line.split("#")[0].split()[:2]
        out[ticker.upper()] = basis
    return out


@cache
def cyclical_industry_keys() -> frozenset[str]:
    return frozenset(industry_key(i) for i in _read_lines("cyclical_industries.txt"))


@cache
def real_estate_industry_keys() -> frozenset[str]:
    return frozenset(industry_key(i) for i in _read_lines("real_estate_owners.txt"))


def utility_industry_keys() -> frozenset[str]:
    return frozenset(industry_key(i) for i in _read_lines("utilities.txt"))


def is_utility(industry) -> bool:
    """Industry in `lists/utilities.txt`: scored among utilities, ranked apart (`ranking.rank_group`, "utility")."""
    return isinstance(industry, str) and industry_key(industry) in utility_industry_keys()


def is_real_estate_owner(industry) -> bool:
    """Industry in `lists/real_estate_owners.txt` (the equity REITs): scored, ranked apart (`ranking.rank_group`)."""
    return isinstance(industry, str) and industry_key(industry) in real_estate_industry_keys()


def is_cyclical_industry(industry) -> bool:
    """Industry in `lists/cyclical_industries.txt`: Cyclical even when five years show no trough."""
    return isinstance(industry, str) and industry_key(industry) in cyclical_industry_keys()
