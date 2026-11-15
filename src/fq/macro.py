"""Historical macro yields: 10-year US Treasury yield (^TNX) vs. S&P 500 TTM earnings yield (Multpl).

Fetches monthly series from 1970-01-01 onward, caches atomically to `data/macro_yields.parquet`, and falls
back to the cached file when offline or if a remote source fails.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

START_DATE = pd.Timestamp("1960-01-01")
FRED_GS10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
MULTPL_EY_URL = "https://www.multpl.com/s-p-500-earnings-yield/table/by-month"
CACHE_FILE = "macro_yields.parquet"

MOF_JGB_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
JPX_PER_INDEX_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/04.html"
JPX_LONGRANGE_FALLBACK_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/tvdivq00000015r6-att/longrange-perpbr.xlsx"
JPX_CACHE_FILE = "macro_japan_yields.parquet"

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def _to_month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index).tz_localize(None)
    return idx + pd.offsets.MonthEnd(0)


def fetch_fred_gs10(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly 10-year US Treasury yield (%) from FRED (`GS10`), clipped to `start` onward."""
    df = pd.read_csv(FRED_GS10_URL)
    dates = pd.to_datetime(df["observation_date"], errors="coerce")
    vals = pd.to_numeric(df["GS10"], errors="coerce")
    valid = dates.notna() & vals.notna() & (vals > 0)
    series = pd.Series(vals[valid].values, index=_to_month_end(pd.DatetimeIndex(dates[valid])), name="treasury_10y", dtype=float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= start]


def fetch_treasury_history(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly 10-year US Treasury yield (%) from Yahoo (^TNX, 1962+) backfilled from FRED GS10 (1960–1961)."""
    hist = yf.download("^TNX", period="max", interval="1d", progress=False, auto_adjust=False)["Close"]
    series = hist.iloc[:, 0] if isinstance(hist, pd.DataFrame) else hist
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.resample("ME").last().dropna()
    if start < series.index.min():
        try:
            gs10 = fetch_fred_gs10(start=start)
            series = series.combine_first(gs10)
        except Exception as exc:
            log.warning("Could not backfill pre-1962 Treasury yields from FRED GS10 (%s)", exc)
    series = series[series.index >= start].sort_index()
    series.name = "treasury_10y"
    return series.astype(float)


def parse_multpl_table(html: str, start: pd.Timestamp = START_DATE) -> pd.Series:
    """Parse Multpl's monthly S&P 500 earnings yield HTML table into a month-end Series (%)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="datatable") or soup.find("table")
    if table is None:
        raise ValueError("No data table found in Multpl HTML")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        raw_date = cells[0].get_text(strip=True)
        raw_val = cells[1].get_text(strip=True).replace("%", "").replace(",", "").strip()
        try:
            dt = pd.Timestamp(raw_date)
            val = float(raw_val)
        except (ValueError, TypeError):
            continue
        dates.append(dt)
        values.append(val)

    if not dates:
        raise ValueError("No rows parsed from Multpl earnings yield table")

    series = pd.Series(values, index=_to_month_end(pd.DatetimeIndex(dates)), name="sp500_earnings_yield", dtype=float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= start]


def fetch_sp500_earnings_yield(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Download and parse Multpl's monthly S&P 500 TTM earnings yield (%) from `start` onward."""
    req = urllib.request.Request(MULTPL_EY_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed https URL
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")
    return parse_multpl_table(html, start=start)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _cache_is_fresh(cache_path: Path, max_age_days: int) -> bool:
    if not cache_path.is_file():
        return False
    age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
    return age_days <= max_age_days


def load_macro_yields(
    cache_path: Path,
    max_age_days: int = 7,
    offline: bool = False,
    start: pd.Timestamp = START_DATE,
) -> pd.DataFrame:
    """Load monthly 10Y Treasury yield, S&P 500 TTM earnings yield, and their spread (pp) from `start`.

    Uses `cache_path` when fresh or when `offline=True`; otherwise refreshes from Yahoo (`^TNX`) and Multpl,
    falling back to the cached parquet if fetching fails. Returns an empty DataFrame if unavailable.
    """
    if cache_path.is_file() and (offline or _cache_is_fresh(cache_path, max_age_days)):
        df = pd.read_parquet(cache_path)
        if offline or df.empty or df.index.min() <= start + pd.offsets.MonthEnd(1):
            return df[df.index >= start].sort_index()

    if offline:
        return pd.DataFrame(columns=["treasury_10y", "sp500_earnings_yield", "spread_pp"])

    try:
        ty = fetch_treasury_history(start=start)
        ey = fetch_sp500_earnings_yield(start=start)
        df = pd.concat([ty, ey], axis=1).dropna(subset=["treasury_10y", "sp500_earnings_yield"]).sort_index()
        if df.empty:
            raise ValueError("Joined macro yields DataFrame is empty")
        df["spread_pp"] = (df["sp500_earnings_yield"] - df["treasury_10y"]).round(4)
        df.index.name = "date"
        _atomic_write_parquet(df, cache_path)
        log.info("Macro yields updated (%d months: %s to %s)", len(df), df.index[0].strftime("%Y-%m"), df.index[-1].strftime("%Y-%m"))
        return df
    except Exception as exc:
        log.warning("Could not refresh macro yields (%s)", exc)
        if cache_path.is_file():
            df = pd.read_parquet(cache_path)
            return df[df.index >= start].sort_index()
        return pd.DataFrame(columns=["treasury_10y", "sp500_earnings_yield", "spread_pp"])


def parse_mof_jgb_csv(csv_text: str) -> pd.Series:
    """Parse Japan MOF daily JGB CSV into a month-end 10-year yield Series (%)."""
    lines = csv_text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Date,")), None)
    if header_idx is None:
        raise ValueError("No Date header found in MOF JGB CSV")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    if "Date" not in df.columns or "10Y" not in df.columns:
        raise ValueError("Expected Date and 10Y columns in MOF JGB CSV")
    dates = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    vals = pd.to_numeric(df["10Y"], errors="coerce")
    valid = dates.notna() & vals.notna()
    series = pd.Series(vals[valid].values, index=pd.DatetimeIndex(dates[valid]), name="jgb_10y", dtype=float)
    series = series.sort_index().resample("ME").last().dropna()
    return series


def fetch_japan_jgb_history() -> pd.Series:
    """Download and parse Japan Ministry of Finance 10-year JGB yields (%)."""
    req = urllib.request.Request(MOF_JGB_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed https URL
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
    return parse_mof_jgb_csv(text)


def _xlsx_col_idx(cell_ref: str) -> int:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _read_xlsx_sheet_rows(z: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[dict[int, str]]:
    root = ET.fromstring(z.read(sheet_path))
    rows: list[dict[int, str]] = []
    for row in root.iter():
        if not row.tag.endswith("row"):
            continue
        cells: dict[int, str] = {}
        for cell in row:
            ref = cell.attrib.get("r", "")
            t = cell.attrib.get("t", "")
            val = next((child.text for child in cell if child.tag.endswith("v") and child.text is not None), None)
            if val is None:
                continue
            if t == "s":
                val = shared[int(val)]
            cells[_xlsx_col_idx(ref)] = val
        if cells:
            rows.append(cells)
    return rows


def _extract_jpx_per_rows(
    z: zipfile.ZipFile, sheet_path: str, shared: list[str], per_col: int
) -> dict[pd.Timestamp, float]:
    records: dict[pd.Timestamp, float] = {}
    for cells in _read_xlsx_sheet_rows(z, sheet_path, shared):
        raw_y = cells.get(0, "").strip()
        raw_m = cells.get(1, "").strip()
        raw_per = cells.get(per_col, "").strip()
        if not (re.fullmatch(r"\d{4}", raw_y) and re.fullmatch(r"\d{1,2}", raw_m)):
            continue
        try:
            per = float(raw_per)
            if per <= 0:
                continue
            dt = pd.Timestamp(year=int(raw_y), month=int(raw_m), day=1) + pd.offsets.MonthEnd(0)
        except (ValueError, TypeError):
            continue
        records[dt] = 100.0 / per
    return records


def parse_jpx_perpbr_xlsx(raw_bytes: bytes) -> pd.Series:
    """Parse JPX `longrange-perpbr.xlsx` into a month-end earnings yield (%) Series (1971–present).

    Priority per month: Consolidated Weighted avg (2013+) > Consolidated AVG (1999–2012) >
    Non-Consolidated AVG (1971–1998), using TSE 1st Section / Prime Section PER.
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(si.itertext()) for si in ss_root.iter() if si.tag.endswith("si")]

        wb_root = ET.fromstring(z.read("xl/workbook.xml"))
        rels_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"].lstrip("/")
            for rel in rels_root
            if rel.tag.endswith("Relationship")
        }
        sheets: dict[str, str] = {}
        for sh in wb_root.iter():
            if sh.tag.endswith("sheet"):
                name = sh.attrib.get("name", "")
                rid = next((v for k, v in sh.attrib.items() if k.endswith("id")), "")
                target = rel_targets.get(rid, "")
                if target and not target.startswith("xl/"):
                    target = f"xl/{target}"
                if name and target:
                    sheets[name] = target

        # Ordered from lowest to highest priority so newer/better bases overwrite older ones on overlap.
        tiers: list[tuple[str, int]] = []
        for name in sheets:
            if name.startswith("Non-Consolidated(AVG)"):
                tiers.append((name, 8))  # col I: TSE 1st Section PER (1971-12+)
        for name in sheets:
            if name.startswith("Consolidated(AVG)"):
                tiers.append((name, 3))  # col D: TSE 1st Section PER (1999-01+)
        for name in sheets:
            if "Consolidated" in name and ("Weighted avg" in name or "WT AVG" in name):
                tiers.append((name, 3))  # col D: TSE 1st Section / Prime PER (2013-01+)

        if not tiers:
            raise ValueError("No recognised PER sheets found in JPX workbook")

        records: dict[pd.Timestamp, float] = {}
        for sheet_name, per_col in tiers:
            records.update(_extract_jpx_per_rows(z, sheets[sheet_name], shared, per_col))

    if not records:
        raise ValueError("No PER rows parsed from JPX workbook")
    return pd.Series(records, name="topix_earnings_yield", dtype=float).sort_index()


def _discover_jpx_longrange_url() -> str:
    try:
        req = urllib.request.Request(JPX_PER_INDEX_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "longrange-perpbr.xlsx" in href:
                return href if href.startswith("http") else f"https://www.jpx.co.jp{href}"
    except Exception as exc:
        log.debug("JPX index page lookup failed (%s); using fallback URL", exc)
    return JPX_LONGRANGE_FALLBACK_URL


def fetch_japan_earnings_yield() -> pd.Series:
    """Download and parse JPX Consolidated Weighted-Average PER into monthly earnings yield (%)."""
    url = _discover_jpx_longrange_url()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw_bytes = resp.read()
    return parse_jpx_perpbr_xlsx(raw_bytes)


def load_japan_yields(
    cache_path: Path,
    max_age_days: int = 7,
    offline: bool = False,
) -> pd.DataFrame:
    """Load monthly Japan 10Y JGB yield, TSE Prime/1st-Sec earnings yield, and spread (pp)."""
    if cache_path.is_file() and (offline or _cache_is_fresh(cache_path, max_age_days)):
        df = pd.read_parquet(cache_path).sort_index()
        if offline or df.empty or df.index.min() <= pd.Timestamp("1975-01-31"):
            return df

    if offline:
        return pd.DataFrame(columns=["jgb_10y", "topix_earnings_yield", "spread_pp"])

    try:
        jgb = fetch_japan_jgb_history()
        ey = fetch_japan_earnings_yield()
        df = pd.concat([jgb, ey], axis=1).dropna(subset=["jgb_10y", "topix_earnings_yield"]).sort_index()
        if df.empty:
            raise ValueError("Joined Japan yields DataFrame is empty")
        df["spread_pp"] = (df["topix_earnings_yield"] - df["jgb_10y"]).round(4)
        df.index.name = "date"
        _atomic_write_parquet(df, cache_path)
        log.info("Japan yields updated (%d months: %s to %s)", len(df), df.index[0].strftime("%Y-%m"), df.index[-1].strftime("%Y-%m"))
        return df
    except Exception as exc:
        log.warning("Could not refresh Japan yields (%s)", exc)
        if cache_path.is_file():
            return pd.read_parquet(cache_path).sort_index()
        return pd.DataFrame(columns=["jgb_10y", "topix_earnings_yield", "spread_pp"])
