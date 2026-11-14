"""Historical macro yields: 10-year US Treasury yield (^TNX) vs. S&P 500 TTM earnings yield (Multpl).

Fetches monthly series from 1970-01-01 onward, caches atomically to `data/macro_yields.parquet`, and falls
back to the cached file when offline or if a remote source fails.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

START_DATE = pd.Timestamp("1970-01-01")
MULTPL_EY_URL = "https://www.multpl.com/s-p-500-earnings-yield/table/by-month"
CACHE_FILE = "macro_yields.parquet"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def _to_month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index).tz_localize(None)
    return idx + pd.offsets.MonthEnd(0)


def fetch_treasury_history(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly 10-year US Treasury yield (%) from Yahoo Finance (^TNX), clipped to `start` onward."""
    hist = yf.download("^TNX", period="max", interval="1d", progress=False, auto_adjust=False)["Close"]
    series = hist.iloc[:, 0] if isinstance(hist, pd.DataFrame) else hist
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.resample("ME").last().dropna()
    series = series[series.index >= start]
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
