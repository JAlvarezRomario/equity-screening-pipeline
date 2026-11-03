"""The long per-share record: share count and dividends per share by calendar year (`Shares_Hist`, `DPS_Hist`, JSON
from `fetch.yearly_shares` / `fetch.yearly_dividends`), longer than the four or five annual reports Yahoo serves.
Shown on the company page, never scored."""

import json

import numpy as np
import pandas as pd

from fq.settings import Settings

DIVIDEND_CUT = 0.95  # a year paying under 95% of the year before is a cut (timing and rounding pass)
MIN_SHARE_YEARS = 5  # years of share counts the 10-year change needs


def parse_history(value) -> pd.Series:
    """A stored JSON history as a float Series indexed by calendar year (int), oldest first; empty when absent."""
    if not isinstance(value, str) or not value:
        return pd.Series(dtype=float)
    try:
        data = json.loads(value)
    except ValueError:
        return pd.Series(dtype=float)
    s = pd.Series({int(k): float(v) for k, v in data.items() if v is not None}, dtype=float).sort_index()
    return s[s > 0]


def _cagr(s: pd.Series, years: int) -> float:
    """%/yr between the latest year and the one `years` before it, both present."""
    if s.empty or (start := s.index.max() - years) not in s.index:
        return np.nan
    return ((s.iloc[-1] / s[start]) ** (1 / years) - 1) * 100


def _shares_change(s: pd.Series) -> float:
    """%/yr log-CAGR of the share count over up to 10 years back from the latest (at least `MIN_SHARE_YEARS`)."""
    s = s[s.index >= s.index.max() - 10] if len(s) else s
    if len(s) < MIN_SHARE_YEARS:
        return np.nan
    span = s.index.max() - s.index.min()
    return (np.exp(np.log(s.iloc[-1] / s.iloc[0]) / span) - 1) * 100


def _cut(s: pd.Series, year: int) -> bool:
    """`year` paid under `DIVIDEND_CUT` of both of the two years before, and the next year does not make it up: a
    payment that slipped across the year end (KO paid five times in one year, so the next looked cut) is no cut."""
    if year - 1 not in s.index:
        return False
    before = s[year - 1]
    if s[year] >= DIVIDEND_CUT * min(before, s.get(year - 2, before)):
        return False
    return not (year + 1 in s.index and s[year] + s[year + 1] >= DIVIDEND_CUT * 2 * before)


def _years_no_cut(s: pd.Series, last_year: float) -> float:
    """Consecutive years back from `last_year` (the last complete year when fetched) that paid a dividend without a
    cut (`_cut`); 0 when `last_year` paid nothing."""
    if s.empty or pd.isna(last_year):
        return np.nan
    years = 0
    for year in range(int(last_year), s.index.min() - 1, -1):
        if year not in s.index:
            break
        years += 1
        if _cut(s, year):
            break
    return float(years)


def long_history(df: pd.DataFrame, cfg: Settings) -> None:
    """`Shares_Change_10Y`, `Dividend_Years_No_Cut`, `DPS_Growth_10Y`, `DPS_Growth_20Y` (display only)."""
    empty = pd.Series([None] * len(df), index=df.index, dtype=object)
    shares = df.get("Shares_Hist", empty).map(parse_history)
    dps = df.get("DPS_Hist", empty).map(parse_history)
    # Dividends are stored for the complete years before the fetch (`fetch.yearly_dividends`).
    last_year = pd.to_datetime(df.get("Fetched_At", empty), errors="coerce").dt.year - 1
    df["Shares_Change_10Y"] = shares.map(_shares_change).astype(float)
    df["Dividend_Years_No_Cut"] = pd.Series(
        [_years_no_cut(s, y) for s, y in zip(dps, last_year, strict=True)], index=df.index, dtype=float
    )
    paying = df["Dividend_Years_No_Cut"] > 0  # a dividend stopped years ago has no growth to show
    df["DPS_Growth_10Y"] = dps.map(lambda s: _cagr(s, 10)).astype(float).where(paying)
    df["DPS_Growth_20Y"] = dps.map(lambda s: _cagr(s, 20)).astype(float).where(paying)
