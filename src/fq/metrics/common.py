"""Helpers shared by the metric steps: column access, periods, TTM overlap, liquidity."""

import numpy as np
import pandas as pd

from fq.fx import MINOR_UNITS
from fq.settings import Settings

# Plain depreciation first: amortisation of acquired intangibles needs no capex to keep the business going.
DA_CONCEPTS = (
    "Depreciation", "Depreciation Amortization Depletion", "Depreciation And Amortization", "Reconciled Depreciation",
)  # fmt: skip


# TTM ending less than this many years after Y0 shares most of its quarters (one quarter later: three of
# four), so Y0 is left out of spreads and medians; half a year or more keeps it (`_distinct`).
TTM_SHARED_SPAN = 0.375


def _num(row: pd.Series, key: str) -> float:
    return pd.to_numeric(row.get(key), errors="coerce")


def _slot(period: str) -> int:
    """Y3 -> 3 (how many years before the latest annual report)."""
    return int(period[1:])


def _first_valid(df: pd.DataFrame, keys: list[str], default=np.nan) -> pd.Series:
    """Coalesce several columns left to right."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for key in keys:
        if key in df.columns:
            out = out.fillna(pd.to_numeric(df[key], errors="coerce"))
    return out.fillna(default) if not pd.isna(default) else out


def _latest(df: pd.DataFrame, name: str, cfg: Settings) -> pd.Series:
    """`name` in the newest period that has it: TTM, else the latest annual report, else the one before."""
    return _first_valid(df, [f"{name}_{p}" for p in ("TTM", cfg.latest, cfg.prior)])


def _reporting_currency(df: pd.DataFrame) -> pd.Series:
    """The statements' currency: Yahoo's `financialCurrency`, else the quote's (pence as pounds). Yahoo leaves the
    field out for some listings (most of Borsa Istanbul), and a blank read as no currency took their inflation out."""
    quote = df.get("currency", pd.Series(None, index=df.index)).map(
        lambda c: MINOR_UNITS.get(c, (c, 1))[0] if isinstance(c, str) else None
    )
    reported = df.get("financialCurrency", pd.Series(None, index=df.index))
    return reported.where(reported.map(lambda c: isinstance(c, str) and c != ""), quote)


def adjusted_r2(r2: float, n: int) -> float:
    """R² net of what chance gives a line through `n` points (1 - (1 - R²)(n - 1)/(n - 2), at least 0): on four points
    pure noise averages R² ~0.33, which weighed a slope a third of the way in; adjusted, about 0. A smooth trend
    keeps nearly all of it (R² 0.95 on four points: 0.93)."""
    if n <= 2:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - (1.0 - r2) * (n - 1) / (n - 2))))


def _col(df: pd.DataFrame, key: str, default=np.nan) -> pd.Series:
    if key in df.columns:
        return pd.to_numeric(df[key], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _depreciation(df: pd.DataFrame, period: str) -> pd.Series:
    """The period's depreciation, the first of `DA_CONCEPTS` with a positive value."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for name in DA_CONCEPTS:
        value = _col(df, f"{name}_{period}")
        out = out.fillna(value.where(value > 0))
    return out


def _liquidity(df: pd.DataFrame, period: str) -> pd.Series:
    """Cash plus short-term investments. A blank row means "none reported", not "unknown": one
    missing row must not void the rest."""
    cash = _col(df, f"Cash And Cash Equivalents_{period}", 0).fillna(0)
    st = _col(df, f"Short Term Investments_{period}", 0).fillna(0)
    ost = _col(df, f"Other Short Term Investments_{period}", 0).fillna(0)
    agg = _col(df, f"Cash Cash Equivalents And Short Term Investments_{period}", 0).fillna(0)
    return np.maximum(cash + st + ost, agg)


def _spare_cash(liquidity: pd.Series, revenue: pd.Series, cfg: Settings) -> pd.Series:
    """Cash beyond what the business needs to operate (`operating_cash_pct` of revenue), never negative."""
    return (liquidity - cfg.operating_cash_pct / 100 * revenue.clip(lower=0).fillna(0)).clip(lower=0)


def _dates(df: pd.DataFrame, period: str) -> pd.Series:
    col = f"PeriodEnd_{period}"
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _ttm_span(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Years between the latest annual report and the TTM end: the share of Y0 that TTM does NOT cover
    (0.25 = TTM shares three quarters with Y0). NaN when either date is unknown or TTM is not newer."""
    span = (_dates(df, "TTM") - _dates(df, cfg.latest)).dt.days / 365.25
    return span.where(span > 0).clip(upper=1.0)


def _distinct(df: pd.DataFrame, name: str, periods: list[str], cfg: Settings) -> pd.DataFrame:
    """`_frame` for spreads, shares and medians: Y0 is left out where TTM has a value and covers most of it.

    TTM ending one quarter after Y0 shares three of its quarters, so counting both weighs that year double
    and understates how much the business moves. With half a year or more between them, Y0 holds quarters
    TTM does not (CNC: most of its loss year) and stays.
    """
    frame = _frame(df, name, periods)
    if "TTM" in frame.columns and cfg.latest in frame.columns:
        mostly_shared = (_ttm_span(df, cfg) < TTM_SHARED_SPAN).fillna(False)
        frame[cfg.latest] = frame[cfg.latest].mask(mostly_shared & frame["TTM"].notna())
    return frame


def _distinct_mean(df: pd.DataFrame, name: str, periods: list[str], cfg: Settings) -> pd.Series:
    """Mean over `periods` where Y0 weighs only the share TTM does not cover (when TTM has a value), so the
    quarters both contain count once."""
    frame = _frame(df, name, periods)
    weights = pd.DataFrame(1.0, index=df.index, columns=frame.columns)
    if "TTM" in frame.columns and cfg.latest in frame.columns:
        weights[cfg.latest] = _ttm_span(df, cfg).where(frame["TTM"].notna()).fillna(1.0)
    weights = weights.where(frame.notna(), 0.0)
    total = weights.sum(axis=1)
    return (frame.fillna(0) * weights).sum(axis=1) / total.where(total > 0)


def _row_years(row: pd.Series, concept: str, years: list[str], latest: str) -> list[str]:
    """`years` for one company's history, without Y0 when TTM has `concept` and covers most of it (`_distinct`)."""
    ttm_end = pd.to_datetime(row.get("PeriodEnd_TTM"), errors="coerce")
    y0_end = pd.to_datetime(row.get(f"PeriodEnd_{latest}"), errors="coerce")
    if pd.isna(ttm_end) or pd.isna(y0_end) or pd.isna(_num(row, f"{concept}_TTM")):
        return years
    span = (ttm_end - y0_end).days / 365.25
    return [y for y in years if y != latest] if 0 < span < TTM_SHARED_SPAN else years


def _text(df: pd.DataFrame, key: str) -> pd.Series:
    return df[key] if key in df.columns else pd.Series(np.nan, index=df.index, dtype=object)


def _frame(df: pd.DataFrame, name: str, periods: list[str]) -> pd.DataFrame:
    """One column per period for a metric (missing columns as NaN)."""
    return pd.DataFrame({p: _col(df, f"{name}_{p}") for p in periods}, index=df.index)


def _log_slope_growth(values: pd.DataFrame) -> pd.Series:
    """Yearly growth (%) from a least-squares line through log values; columns = years before Y0.

    Uses every year, so one unusual start year (a post-COVID rebound, a cycle low) weighs less
    than in a two-point CAGR. Needs two positive values.
    """
    out = []
    for _, row in values.iterrows():
        v = row[row > 0]
        if len(v) < 2:
            out.append(np.nan)
            continue
        slope = np.polyfit(-v.index.to_numpy(dtype=float), np.log(v.to_numpy(dtype=float)), 1)[0]
        out.append((np.exp(slope) - 1) * 100)
    return pd.Series(out, index=values.index)
