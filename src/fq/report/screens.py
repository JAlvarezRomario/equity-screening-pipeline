"""The front page's list screens and their columns (no markup here). Not a company-page input
(`company_page.PAGE_INPUTS`): editing a list re-renders no company page; what a page shows of the lists travels
in its row (`On_Lists`, `Industry_Leader`, `In_Best_Fraction`)."""

from functools import cache

import numpy as np
import pandas as pd

from fq.metrics.capital import _to_value_base
from fq.metrics.normalise import normalized_fcf_yield
from fq.ranking import UNTRUSTED_FLAGS, comparable, top_n
from fq.report.format import DASH, NA, benchmark, fmt
from fq.report.links import slug
from fq.settings import GAINS_INSIDE_MODELS, Settings
from fq.universe import _read_lines, industry_key, is_cyclical_industry

# The top 20 Value, Quality and combined lists on the index page: the same facts as the other lists (user's choice,
# Oct 21, 2026: every list shows how many analysts follow the company).
TOP_LIST_COLUMNS = [
    ("Market Cap (B$)", "Market Cap", ",.1f", 1, ""),
    ("Above 52w Low", "Above_52W_Low", ".0f", 1, "%"),
    ("Analysts", "numberOfAnalystOpinions", ".0f", 1, ""),
    ("Dividend WHT", "Dividend_WHT", "g", 1, "%"),
    ("Value Score", "Value_Score", ".0f", 1, ""),
    ("Quality Score", "Quality_Score", ".0f", 1, ""),
]  # fmt: skip

# Repeaters (`highlights.repeater_list`): how many lists, then the usual facts.
REPEATER_COLUMNS = [("Lists", "Repeat_Count", ".0f", 1, ""), *TOP_LIST_COLUMNS]

# The top Value and Quality lists add the numbers behind the score (a percentile alone says little at the top).
VALUE_LIST_COLUMNS = [*TOP_LIST_COLUMNS, ("Owner Yield", "Owner_Yield", ".1f", 1, "%")]
QUALITY_LIST_COLUMNS = [
    *TOP_LIST_COLUMNS,
    ("Return Quality", "Return_Quality", ".1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
]  # fmt: skip


# Per-industry medians (index page).
INDUSTRY_COLUMNS = [
    ("FCF Yield Norm", "FCFy_Norm", ".1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
]  # fmt: skip


def industry_averages(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> dict:
    """Medians of `INDUSTRY_COLUMNS` (robust: one ROIC at the cap does not lift a small industry) per Yahoo industry over every benchmark company (the run itself when
    it has no benchmark), ranked models only, highest normalised FCF yield first (index page; user's choice,
    Oct 21, 2026). `best` is the industry's best combined rank in the run. Returns {"headers", "rows", "scope"}."""
    use_bench = bench is not None and not bench.empty and "industry" in bench.columns
    ref = bench if use_bench else df
    ref = ref[comparable(ref) & ref.get("industry", pd.Series(index=ref.index, dtype=object)).notna()]
    run = df[df.get("industry", pd.Series(index=df.index, dtype=object)).notna()]
    ranked = run.dropna(subset=["Rank"]) if "Rank" in run else run.iloc[0:0]
    best_of = ranked.sort_values("Rank").groupby("industry")["Ticker"].first()
    rows = []
    for industry, group in ref.groupby("industry"):
        means = [pd.to_numeric(group[col], errors="coerce").median() for _, col, *_ in INDUSTRY_COLUMNS]
        best = best_of.get(industry)
        specs = [spec for _, _, *spec in INDUSTRY_COLUMNS]
        cells = [fmt(m, *spec).replace(NA, DASH) for m, spec in zip(means, specs, strict=True)]
        rows.append({"industry": industry, "n": len(group), "sort": means[0], "cells": cells,
                     "best": best, "slug": slug(best) if best else None})  # fmt: skip
    rows.sort(key=lambda r: (-r["sort"] if pd.notna(r["sort"]) else np.inf, r["industry"]))
    headers = ["Industry", "Companies", *[f"{c[0]} (median)" for c in INDUSTRY_COLUMNS], "Best ranked"]
    return {"headers": headers, "rows": rows, "scope": (benchmark(df) or ("benchmark",))[0] if use_bench else None}


# The Compounders list (`ranking.compounder`, "Mastercard-like"): the traits that put a company on it, then the usual
# facts (user's choice, Oct 21, 2026).
COMPOUNDER_COLUMNS = [
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def in_best_fraction(df: pd.DataFrame) -> pd.Series:
    """In the best `Settings.highlight_top_fraction` of the combined ranking (`Rank` of the run's ranked companies), as
    every highlight list but "Best businesses" asks. `html.build_site` stores it as `In_Best_Fraction` on the whole
    run, so a subset (the zero-withholding rows) keeps the run's cut instead of a cut of its own. Without a `Rank`
    column there is no ranking to cut, and every row is kept."""
    if "In_Best_Fraction" in df.columns:
        return df["In_Best_Fraction"].fillna(False).astype(bool)
    if "Rank" not in df.columns:
        return pd.Series(True, index=df.index)
    rank = pd.to_numeric(df["Rank"], errors="coerce")
    return rank <= np.ceil(Settings().highlight_top_fraction * rank.notna().sum())


def compounder_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Flagged `Compounder` companies in the best fraction of the combined ranking, ordered by combined rank."""
    if "Compounder" not in df.columns:
        return df.head(0)
    d = df[(df["Compounder"] == True) & in_best_fraction(df)]  # noqa: E712
    if "Rank" in d.columns:
        return d.sort_values("Rank", ascending=True).head(n)
    if "Value_Score" in d.columns:
        return d.sort_values("Value_Score", ascending=False).head(n)
    return d.head(n)


# The Asset plays list (`ranking.asset_play`): tangible and common book bargains exceeding 100% of price,
# ordered by combined rank.
ASSET_PLAY_COLUMNS = [
    ("NAV / Price", "Effective_NAV_to_Price", ".0f", 1, "%"),
    ("Book Compounding / yr", "Book_Per_Share_Compounding", "+.1f", 1, "%"),
    ("Net Cash / Cap", "Net_Cash_to_Cap", "+.0f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def asset_play_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Flagged `Asset_Play` companies in the best fraction of the combined ranking, ordered by combined rank."""
    if "Asset_Play" not in df.columns:
        return df.head(0)
    d = df[(df["Asset_Play"] == True) & in_best_fraction(df)]  # noqa: E712
    if "Rank" in d.columns:
        return d.sort_values("Rank", ascending=True).head(n)
    sort_cols = [c for c in ("Asset_Play_Score", "Value_Score") if c in d.columns]
    return d.sort_values(sort_cols, ascending=False).head(n) if sort_cols else d.head(n)


# The Cannibals list (`ranking.cannibal`): companies retiring shares (>7%/yr) from their own free cash flow,
# ordered by combined rank.
CANNIBAL_COLUMNS = [
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    ("Cash Return Yield", "DirectCashtoInvestors_Yield_Avg", ".1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]


def cannibal_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Flagged `Cannibal` companies in the best fraction of the combined ranking, ordered by combined rank."""
    if "Cannibal" not in df.columns:
        return df.head(0)
    d = df[(df["Cannibal"] == True) & in_best_fraction(df)]  # noqa: E712
    if "Rank" in d.columns:
        return d.sort_values("Rank", ascending=True).head(n)
    if "Share_Reduction_Rate" in d.columns:
        return d.sort_values("Share_Reduction_Rate", ascending=False).head(n)
    return d.head(n)


# The "Off the radar" list: what put a company on it, then the usual facts (user's choice, Oct 21, 2026).
OFF_RADAR_COLUMNS = [
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def off_radar_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Small, little-followed companies outside the US with solid Value, Quality and ROIC, smooth per-share revenue
    and owner earnings compounding, shares not meaningfully rising, and recent numbers held, in combined rank order
    (`Settings.off_radar_*`). Leaves out financials and real estate, cyclicals and companies earning well above their
    usual margin (`Peak_Margin_Ratio` > `peak_margin_ratio`, for every model, not only cyclicals), and keeps only
    companies `in_best_fraction` of the combined ranking; `ranking.top_n` leaves out suspect, thin or distressed
    data. Display only."""
    s = Settings()
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    text = lambda col: df[col] if col in df.columns else pd.Series(None, index=df.index, dtype=object)  # noqa: E731
    held = (
        ~(num("Revenue_Change_Recent") < s.dislocation_revenue_change_min)
        & ~(num("EBIT_Change_Recent") < s.dislocation_ebit_change_min)
        & ~(num("EPS_Rev_90d") < s.estimate_cut_pct)
    )
    keep = (
        in_best_fraction(df)
        & (text("country") != "United States")
        & ~text("sector").isin(s.off_radar_sectors_out)
        & (text("Cyclical") != True)  # noqa: E712 - handles NaN / numpy bool
        & ~(num("Peak_Margin_Ratio") > s.peak_margin_ratio)
        & (num("numberOfAnalystOpinions").fillna(0) <= s.off_radar_max_analysts)  # unknown: Yahoo seldom covers
        & (num("Market Cap") < s.off_radar_max_cap)
        & (num("Value_Score") >= s.off_radar_min_value)
        & (num("Quality_Score") >= s.off_radar_min_quality)
        & (num("ROIC_Norm") >= s.off_radar_min_roic)
        & (num("Per_Share_Compounding") >= s.off_radar_min_compounding)
        & ~(num("Owner_Earnings_Compounding") < s.off_radar_min_oe_growth)
        & (num("Share_Reduction_Rate") >= s.off_radar_min_share_reduction)
        & held
    )
    return top_n(df[keep], "Rank", n)


# The "Better every year" list: what put a company on it, then the usual facts (user's choice, Oct 28, 2026).
IMPROVER_COLUMNS = [
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def _every_year(df: pd.DataFrame, concept: str, periods: list[str], min_years: int, rising: bool = True) -> pd.Series:
    """Over the consecutive annual reports from Y0 back (at least `min_years`), `concept` rose every year (`rising`),
    or never rose. A missing Y0 or a gap after fewer than `min_years` fails."""
    values = pd.DataFrame(
        {p: pd.to_numeric(df[f"{concept}_{p}"], errors="coerce") if f"{concept}_{p}" in df.columns else np.nan
         for p in periods}, index=df.index,
    ).to_numpy(dtype=float)  # fmt: skip
    run = np.cumprod(~np.isnan(values), axis=1).astype(bool)  # present without a gap from Y0
    newer, older = values[:, :-1], values[:, 1:]
    step = newer > older if rising else newer <= older
    ok = np.where(run[:, 1:], step, True).all(axis=1) & (run.sum(axis=1) >= min_years)
    return pd.Series(ok, index=df.index)


def improver_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Companies better every year (`Settings.improver_min_years` annual reports at least, audited, so TTM's overlap
    with Y0 is left out): owner earnings per share, revenue, ROIC (ROE for financials) and operating margin (not for
    banks, insurers and holdings, where it says little) each rose every year, the share count never rose, and the
    latest owner earnings are positive; in combined rank order, trusted as `ranking.top_n`. Statements in their own
    currency (an exchange rate is not the business). Display only."""
    s = Settings()
    years, k = s.annual_years, s.improver_min_years
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    owner = pd.DataFrame({f"OE_PS_{p}": num(f"Owner_Earnings_{p}") / num(f"Ordinary Shares Number_{p}").where(
        lambda x: x > 0) for p in years}, index=df.index)  # fmt: skip
    financial = df["Financial_Mode"].fillna(False).astype(bool) if "Financial_Mode" in df.columns else False
    keep = (
        _every_year(owner, "OE_PS", years, k)
        & (owner[f"OE_PS_{s.latest}"] > 0)
        & _every_year(df, "Total Revenue", years, k)
        & _every_year(df, "ROIC", years, k)
        & (_every_year(df, "operatingMargins", years, k) | financial)
        & _every_year(df, "Ordinary Shares Number", years, k, rising=False)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


# The "Dividend carry" list: what put a company on it, then the usual facts (user's choice, Oct 29, 2026).
CARRY_COLUMNS = [
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def _trending_up(per_share: pd.DataFrame, min_years: int, max_dips: int) -> pd.Series:
    """Over the consecutive annual reports from the first column (Y0) back, every value positive, at least `min_years`
    of them, the newest above the oldest and at most `max_dips` year-on-year falls (a tendency, not a straight line)."""
    values = per_share.to_numpy(dtype=float)
    run = np.cumprod(~np.isnan(values), axis=1).astype(bool)  # present without a gap from Y0
    n = run.sum(axis=1)
    clean = np.where(run, values, np.nan)
    oldest = clean[np.arange(len(values)), np.maximum(n - 1, 0)]
    dips = np.where(run[:, 1:], clean[:, :-1] < clean[:, 1:], False).sum(axis=1)
    positive = np.where(run, values > 0, True).all(axis=1)
    return pd.Series((n >= min_years) & positive & (clean[:, 0] > oldest) & (dips <= max_dips), index=per_share.index)


def dividend_carry_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Dividend payers a non-resident keeps whole (`Dividend_WHT` 0): yield at least `Settings.carry_min_yield` and above
    `carry_borrow_rate`, covered by the owner yield (what is left is retained and grows the value), dividends and
    owner earnings per share trending up over the annual reports (own currency), `in_best_fraction` of the combined
    ranking, not cyclical and not at peak margins; trusted as `ranking.top_n`, in combined rank order.
    Display only."""
    s = Settings()
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    shares = {p: num(f"Ordinary Shares Number_{p}").where(lambda x: x > 0) for p in s.annual_years}
    dividends = pd.DataFrame({p: -num(f"Cash Dividends Paid_{p}") / shares[p] for p in s.annual_years})
    owner = pd.DataFrame({p: num(f"Owner_Earnings_{p}") / shares[p] for p in s.annual_years})
    yield_ = num("Dividend_Yield")
    keep = (
        (num("Dividend_WHT") == 0)
        & (yield_ >= max(s.carry_min_yield, s.carry_borrow_rate))
        & (num("Owner_Yield") > yield_)
        & _trending_up(dividends, s.carry_min_years, s.carry_max_dips)
        & _trending_up(owner, s.carry_min_years, s.carry_max_dips)
        & in_best_fraction(df)
        & (df["Cyclical"] != True if "Cyclical" in df.columns else True)  # noqa: E712 - handles NaN / numpy bool
        & ~(num("Peak_Margin_Ratio") > s.peak_margin_ratio)
    )
    return top_n(df[keep], "Rank", n)


# The "Net cash" list: what put a company on it, then the usual facts (user's choice, Oct 29, 2026).
NET_CASH_COLUMNS = [
    ("Net Cash / Cap", "Net_Cash_to_Cap", "+.0f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def net_cash_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Companies whose net cash is at least `Settings.net_cash_min_pct` of their market cap, shares not issued and
    reported free cash flow positive in most years (the pile neither diluted nor burned), `in_best_fraction` of the
    combined ranking; trusted as `ranking.top_n`, in combined rank order. Cyclicals stay: the cash is what carries
    them through the downturn (the pill says so). Display only."""
    s = Settings()
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    keep = (
        (num("Net_Cash_to_Cap") >= s.net_cash_min_pct)
        & (num("Share_Reduction_Rate") >= s.net_cash_min_share_reduction)
        & ~(num("FCF_Positive_Share") < s.net_cash_min_fcf_pos_share)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


# The "Shortlist 100" list (user's choice, Oct 29, 2026): the hand-off for research in depth, every diagnostic the
# cache already holds applied at once. Display only; the thresholds live here, not in `Settings`, so the method
# fingerprint (`evaluate.METHOD_MODULES`) and every snapshot's method group stay as they are.
SHORTLIST_N = 100
SHORTLIST_MIN_FCF_POS = 75.0  # % of years with positive reported free cash flow: the profit turns into cash
SHORTLIST_MAX_BENEISH = -1.78  # Beneish M-score above it is the usual earnings-manipulation warning
SHORTLIST_MAX_ACCRUALS = 5.0  # (net income - operating cash flow) / assets, %: earnings well ahead of cash
SHORTLIST_MIN_INCREMENTAL_ROIC = 10.0  # % on the capital added: growth that earns its keep
SHORTLIST_MAX_NET_DEBT_EBIT = 3.0  # operating companies: net debt / EBIT, x ...
SHORTLIST_MIN_INTEREST_COVER = 5.0  # ... or interest covered this many times
SHORTLIST_MIN_EQUITY_ASSETS = 6.0  # banks, insurers, holdings: common equity / assets, % (as "Below book")
SHORTLIST_MIN_ROIC = 12.0  # normalised ROIC (ROE for financials), %
SHORTLIST_MIN_VALUE = 50.0  # score floors: bind little in the benchmark, guard a short run's best fifth
SHORTLIST_MIN_QUALITY = 60.0
# Stress tests (user's choice, Oct 29, 2026): AI disruption, a global recession, a rate-hike / inflation shock.
SHORTLIST_MAX_NET_DEBT_EQUITY = 50.0  # operating companies: net debt / common equity, %; net cash passes
SHORTLIST_LEADER_SHARE = 0.10  # an industry leader: in the top tenth of its Yahoo industry by size ...
SHORTLIST_LEADER_MIN = 3  # ... and the top 3 always count, however small the industry
# Greenblatt's bar: an after-tax earnings yield is worth having only above the 10-year Treasury, and never below 6%.
BOND_HURDLE_FLOOR = 6.0


def bond_hurdle(treasury: float | None) -> float:
    """The yield a business must beat, %: the 10-year Treasury yield, at least `BOND_HURDLE_FLOOR`."""
    t = pd.to_numeric(treasury, errors="coerce")
    return BOND_HURDLE_FLOOR if pd.isna(t) else max(BOND_HURDLE_FLOOR, float(t))


@cache
def _industry_keys(name: str) -> frozenset[str]:
    """Industries of a Shortlist list file (`lists/shortlist_*.txt`), normalised as `universe.industry_key`."""
    return frozenset(industry_key(i) for i in _read_lines(name))


def _in_industries(df: pd.DataFrame, name: str) -> pd.Series:
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    keys = _industry_keys(name)
    return industry.map(lambda i: isinstance(i, str) and industry_key(i) in keys).astype(bool)


# The "Nectar" list (user's design, Nov 2, 2026): what put a company on it, then the usual facts.
NECTAR_COLUMNS = [
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip
NECTAR_MIN_YIELD = 6.0  # owner earnings yield, %
NECTAR_MAX_NEW_SHARES = 1.005  # latest share count / the last annual report's: above this, shares were issued
NECTAR_STABLE_ROIC = 20.0  # normalised ROIC above this may stay flat instead of rising, if steady
NECTAR_MAX_OWNER_SPREAD = 0.2  # owner earnings per share: spread around their rising trend line / their mean
NECTAR_MAX_ROIC_VARIABILITY = 0.1  # `ROIC_Variability`, spread around its trend: about the median company's
NECTAR_FLAT = 0.02  # a year within this share of the year before is flat, not a fall (user's choice, Nov 2, 2026)


def _up_but_one_flat(df: pd.DataFrame, concept: str, periods: list[str], min_years: int) -> pd.Series:
    """As `_every_year`, but one year may be flat (`NECTAR_FLAT`) instead of up; a real fall fails."""
    values = pd.DataFrame(
        {p: pd.to_numeric(df[f"{concept}_{p}"], errors="coerce") if f"{concept}_{p}" in df.columns else np.nan
         for p in periods}, index=df.index,
    ).to_numpy(dtype=float)  # fmt: skip
    run = np.cumprod(~np.isnan(values), axis=1).astype(bool)
    newer, older = values[:, :-1], values[:, 1:]
    up = np.where(run[:, 1:], newer > older, True)
    flat = np.where(run[:, 1:], newer >= older - np.abs(older) * NECTAR_FLAT, True)
    ok = flat.all(axis=1) & ((~up).sum(axis=1) <= 1) & (run.sum(axis=1) >= min_years)
    return pd.Series(ok, index=df.index)


def _owner_earnings_smooth(per_share: pd.DataFrame, min_years: int) -> pd.Series:
    """Owner earnings per share (columns newest first) positive in every annual report, at least `min_years` of them,
    on a rising trend line and within `NECTAR_MAX_OWNER_SPREAD` of it."""

    def smooth(row: np.ndarray) -> bool:
        values = row[~np.isnan(row)]
        if len(values) < min_years or (values <= 0).any():
            return False
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        return bool(slope > 0 and np.std(values - (intercept + slope * x)) / values.mean() <= NECTAR_MAX_OWNER_SPREAD)

    rows = per_share.to_numpy(dtype=float)[:, ::-1]
    return pd.Series([smooth(row) for row in rows], index=per_share.index, dtype=bool)


def nectar_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Durable, cheap and boring: in the best fifth, net cash, owner yield >= `NECTAR_MIN_YIELD`, not cyclical, owner earnings per share up in every annual report but at most one flat year,
    ROIC (ROE for financials) the same or, normalised above `NECTAR_STABLE_ROIC`, merely steady, revenue (own currency)
    up in every one and the share count never up (nor since the last report), over at least `Settings.improver_min_years` of them, owner earnings and ROIC smooth, outside the industries of
    `nectar_avoid.txt` and `shortlist_ai_fragile.txt`. Combined rank order, trusted as `ranking.top_n`. Display only."""
    s = Settings()
    years, k = s.annual_years, s.improver_min_years
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    owner = pd.DataFrame({p: num(f"Owner_Earnings_{p}") / num(f"Ordinary Shares Number_{p}").where(lambda x: x > 0)
                          for p in years}, index=df.index)  # fmt: skip
    rising = owner.rename(columns=lambda p: f"OE_PS_{p}")
    cyclical = df["Cyclical"].fillna(False).astype(bool) if "Cyclical" in df.columns else False
    keep = (
        (num("Owner_Yield") >= NECTAR_MIN_YIELD)
        & (num("Net_Cash_to_Cap") > 0)
        & ~cyclical
        & ~_in_industries(df, "nectar_avoid.txt")
        & ~_in_industries(df, "shortlist_ai_fragile.txt")
        & (num("ROIC_Variability") <= NECTAR_MAX_ROIC_VARIABILITY)
        & _up_but_one_flat(rising, "OE_PS", years, k)
        & (_up_but_one_flat(df, "ROIC", years, k) | (num("ROIC_Norm") > NECTAR_STABLE_ROIC))
        & _every_year(df, "Total Revenue", years, k)
        & _every_year(df, "Ordinary Shares Number", years, k, rising=False)
        & ~(num("Ordinary Shares Number_TTM") > num(f"Ordinary Shares Number_{s.latest}") * NECTAR_MAX_NEW_SHARES)
        & _owner_earnings_smooth(owner, k)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


def _latest_value(df: pd.DataFrame, concept: str) -> pd.Series:
    """TTM, else the latest annual report (`Y0`)."""
    out = pd.Series(np.nan, index=df.index)
    for period in ("TTM", "Y0"):
        if f"{concept}_{period}" in df.columns:
            out = out.fillna(pd.to_numeric(df[f"{concept}_{period}"], errors="coerce"))
    return out


def industry_leader(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.Series:
    """Among the largest of its Yahoo industry in `reference` (default `df`; the benchmark, so a company is a leader
    in every run or none): the top `SHORTLIST_LEADER_SHARE` of the industry, the top `SHORTLIST_LEADER_MIN` always.
    Size is USD revenue, total assets for banks (a lender's revenue moves with rates). A company without the figure
    or the industry is not a leader."""
    ref = df if reference is None else reference
    bank = (ref["Business_Model"] == "bank") if "Business_Model" in ref.columns else pd.Series(False, index=ref.index)
    size = _latest_value(ref, "Total Revenue").where(~bank, _latest_value(ref, "Total Assets"))
    industry = ref["industry"] if "industry" in ref.columns else pd.Series(None, index=ref.index, dtype=object)
    rank = size.groupby(industry).rank(ascending=False, method="min")
    count = size.groupby(industry).transform("count")
    leader = rank <= np.maximum(SHORTLIST_LEADER_MIN, np.ceil(SHORTLIST_LEADER_SHARE * count))
    by_ticker = pd.Series(leader.to_numpy(), index=ref["Ticker"].to_numpy())
    by_ticker = by_ticker[~by_ticker.index.duplicated()]
    return df["Ticker"].map(by_ticker).fillna(False).astype(bool)


INSIDERS_COLUMN = ("Insiders", "heldPercentInsiders", ".0f", 100, "%")
SHORTLIST_COLUMNS = [
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Incremental ROIC", "Incremental_ROIC", ".0f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    INSIDERS_COLUMN,
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _net_debt_within_equity(df: pd.DataFrame) -> pd.Series:
    """Net debt at most `SHORTLIST_MAX_NET_DEBT_EQUITY` % of common equity (net cash and a blank pass, net debt with
    negative equity fails). Meant for operating companies: a lender's debt is its business."""
    net_debt = _num(df, "Net_Debt")
    equity = _latest_value(df, "Common Stock Equity").fillna(_latest_value(df, "Stockholders Equity"))
    return ~(net_debt > 0) | ~(
        net_debt * 100 > SHORTLIST_MAX_NET_DEBT_EQUITY * equity.where(equity.isna() | (equity > 0), 0)
    )


def _shortlist_gates(df: pd.DataFrame, hurdle: float = BOND_HURDLE_FLOOR) -> list[tuple[str, pd.Series]]:
    """The Shortlist's conditions in funnel order, each (label, mask). A blank figure passes, as in every list: the
    operating tests are blank for banks, insurers and holdings, which get the equity / assets floor instead. `hurdle`
    (`bond_hurdle`) is the owner yield to beat."""
    s = Settings()
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    financial = df["Financial_Mode"] if "Financial_Mode" in df.columns else pd.Series(False, index=df.index)
    financial = financial.fillna(False).astype(bool)
    operating_debt = ~(num("NetDebt_to_EBIT") > SHORTLIST_MAX_NET_DEBT_EBIT) | (
        num("Interest_Coverage") >= SHORTLIST_MIN_INTEREST_COVER
    )
    leader = (
        df["Industry_Leader"].fillna(False).astype(bool) if "Industry_Leader" in df.columns else industry_leader(df)
    )
    bank = (df["Business_Model"] == "bank") if "Business_Model" in df.columns else pd.Series(False, index=df.index)
    cyclical = df["Cyclical"].fillna(False).astype(bool) if "Cyclical" in df.columns else False
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    recession_prone = ~financial & (
        _in_industries(df, "shortlist_recession_industries.txt") | industry.map(is_cyclical_industry).astype(bool)
        | cyclical
    )  # fmt: skip
    net_debt = num("Net_Debt")
    net_debt_ok = _net_debt_within_equity(df)
    return [
        ("growing per share, no dilution",
         ~(num("Per_Share_Compounding") < 0) & ~(num("Share_Reduction_Rate") < s.net_cash_min_share_reduction)),
        ("owner earnings not shrinking", ~(num("Owner_Earnings_Compounding") < 0)),
        ("free cash flow in most years", ~(num("FCF_Positive_Share") < SHORTLIST_MIN_FCF_POS)),
        ("clean earnings",
         ~(num("Beneish_M") > SHORTLIST_MAX_BENEISH) & ~(num("Accruals") > SHORTLIST_MAX_ACCRUALS)),
        ("growth earns its keep", ~(num("Incremental_ROIC") < SHORTLIST_MIN_INCREMENTAL_ROIC)),
        ("sound balance sheet",
         (operating_debt & ~financial) | (~(num("Equity_to_Assets") < SHORTLIST_MIN_EQUITY_ASSETS) & financial)),
        ("nothing breaking now",
         ~(num("EPS_Rev_90d") < s.estimate_cut_pct)
         & ~(num("Revenue_Change_Recent") < s.dislocation_revenue_change_min)
         & ~(num("EBIT_Change_Recent") < s.dislocation_ebit_change_min)),
        ("not fragile to AI", ~_in_industries(df, "shortlist_ai_fragile.txt")),
        ("recession: leaders only", ~(recession_prone | bank) | (leader & (bank | (net_debt <= 0)))),
        ("net debt within half of equity", net_debt_ok | financial),
        ("beats the bond", ~(num("Owner_Yield") < hurdle)),
        ("durable returns",
         (num("ROIC_Norm") >= SHORTLIST_MIN_ROIC) & (num("Value_Score") >= SHORTLIST_MIN_VALUE)
         & (num("Quality_Score") >= SHORTLIST_MIN_QUALITY)),
    ]  # fmt: skip


def shortlist_list(df: pd.DataFrame, n: int = SHORTLIST_N, hurdle: float = BOND_HURDLE_FLOOR) -> pd.DataFrame:
    """The companies worth researching in depth: `in_best_fraction` of the combined ranking and every condition of
    `_shortlist_gates` (per-share growth without dilution, owner earnings not shrinking, cash-backed and clean
    earnings, growth that earns its keep, a sound balance sheet, nothing breaking now, not fragile to AI, recession
    leaders only, net debt within half of equity, an owner yield above the bond `hurdle`, durable returns); trusted
    as `ranking.top_n`, in combined rank order. Display only."""
    keep = in_best_fraction(df)
    for _, mask in _shortlist_gates(df, hurdle):
        keep &= mask
    return top_n(df[keep], "Rank", n)


def shortlist_funnel(df: pd.DataFrame, hurdle: float = BOND_HURDLE_FLOOR) -> list[tuple[str, int]]:
    """How many companies are left after each step of the Shortlist: ranked and trusted, the best fraction of the
    combined ranking, then each condition in turn (the last count is `len(shortlist_list(df, len(df)))`)."""
    keep = top_n(df, "Rank", len(df)).index
    left = df.index.isin(keep)
    steps = [("ranked and trusted", int(left.sum()))]
    left &= in_best_fraction(df).to_numpy()
    steps.append((f"best {Settings().highlight_top_fraction:.0%} combined", int(left.sum())))
    for label, mask in _shortlist_gates(df, hurdle):
        left &= mask.to_numpy(dtype=bool)
        steps.append((label, int(left.sum())))
    return steps


# The "Temporary trouble" list (Oct 30, 2026, from Greenblatt's class: Pzena's "value is created by deterioration"):
# good businesses whose profit just fell and has stopped falling, cheap on normal earnings. Every other list leaves
# them out (`ranking.UNTRUSTED_FLAGS` holds `Earnings_Collapse`; the others ask for recent numbers held). Brian Gains'
# other three kinds of hated company stay out: unsustainable (fragile to AI), bad balance sheet (`Distress`, net debt
# above half of equity) and dying (`Revenue_Decline`). Display only; constants here, as the Shortlist's.
TROUBLE_EBIT_DROP = -20.0  # %: operating profit, latest twelve months vs the year before (`EBIT_Change_Recent`)
TROUBLE_MIN_QUALITY = 50.0
# Pzena's "customers have not left": revenue now (TTM, else Y0) at least this share of the three annual reports
# before it. `Revenue_Decline` sees a slow shrink, not a sudden fall in demand or the end of a peak (Oct 30, 2026).
TROUBLE_MIN_REVENUE_HELD = 0.85
TROUBLE_COLUMNS = [
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("EBIT Change", "EBIT_Change_Recent", "+.0f", 1, "%"),
    ("Normal Yield", "Normal_Earnings_Yield", ".1f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    INSIDERS_COLUMN,
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def normal_earnings_yield(df: pd.DataFrame) -> pd.Series:
    """`EPV_Yield` as it would be without the cap of a lasting fall in profit (`Earnings_Collapse`): the same
    normalisation as `valuation._value` (mid-cycle for cyclicals, insurers and holdings), no TTM cap. Equal to
    `EPV_Yield` elsewhere, and where the inputs are not in the frame."""
    out = _num(df, "EPV_Yield")
    needed = ("Earnings_Collapse", "Value_Base", "Equity_Value", "FCF_Base", "Financial_Mode", "Business_Model",
              "Cyclical")  # fmt: skip
    if any(c not in df.columns for c in needed):
        return out
    collapse = df["Earnings_Collapse"].fillna(False).astype(bool)
    if not collapse.any():
        return out
    d = df[collapse]
    cyclical = d["Cyclical"].fillna(False).astype(bool) | d["Business_Model"].isin(GAINS_INSIDE_MODELS)
    uncapped = pd.Series(normalized_fcf_yield(d, Settings().fiscal_years, 1.0, cyclical, "FCF_Owner"), index=d.index)
    return out.where(~collapse, _to_value_base(d, uncapped).reindex(df.index))


def _clean_quarter(df: pd.DataFrame, q: str) -> pd.Series:
    """A stored quarter's pre-tax profit without unusual items."""
    return _num(df, f"Pretax Income_{q}") - _num(df, f"Total Unusual Items_{q}").fillna(0)


def _revenue_held(df: pd.DataFrame) -> pd.Series:
    """Revenue now (TTM, else Y0) at least `TROUBLE_MIN_REVENUE_HELD` x the mean of the three annual reports before
    it (Y0-Y2 beside a TTM, else Y1-Y3; two at least). A blank figure and financials pass (a lender's revenue says
    little, as its EBIT)."""
    ttm = _num(df, "Total Revenue_TTM")
    annual = [_num(df, f"Total Revenue_Y{i}") for i in range(4)]
    has_ttm = ttm.notna()
    now = ttm.where(has_ttm, annual[0])
    with_ttm, without = pd.concat(annual[:3], axis=1), pd.concat(annual[1:], axis=1)
    before = with_ttm.mean(axis=1).where(with_ttm.count(axis=1) >= 2)
    before = before.where(has_ttm, without.mean(axis=1).where(without.count(axis=1) >= 2))
    return ~(now < TROUBLE_MIN_REVENUE_HELD * before) | _flag(df, "Financial_Mode")


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool) if col in df.columns else pd.Series(False, index=df.index)


def _trouble_gates(df: pd.DataFrame, hurdle: float = BOND_HURDLE_FLOOR) -> list[tuple[str, pd.Series]]:
    """The Temporary trouble conditions in funnel order, each (label, mask); a blank figure passes unless stated."""
    financial, collapse = _flag(df, "Financial_Mode"), _flag(df, "Earnings_Collapse")
    ebit_change = _num(df, "EBIT_Change_Recent")
    # Banks, insurers and holdings: EBIT says little (BBVA read -129%), so only a lasting fall in clean pre-tax profit
    # (estimate cuts alone let in a dozen whose profit had not fallen, Oct 30, 2026).
    trouble = collapse | ((ebit_change < TROUBLE_EBIT_DROP) & ~financial)
    broken = pd.Series(False, index=df.index)
    for f in UNTRUSTED_FLAGS:  # `ranking.trusted`, less the fall this list looks for
        if f != "Earnings_Collapse":
            broken |= _flag(df, f)
    if "Model_Review" in df.columns:
        broken |= df["Model_Review"].notna()
    # Pzena's trough of stabilisation, year on year so a seasonal quarter does not read as a fall: the latest
    # quarter's change on the same quarter a year earlier no worse than the drop that put it here, or a rise.
    q0, q4 = _clean_quarter(df, "Q0"), _clean_quarter(df, "Q4")
    yoy = (q0 - q4) * 100 / q4.abs().where(q4.abs() > 0)
    # A lasting fall is at least `earnings_collapse_pct` deep, also where EBIT says nothing (banks, insurers, holdings).
    drop = ebit_change.where(~financial)
    drop = drop.where(~collapse, np.fmin(drop, -Settings().earnings_collapse_pct)).fillna(0).clip(upper=0)
    return [
        ("profit just fell", trouble),
        ("good business",
         (_num(df, "ROIC_Norm") >= SHORTLIST_MIN_ROIC) & (_num(df, "Quality_Score") >= TROUBLE_MIN_QUALITY)
         & ~(_num(df, "FCF_Positive_Share") < SHORTLIST_MIN_FCF_POS)),
        ("customers still there", _revenue_held(df)),
        ("not dying, distressed or fragile",
         ~broken & ~_in_industries(df, "shortlist_ai_fragile.txt") & (_net_debt_within_equity(df) | financial)),
        ("stopped falling", ~(yoy < drop)),
        ("cheap on normal earnings", normal_earnings_yield(df) >= hurdle),
    ]  # fmt: skip


def trouble_list(df: pd.DataFrame, n: int = 20, hurdle: float = BOND_HURDLE_FLOOR) -> pd.DataFrame:
    """Good businesses in temporary trouble (`_trouble_gates`), ranked companies only, in combined rank order, with
    `Normal_Earnings_Yield` added. Not cut to the best fraction: a lasting fall caps Value, which sinks them in the
    combined ranking. Display only."""
    if "Rank" not in df.columns:
        return df.head(0)
    keep = df["Rank"].notna()
    for _, mask in _trouble_gates(df, hurdle):
        keep &= mask
    d = df[keep]
    return d.assign(Normal_Earnings_Yield=normal_earnings_yield(d)).sort_values("Rank").head(n)


def trouble_funnel(df: pd.DataFrame, hurdle: float = BOND_HURDLE_FLOOR) -> list[tuple[str, int]]:
    """How many ranked companies are left after each Temporary trouble condition."""
    left = df["Rank"].notna().to_numpy(copy=True) if "Rank" in df.columns else np.zeros(len(df), dtype=bool)
    steps = [("ranked", int(left.sum()))]
    for label, mask in _trouble_gates(df, hurdle):
        left &= mask.to_numpy(dtype=bool)
        steps.append((label, int(left.sum())))
    return steps
