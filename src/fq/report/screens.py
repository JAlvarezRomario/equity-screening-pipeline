"""The front page's list screens and their columns (no markup here). Not a company-page input
(`company_page.PAGE_INPUTS`): editing a list re-renders no company page; what a page shows of the lists travels
in its row (`On_Lists`, `Industry_Leader`, `In_Best_Fraction`)."""

from functools import cache

import numpy as np
import pandas as pd

from fq.fx import FALLBACK_BORROW_RATE
from fq.ranking import combined_order, rankable, top_n, trusted
from fq.report.dividend_record import COLUMNS as RECORD_COLUMNS
from fq.report.dividend_record import dividend_record
from fq.report.format import DASH, NA, benchmark, fmt
from fq.report.links import slug
from fq.settings import Settings
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
    it has no benchmark), ranked companies only (`rankable`), highest normalised FCF yield first (index page; user's choice,
    Oct 21, 2026). `best` is the industry's best combined rank in the run. Returns {"headers", "rows", "scope"}."""
    use_bench = bench is not None and not bench.empty and "industry" in bench.columns
    ref = bench if use_bench else df
    ref = ref[rankable(ref) & ref.get("industry", pd.Series(index=ref.index, dtype=object)).notna()]
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


# The kinds ranked apart (`ranking.rank_group`, user's choice, Nov 5, 2026), each judged against its own kind in a
# list of its own: banks and insurers (the "financial" group split by model), holdings and equity REITs.
KINDS = ("bank", "insurer", "holding", "real_estate")


def kind(df: pd.DataFrame) -> pd.Series:
    """The kind a company ranked apart is judged among (`KINDS`), else None (ranked, loss-making, too little data)."""
    group = df["Rank_Group"] if "Rank_Group" in df.columns else pd.Series(None, index=df.index, dtype=object)
    model = df["Business_Model"] if "Business_Model" in df.columns else pd.Series(None, index=df.index, dtype=object)
    out = group.where(group.isin({"holding", "real_estate"}), model.where(group == "financial"))
    return out.where(out.isin(KINDS), None)


def kind_order(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(`Kind_Rank`, `In_Kind_Best`): the combined ranking's order (`ranking.combined_order`: the worse of the Value
    and Quality positions) among the trusted, scored companies of each kind, and whether a company is in its kind's
    best `Settings.highlight_top_fraction`, as `in_best_fraction` asks of the ranked companies. `highlights.prepare`
    stores both on the run, so a subset keeps them."""
    if "Kind_Rank" in df.columns and "In_Kind_Best" in df.columns:
        return df["Kind_Rank"], df["In_Kind_Best"].fillna(False).astype(bool)
    kinds = kind(df)
    rank = pd.Series(np.nan, index=df.index)
    best = pd.Series(False, index=df.index)
    ok = trusted(df) & _num(df, "Value_Score").notna() & _num(df, "Quality_Score").notna()
    for k in KINDS:
        rows = ok & (kinds == k)
        if not rows.any():
            continue
        members = df[rows]
        value_pos = _num(members, "Value_Score").rank(method="min", ascending=False)
        quality_pos = _num(members, "Quality_Score").rank(method="min", ascending=False)
        order = combined_order(value_pos, quality_pos, len(members) + 1).rank(method="first")
        rank[members.index] = order
        best[members.index] = order <= np.ceil(Settings().highlight_top_fraction * len(members))
    return rank, best


def _dividend_held(df: pd.DataFrame, through_downturns: bool = False) -> pd.Series:
    """No dividend cut paid in the last 12 months (`Dividend_Cut_Recent`, `dividend_record` rule 6) and, with
    `through_downturns`, no fall of more than `RECESSION_MAX_FALL` % through a downturn on file (rule 7). A company
    without a record passes (user's choice, Nov 5, 2026: banks, insurers, REITs, energy and the Shortlist)."""
    record = with_dividend_record(df)
    held = ~_flag(record, "Dividend_Cut_Recent")
    if through_downturns:
        held &= ~(_num(record, "Recession_Dividend_Fall") < -RECESSION_MAX_FALL)
    return held


# The worst fall of the regular dividend through a downturn on file (`dividend_record` rule 7), as a list column.
RECESSION_COLUMN = ("Dividend in Downturns", "Recession_Dividend_Fall", "+.0f", 1, "%")


def _kind_list(df: pd.DataFrame, name: str, n: int, keep: pd.Series | None = None) -> pd.DataFrame:
    """Kind `name`'s list: its best fraction (`kind_order`), past `keep`, in its own combined order."""
    rank, best = kind_order(df)
    mask = (kind(df) == name) & best & (True if keep is None else keep)
    return top_n(df[mask].assign(Kind_Rank=rank[mask]), "Kind_Rank", n)


# "Banks and lenders" and "Insurers" (user's choice, Nov 5, 2026): judged against their own kind only (ROE, book,
# equity / assets), outside the rankings. Display only.
FINANCIAL_KIND_COLUMNS = [
    ("ROE Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Book / Price", "BV_to_Price", ".0f", 1, "%"),
    ("Equity / Assets", "Equity_to_Assets", ".1f", 1, "%"),
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Dividend in 2009", "Recession_Dividend_Fall", "+.0f", 1, "%"),  # 2020 is not judged for them: 2009 alone
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def bank_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Banks and lenders (model `bank`: banks, credit services, capital markets, mortgage finance, mortgage REITs) in
    the best fraction of their kind (`kind_order`), equity at least `SHORTLIST_MIN_EQUITY_ASSETS` % of assets
    (`_debt_bounded`), no dividend cut in the last 12 months (`_dividend_held`; the 2009 fall is shown, not tested:
    most lenders cut then, user's choice, Nov 5, 2026), trusted, in their kind's combined order. Display only."""
    return _kind_list(df, "bank", n, _debt_bounded(df) & _dividend_held(df))


def insurer_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Insurers (model `insurer`), as `bank_list`."""
    return _kind_list(df, "insurer", n, _debt_bounded(df) & _dividend_held(df))


# "Real estate" (user's choice, Nov 5, 2026): equity REITs against each other, on owner yield (close to their
# cash-earnings yield, AFFO) and returns, with book beside it but never a test (cost under US GAAP, fair value under
# IFRS). A REIT at half its book with interest barely covered is a debt trap, so the debt is bounded. Display only.
REIT_MIN_INTEREST_COVER = 2.5  # times: as the asset plays' cover
REIT_MAX_NET_DEBT_EBIT = 8.0  # x: property is financed long; a REIT above this is a bet on its lenders
REAL_ESTATE_COLUMNS = [
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Book / Price", "BV_to_Price", ".0f", 1, "%"),
    ("Interest Cover", "Interest_Coverage", ".1f", 1, "x"),
    ("Net Debt / EBIT", "NetDebt_to_EBIT", ".1f", 1, "x"),
    RECESSION_COLUMN,
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def real_estate_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Equity REITs (`lists/real_estate_owners.txt`) in the best fraction of their kind, owner yield positive,
    interest covered `REIT_MIN_INTEREST_COVER` times and net debt at most `REIT_MAX_NET_DEBT_EBIT` x EBIT (a blank
    passes, as in every list), the dividend held through the downturns and this year (`_dividend_held`: a REIT pays
    out most of its income, so its dividend follows the business; user's choice, Nov 5, 2026), trusted, in their
    kind's combined order. Display only."""
    keep = (
        (_num(df, "Owner_Yield") > 0)
        & ~(_num(df, "Interest_Coverage") < REIT_MIN_INTEREST_COVER)
        & ~(_num(df, "NetDebt_to_EBIT") > REIT_MAX_NET_DEBT_EBIT)
        & _dividend_held(df, through_downturns=True)
    )
    return _kind_list(df, "real_estate", n, keep)


# "Energy" (user's choice, Nov 5, 2026): energy stays in the rankings; this is a clean lens on it. Nearly all of it is
# cyclical, valued at mid-cycle, and Yahoo's few annual reports hold the 2022 spike, so the list asks for mid-cycle
# returns and a balance sheet that survives a price fall rather than the cheapest. Display only.
ENERGY_SECTOR = "Energy"
ENERGY_MIN_VALUE = 60.0
ENERGY_MIN_QUALITY = 50.0
ENERGY_MIN_ROIC = 10.0  # normalised ROIC, %
ENERGY_COLUMNS = [
    ("Net Cash / Cap", "Net_Cash_to_Cap", "+.0f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    RECESSION_COLUMN,
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def energy_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Ranked companies of Yahoo's Energy sector (producers, pipelines, refiners, services, coal, uranium) with Value
    at least `ENERGY_MIN_VALUE`, Quality at least `ENERGY_MIN_QUALITY`, normalised ROIC at least `ENERGY_MIN_ROIC`,
    net cash or the Shortlist's two debt tests (`_sound_balance_sheet`), and the dividend held through the downturns
    on file, the 2015 oil price collapse included, and this year (`_dividend_held`; a company that pays none passes);
    trusted as `ranking.top_n` (peak margins and distress out), not cut to the best fraction (too few would pass), in
    combined rank order. Display only."""
    sector = df["sector"] if "sector" in df.columns else pd.Series(None, index=df.index, dtype=object)
    keep = (
        (sector == ENERGY_SECTOR)
        & (_num(df, "Value_Score") >= ENERGY_MIN_VALUE)
        & (_num(df, "Quality_Score") >= ENERGY_MIN_QUALITY)
        & (_num(df, "ROIC_Norm") >= ENERGY_MIN_ROIC)
        & ((_num(df, "Net_Cash_to_Cap") > 0) | _sound_balance_sheet(df))
        & _dividend_held(df, through_downturns=True)
    )
    return top_n(df[keep], "Rank", n)


# The Below book list (`ranking.asset_play`), the balance-sheet list (user's choice, Nov 5, 2026): tangible book
# bargains among the ranked companies and holdings on their book with goodwill, ordered by `Asset_Play_Score`.
ASSET_PLAY_COLUMNS = [
    ("NAV / Price", "Effective_NAV_to_Price", ".0f", 1, "%"),
    ("Book / Price", "BV_to_Price", ".0f", 1, "%"),
    ("NCAV / Price", "NCAV_to_Price", ".0f", 1, "%"),
    ("Book Compounding / yr", "Book_Per_Share_Compounding", "+.1f", 1, "%"),
    ("Net Cash / Cap", "Net_Cash_to_Cap", "+.0f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def asset_play_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Flagged `Asset_Play` companies in the best fraction of the combined ranking, or holdings (no rank and too few
    for a best fifth of their own: the asset-play tests alone decide, user's choice, Nov 5, 2026), ordered by
    `Asset_Play_Score` (`Book_Order`, 1 = best)."""
    if "Asset_Play" not in df.columns:
        return df.head(0)
    d = df[(df["Asset_Play"] == True) & (in_best_fraction(df) | (kind(df) == "holding"))]  # noqa: E712
    score = _num(d, "Asset_Play_Score").fillna(_num(d, "Value_Score"))
    d = d.assign(Book_Order=score.rank(method="first", ascending=False))
    return d.sort_values("Book_Order").head(n)


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


# The "Dividend carry" list: what put a company on it, then the usual facts (user's choice, Oct 29, 2026).
CARRY_COLUMNS = [
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Unbroken Years", "Dividend_Unbroken_Years", ".0f", 1, ""),
    ("Dividend Growth / yr", "Regular_DPS_Growth_10Y", "+.1f", 1, "%"),
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


CARRY_MIN_UNBROKEN_YEARS = 10  # complete years back to the latest dividend cut or gap (`Dividend_Unbroken_Years`)


def dividend_carry_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Dividend payers a non-resident keeps whole (`Dividend_WHT` 0): yield at least `Settings.carry_min_yield` and above
    the cost of a euro margin loan (`Borrow_Rate`, set by `highlights.Highlights.compute` from `fx.get_borrow_rate`;
    `fx.FALLBACK_BORROW_RATE` without it), covered by the owner yield (what is left is retained and grows the value), the dividend
    record (`dividend_record`, user's choice, Nov 5, 2026: the list is bought partly on borrowed money, so a cut is the
    risk that matters): no cut in the last `CARRY_MIN_UNBROKEN_YEARS` complete years nor in the last 12 months, the
    regular dividend not lower than 10 years ago, and no record means no evidence (left out); owner earnings per share
    trending up over the annual reports (own currency), `in_best_fraction` of the combined ranking, not cyclical and
    not at peak margins, and the Shortlist's two debt tests (`_sound_balance_sheet`: the company's own debt must stay
    low under the buyer's loan); trusted as `ranking.top_n`, in combined rank order. Display only."""
    s = Settings()
    df = with_dividend_record(df)
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    shares = {p: num(f"Ordinary Shares Number_{p}").where(lambda x: x > 0) for p in s.annual_years}
    owner = pd.DataFrame({p: num(f"Owner_Earnings_{p}") / shares[p] for p in s.annual_years})
    yield_ = num("Dividend_Yield")
    keep = (
        (num("Dividend_WHT") == 0)
        & (yield_ >= np.maximum(s.carry_min_yield, num("Borrow_Rate").fillna(FALLBACK_BORROW_RATE)))
        & (num("Owner_Yield") > yield_)
        & (num("Dividend_Unbroken_Years") >= CARRY_MIN_UNBROKEN_YEARS)  # 0 when a cut was paid this year
        & ~_flag(df, "Dividend_Cut_Recent")
        & (num("Regular_DPS_Growth_10Y") >= 0)
        & _trending_up(owner, s.carry_min_years, s.carry_max_dips)
        & in_best_fraction(df)
        & (df["Cyclical"] != True if "Cyclical" in df.columns else True)  # noqa: E712 - handles NaN / numpy bool
        & ~(num("Peak_Margin_Ratio") > s.peak_margin_ratio)
        & _sound_balance_sheet(df)
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
# fingerprint (`evaluate.METHOD_MODULES`) and every snapshot's method group stay as they are. The top 20 by combined
# rank since Nov 5, 2026 (user's choice: about 20 picks before the grade, more cannot be followed; it was 100).
SHORTLIST_N = 20
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


# The "Operating companies" and "Fast growers" lists (user's design, Nov 4, 2026; they replaced Nectar, Compounders and
# Better every year, which overlapped them): businesses a holder could keep for 20 years, since a long holding returns
# roughly its owner yield now plus the per-share growth high returns on capital sustain. Display only; constants here,
# as the Shortlist's, so the method fingerprint stays as it is.
OPERATING_MIN_ROIC = 15.0  # normalised ROIC, %
OPERATING_MIN_YIELD = 6.0  # owner earnings yield, %
FAST_MIN_OE_GROWTH = 15.0  # `Owner_Earnings_Compounding`, %/yr: the USD per-share trend x its adjusted R²
FAST_MIN_YIELD = 4.0  # owner earnings yield, %: already profitable, not a promise
# `Per_Share_Compounding`, %/yr: sales per share compound as fast as owner earnings (user's choice, Nov 4, 2026: at 8%
# slow sellers passed on margin gains alone; at 15%, 8 of the benchmark against 19)
FAST_MIN_REVENUE_GROWTH = 15.0
MAX_NEW_SHARES = 1.005  # latest share count / the last annual report's: above this, shares were issued
MAX_OWNER_SPREAD = 0.2  # owner earnings per share: spread around their rising trend line / their mean
FLAT_TOLERANCE = 0.02  # a year within this share of the year before is flat, not a fall (user's choice, Nov 2, 2026)
# Operating companies: ROIC and operating margin may slip this share in a year if the newest is at or above the oldest
# (user's choice, Nov 4, 2026: at 2%, with the industries left out, 3 of the benchmark passed; at 10%, 10).
OPERATING_MAX_DIP = 0.10
# Growth that earns its keep (Buffett's reinvestment runway; Nov 4, 2026): the capital added since the oldest annual
# report earns at least this, in US terms (`Incremental_ROIC`; blank when capital barely grew passes). Both lists.
LIST_MIN_INCREMENTAL_ROIC = 15.0
# Pricing power (Nov 4, 2026): the gross margin now at most this many points below the year of the highest US inflation
# (`Gross_Margin_Held`; blank passes). Operating companies and the Shortlist's inflation stress test.
MAX_MARGIN_LOSS = 2.0
OPERATING_COLUMNS = [
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Incremental ROIC", "Incremental_ROIC", ".0f", 1, "%"),
    ("Gross Margin Since Inflation", "Gross_Margin_Held", "+.1f", 1, " pp"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip
FAST_COLUMNS = [
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Incremental ROIC", "Incremental_ROIC", ".0f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def _earns_its_keep(df: pd.DataFrame) -> pd.Series:
    """`Incremental_ROIC` not below `LIST_MIN_INCREMENTAL_ROIC` (blank passes)."""
    return ~(_num(df, "Incremental_ROIC") < LIST_MIN_INCREMENTAL_ROIC)


def _margin_held(df: pd.DataFrame) -> pd.Series:
    """`Gross_Margin_Held` not below -`MAX_MARGIN_LOSS` (blank passes)."""
    return ~(_num(df, "Gross_Margin_Held") < -MAX_MARGIN_LOSS)


def _annual(df: pd.DataFrame, concept: str, periods: list[str]) -> np.ndarray:
    """`concept` over the annual `periods` (newest first), NaN where missing."""
    return pd.DataFrame(
        {p: pd.to_numeric(df[f"{concept}_{p}"], errors="coerce") if f"{concept}_{p}" in df.columns else np.nan
         for p in periods}, index=df.index,
    ).to_numpy(dtype=float)  # fmt: skip


def _up_but_one_flat(df: pd.DataFrame, concept: str, periods: list[str], min_years: int) -> pd.Series:
    """As `_every_year`, but one year may be flat (`FLAT_TOLERANCE`) instead of up; a real fall fails."""
    values = _annual(df, concept, periods)
    run = np.cumprod(~np.isnan(values), axis=1).astype(bool)
    newer, older = values[:, :-1], values[:, 1:]
    up = np.where(run[:, 1:], newer > older, True)
    flat = np.where(run[:, 1:], newer >= older - np.abs(older) * FLAT_TOLERANCE, True)
    ok = flat.all(axis=1) & ((~up).sum(axis=1) <= 1) & (run.sum(axis=1) >= min_years)
    return pd.Series(ok, index=df.index)


def _never_fell(df: pd.DataFrame, concept: str, periods: list[str], min_years: int, dip: float) -> pd.Series:
    """Over the consecutive annual reports from Y0 back (at least `min_years`), no year below the one before by more
    than the share `dip`, and the newest at or above the oldest: held or rising, never sliding."""
    values = _annual(df, concept, periods)
    run = np.cumprod(~np.isnan(values), axis=1).astype(bool)
    n = run.sum(axis=1)
    newer, older = values[:, :-1], values[:, 1:]
    held = np.where(run[:, 1:], newer >= older - np.abs(older) * dip, True).all(axis=1)
    oldest = np.where(run, values, np.nan)[np.arange(len(values)), np.maximum(n - 1, 0)]
    return pd.Series(held & (n >= min_years) & (values[:, 0] >= oldest), index=df.index)


def _owner_earnings_smooth(per_share: pd.DataFrame, min_years: int) -> pd.Series:
    """Owner earnings per share (columns newest first) positive in every annual report, at least `min_years` of them,
    on a rising trend line and within `MAX_OWNER_SPREAD` of it."""

    def smooth(row: np.ndarray) -> bool:
        values = row[~np.isnan(row)]
        if len(values) < min_years or (values <= 0).any():
            return False
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        return bool(slope > 0 and np.std(values - (intercept + slope * x)) / values.mean() <= MAX_OWNER_SPREAD)

    rows = per_share.to_numpy(dtype=float)[:, ::-1]
    return pd.Series([smooth(row) for row in rows], index=per_share.index, dtype=bool)


def _owner_per_share(df: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    """Owner earnings per share in the company's own currency, one column `OE_PS_<year>` per annual report."""
    return pd.DataFrame({f"OE_PS_{p}": _num(df, f"Owner_Earnings_{p}") / _num(df, f"Ordinary Shares Number_{p}").where(
        lambda x: x > 0) for p in years}, index=df.index)  # fmt: skip


# AI screen for every screened highlight list (the three pure rankings Cheap and good, Cheapest and Best businesses
# are not screened; user's choice, Nov 4, 2026): industries whose business model AI can substitute or disrupt
# (`AI_FRAGILE`) stay out so the screens focus on boring, durable businesses. Display only; constants here, as the
# Shortlist's, so the method fingerprint stays as it is.
AI_FRAGILE = "ai_fragile.txt"
FAST_AVOID_KEYS = frozenset({industry_key("Biotechnology")})


def ai_resilient(df: pd.DataFrame) -> pd.Series:
    """Not in an industry fragile or exposed to AI disruption (`AI_FRAGILE`). `highlights.highlight_picks` applies it
    to every screened tab."""
    return ~_in_industries(df, AI_FRAGILE)


# A dividend that fell more than this % in a past global recession (`dividend_record`, rule 7; user's choice, Nov 4,
# 2026) shows a business the recession reached: the 20-year lists leave it out. A blank (no dividend, or a record
# too short) passes, as in every list; never scored (only payers have it).
RECESSION_MAX_FALL = 25.0


def _lasting_business(df: pd.DataFrame) -> pd.Series:
    """Not cyclical, not earning well above its usual margin (`Peak_Margin_Ratio`), past the AI screen
    (`ai_resilient`), its dividend not cut more than `RECESSION_MAX_FALL` % in a past recession: what the 20-year
    lists (Operating companies, Fast growers, Unbroken dividends) ask before they look at growth. Where a recession
    hits hard (`lists/shortlist_recession_industries.txt`, `cyclical_industries.txt`) and no dividend record reaches
    one (a young company, or one that pays none), industry evidence decides instead, as on the Shortlist: only an
    industry leader (`industry_leader`) holding net cash (user's choice, Nov 5, 2026)."""
    record = with_dividend_record(df)
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    hard_hit = _in_industries(df, "shortlist_recession_industries.txt") | industry.map(is_cyclical_industry).astype(
        bool
    )
    untested = _num(record, "Recession_Dividend_Fall").isna()
    leader = (
        df["Industry_Leader"].fillna(False).astype(bool) if "Industry_Leader" in df.columns else industry_leader(df)
    )
    return (
        ~_flag(df, "Cyclical")
        & ~(_num(df, "Peak_Margin_Ratio") > Settings().peak_margin_ratio)
        & ai_resilient(df)
        & ~(_num(record, "Recession_Dividend_Fall") < -RECESSION_MAX_FALL)
        & (~(hard_hit & untested) | (leader & (_num(df, "Net_Debt") <= 0)))
    )


def operating_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Operating companies that compound smoothly at a price already paying well: no bank, insurer or holding,
    `_lasting_business` and outside `operating_avoid.txt`; normalised ROIC >= `OPERATING_MIN_ROIC` and owner yield >=
    `OPERATING_MIN_YIELD`; over at least `Settings.improver_min_years` annual reports (own currency) owner earnings per
    share positive and close to their rising trend, revenue up every year but at most one flat, ROIC and
    operating margin never down more than `OPERATING_MAX_DIP` in a year and the newest at or above the oldest
    (`_never_fell`); the share count never up (nor since the last
    report); growth that earns its keep (`_earns_its_keep`) and costs passed on (`_margin_held`); `in_best_fraction`.
    Combined rank order, trusted as `ranking.top_n`. Display only."""
    s = Settings()
    years, k = s.annual_years, s.improver_min_years
    owner = _owner_per_share(df, years)
    keep = (
        ~_flag(df, "Financial_Mode")
        & _lasting_business(df)
        & ~_in_industries(df, "operating_avoid.txt")
        & (_num(df, "ROIC_Norm") >= OPERATING_MIN_ROIC)
        & (_num(df, "Owner_Yield") >= OPERATING_MIN_YIELD)
        & _owner_earnings_smooth(owner, k)
        & _up_but_one_flat(df, "Total Revenue", years, k)
        & _never_fell(df, "ROIC", years, k, OPERATING_MAX_DIP)
        & _never_fell(df, "operatingMargins", years, k, OPERATING_MAX_DIP)
        & _every_year(df, "Ordinary Shares Number", years, k, rising=False)
        & ~(_num(df, "Ordinary Shares Number_TTM") > _num(df, f"Ordinary Shares Number_{s.latest}") * MAX_NEW_SHARES)
        & _earns_its_keep(df)
        & _margin_held(df)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


def fast_grower_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Very fast, smooth growth that already pays: revenue per share compounding at >= `FAST_MIN_REVENUE_GROWTH` and
    owner earnings per share at >= `FAST_MIN_OE_GROWTH` (USD trends x their R²), owner yield >= `FAST_MIN_YIELD`;
    operating companies only, outside biotechnology (user's choice, Nov 4, 2026: banks and insurers filled half of it
    on the 2022-25 rate cycle, and clinical/patent binary risk is not a 20-year compounder); over at least
    `Settings.improver_min_years` annual reports (own currency) owner earnings per share positive and close to their
    rising trend, revenue and operating margin up every year but at most one flat, the share count stable or falling
    every year (nor up since the last report); `_lasting_business`; growth that earns its keep (`_earns_its_keep`: fast growth bought with
    capital earning little is no compounder) and costs passed on (`_margin_held`); `in_best_fraction`.
    Combined rank order, trusted as `ranking.top_n`. Display only."""
    s = Settings()
    years, k = s.annual_years, s.improver_min_years
    owner = _owner_per_share(df, years)
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    avoid = industry.map(lambda i: isinstance(i, str) and industry_key(i) in FAST_AVOID_KEYS).astype(bool)
    keep = (
        ~_flag(df, "Financial_Mode")
        & ~avoid
        & (_num(df, "Owner_Earnings_Compounding") >= FAST_MIN_OE_GROWTH)
        & (_num(df, "Per_Share_Compounding") >= FAST_MIN_REVENUE_GROWTH)
        & (_num(df, "Owner_Yield") >= FAST_MIN_YIELD)
        & _owner_earnings_smooth(owner, k)
        & _up_but_one_flat(df, "Total Revenue", years, k)
        & _up_but_one_flat(df, "operatingMargins", years, k)
        & _every_year(df, "Ordinary Shares Number", years, k, rising=False)
        & ~(_num(df, "Ordinary Shares Number_TTM") > _num(df, f"Ordinary Shares Number_{s.latest}") * MAX_NEW_SHARES)
        & _lasting_business(df)
        & _earns_its_keep(df)
        & _margin_held(df)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


# The "Unbroken dividends" list (user's design, Nov 4, 2026): boring businesses whose dividend per share was raised or
# held every year of the whole record Yahoo keeps (`report.dividend_record`, payment by payment), paying well and
# covered, with stable returns, little debt and pricing power. Display only; constants here, as the Shortlist's.
UNBROKEN_MIN_YEARS = 12  # the whole record, at least this long
UNBROKEN_MIN_YIELD = 3.0  # dividend yield, %
UNBROKEN_MIN_OWNER_YIELD = 6.0  # owner earnings yield, %
UNBROKEN_MAX_PAYOUT = 0.8  # the dividend at most this share of owner earnings: the rest grows the next dividend
UNBROKEN_MIN_GROWTH = 3.0  # regular dividend per share, %/yr over 10 years: held for a decade is not enough
UNBROKEN_MIN_ROIC = 10.0  # normalised ROIC (ROE for financials), %
UNBROKEN_MAX_SHARE_GROWTH = 0.5  # `Shares_Change_10Y`, %/yr: dividends not paid for with new shares
UNBROKEN_COLUMNS = [
    ("Unbroken Years", "Dividend_Unbroken_Years", ".0f", 1, ""),
    ("Dividend Yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Dividend Growth / yr", "Regular_DPS_Growth_10Y", "+.1f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Shares Reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def with_dividend_record(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with the `RECORD_COLUMNS` of `dividend_record.dividend_record` it lacks (`highlights.prepare` adds them
    once per report); a column already there is kept."""
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if not missing:
        return df
    return df.assign(**dividend_record(df)[missing])


def unbroken_dividend_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Dividends per share raised or held every year of the whole record Yahoo keeps, at least `UNBROKEN_MIN_YEARS`
    (`dividend_record`: specials and payment timing judged payment by payment, in the declared currency, a cut this
    year counting at once), reaching back through every recession in `lists/recessions.txt`; dividend yield
    >= `UNBROKEN_MIN_YIELD`, owner yield >= `UNBROKEN_MIN_OWNER_YIELD`, the dividend at most `UNBROKEN_MAX_PAYOUT` of
    owner earnings, the regular dividend growing >= `UNBROKEN_MIN_GROWTH` %/yr over 10 years; normalised ROIC (ROE for
    financials) >= `UNBROKEN_MIN_ROIC` and never down more than `OPERATING_MAX_DIP` in a year nor below where it
    started (`_never_fell`); shares not issued (`Shares_Change_10Y` <= `UNBROKEN_MAX_SHARE_GROWTH`, nor since the last
    report); a boring business (`_lasting_business`, outside `operating_avoid.txt`) with pricing power
    (`_margin_held`); a sound balance sheet (`_sound_balance_sheet`); `in_best_fraction`, or for a bank or insurer
    (ranked apart, so it has no `Rank`) the best fraction of its own kind (`kind_order`). Ordered by position as a
    share of the ranking (`Rank` among the ranked, `Kind_Rank` within the kind), trusted as `ranking.top_n`. Display
    only."""
    s = Settings()
    df = with_dividend_record(df)
    kind_rank, kind_best = kind_order(df)
    kinds = kind(df)
    financial = kinds.isin(("bank", "insurer"))
    rank = _num(df, "Rank")
    order = (rank / max(int(rank.notna().sum()), 1)).where(~financial)
    for k in ("bank", "insurer"):
        members = kinds == k
        order = order.where(~members, kind_rank / max(int(kind_rank[members].notna().sum()), 1))
    yield_ = _num(df, "Dividend_Yield")
    keep = (
        _flag(df, "Dividend_Record_Whole")
        & (_num(df, "Recessions_Covered") >= _num(df, "Recessions_Due"))  # held through every recession on file
        & (_num(df, "Dividend_Unbroken_Years") >= UNBROKEN_MIN_YEARS)
        & (yield_ >= UNBROKEN_MIN_YIELD)
        & (_num(df, "Owner_Yield") >= UNBROKEN_MIN_OWNER_YIELD)
        & ~(yield_ > UNBROKEN_MAX_PAYOUT * _num(df, "Owner_Yield"))
        & (_num(df, "Regular_DPS_Growth_10Y") >= UNBROKEN_MIN_GROWTH)
        & (_num(df, "ROIC_Norm") >= UNBROKEN_MIN_ROIC)
        & _never_fell(df, "ROIC", s.annual_years, s.improver_min_years, OPERATING_MAX_DIP)
        & ~(_num(df, "Shares_Change_10Y") > UNBROKEN_MAX_SHARE_GROWTH)
        & ~(_num(df, "Ordinary Shares Number_TTM") > _num(df, f"Ordinary Shares Number_{s.latest}") * MAX_NEW_SHARES)
        & _lasting_business(df)
        & ~_in_industries(df, "operating_avoid.txt")
        & _margin_held(df)
        & _sound_balance_sheet(df)
        & (in_best_fraction(df) | (financial & kind_best))
    )
    return top_n(df[keep].assign(List_Order=order[keep]), "List_Order", n)


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


def _debt_bounded(df: pd.DataFrame) -> pd.Series:
    """Operating companies: net debt at most `SHORTLIST_MAX_NET_DEBT_EBIT` x EBIT or interest covered
    `SHORTLIST_MIN_INTEREST_COVER` times; banks, insurers and holdings: equity at least `SHORTLIST_MIN_EQUITY_ASSETS` %
    of assets. A blank figure passes."""
    financial = _flag(df, "Financial_Mode")
    operating = ~(_num(df, "NetDebt_to_EBIT") > SHORTLIST_MAX_NET_DEBT_EBIT) | (
        _num(df, "Interest_Coverage") >= SHORTLIST_MIN_INTEREST_COVER
    )
    return (operating & ~financial) | (~(_num(df, "Equity_to_Assets") < SHORTLIST_MIN_EQUITY_ASSETS) & financial)


def _sound_balance_sheet(df: pd.DataFrame) -> pd.Series:
    """`_debt_bounded`, and for operating companies net debt within `SHORTLIST_MAX_NET_DEBT_EQUITY` % of equity: the
    Shortlist's two debt tests at once."""
    return _debt_bounded(df) & (_net_debt_within_equity(df) | _flag(df, "Financial_Mode"))


def _shortlist_gates(df: pd.DataFrame, hurdle: float = BOND_HURDLE_FLOOR) -> list[tuple[str, pd.Series]]:
    """The Shortlist's conditions in funnel order, each (label, mask). A blank figure passes, as in every list: the
    operating tests are blank for banks, insurers and holdings, which get the equity / assets floor instead. `hurdle`
    (`bond_hurdle`) is the owner yield to beat."""
    s = Settings()
    num = lambda col: (  # noqa: E731
        pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    )
    financial = _flag(df, "Financial_Mode")
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
        ("sound balance sheet", _debt_bounded(df)),
        ("nothing breaking now",
         ~(num("EPS_Rev_90d") < s.estimate_cut_pct)
         & _dividend_held(df)  # no dividend cut paid in the last 12 months (Nov 5, 2026)
         & ~(num("Revenue_Change_Recent") < s.dislocation_revenue_change_min)
         & ~(num("EBIT_Change_Recent") < s.dislocation_ebit_change_min)),
        ("AI: no fragile or exposed industry", ai_resilient(df)),
        ("recession: leaders only", ~(recession_prone | bank) | (leader & (bank | (net_debt <= 0)))),
        ("net debt within half of equity", net_debt_ok | financial),
        ("kept its margin through inflation", _margin_held(df)),
        ("beats the bond", ~(num("Owner_Yield") < hurdle)),
        ("durable returns",
         (num("ROIC_Norm") >= SHORTLIST_MIN_ROIC) & (num("Value_Score") >= SHORTLIST_MIN_VALUE)
         & (num("Quality_Score") >= SHORTLIST_MIN_QUALITY)),
    ]  # fmt: skip


def shortlist_list(df: pd.DataFrame, n: int = SHORTLIST_N, hurdle: float = BOND_HURDLE_FLOOR) -> pd.DataFrame:
    """The companies worth researching in depth: `in_best_fraction` of the combined ranking and every condition of
    `_shortlist_gates` (per-share growth without dilution, owner earnings not shrinking, cash-backed and clean
    earnings, growth that earns its keep, a sound balance sheet, nothing breaking now, not fragile to AI, recession
    leaders only, net debt within half of equity, the gross margin kept through inflation, an owner yield above the
    bond `hurdle`, durable returns); trusted
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


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool) if col in df.columns else pd.Series(False, index=df.index)
