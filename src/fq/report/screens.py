"""The front page's list screens and their columns (no markup here). Not a company-page input
(`company_page.PAGE_INPUTS`): editing a list re-renders no company page; what a page shows of the lists travels
in its row (`On_Lists`, `Industry_Leader`, `In_Best_Fraction`)."""

from functools import cache

import numpy as np
import pandas as pd

from fq import grading
from fq.fx import FALLBACK_BORROW_RATE
from fq.metrics.common import us_terms
from fq.metrics.long_history import parse_history
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

# The lists named cheap (Cheap and good, Cheapest; user's choice, Nov 8, 2026): what one share earns after today's
# interest at zero growth, over the share price (`EPV_Per_Share_Yield`), above this plus the home country's sovereign
# spread (`Sovereign_Spread`; Nov 11, 2026, user's choice: one price bar, as Boring businesses and The List ask, so
# Cheap and good no longer gives seats The List then refuses on country risk); a blank yield fails. The ranking
# stays on the yield over market cap plus debt, less spare cash.
CHEAP_MIN_SHARE_YIELD = 6.0
SHARE_YIELD_COLUMN = ("Owner Yield, Share Price", "EPV_Per_Share_Yield", ".1f", 1, "%")
CHEAP_AND_GOOD_COLUMNS = [*TOP_LIST_COLUMNS, SHARE_YIELD_COLUMN]
# The List (Nov 9, 2026): its order first, then the business yield and the usual columns.
THE_LIST_COLUMNS = [SHARE_YIELD_COLUMN, ("Owner Yield", "Owner_Yield", ".1f", 1, "%"), *TOP_LIST_COLUMNS]


def cheap_bar(df: pd.DataFrame) -> pd.Series:
    """The owner yield on the share price above `CHEAP_MIN_SHARE_YIELD` plus the home country's sovereign spread
    (`Sovereign_Spread`, a blank counts as 0, the US); a blank yield fails."""
    return _num(df, "EPV_Per_Share_Yield") > CHEAP_MIN_SHARE_YIELD + _num(df, "Sovereign_Spread").fillna(0.0)


def cheap_enough(df: pd.DataFrame) -> pd.DataFrame:
    """The companies past `cheap_bar` (the lists named cheap)."""
    return df[cheap_bar(df)]


# A list needs a record (user's choice, Nov 10, 2026, from a review "as Graham, Buffett, Munger and Greenblatt": a
# company with one or two annual reports scored a median Quality of 62-65 against 50 for a full record, and a quarter
# of the top 100 had listed since about 2021; the seller of a new issue picks the date and the figures). Display only.
RECORD_MIN_YEARS = 4  # annual ROIC values on file, of the `Settings.n_years` Yahoo serves
LISTED_MIN_YEARS = 5  # years spanned by the yearly share counts or dividends (`listed_years`): about how long listed
# Tickers that pass anyway: an old business under a new ticker (a new holding company, a spin-off with its history).
RECORD_CONFIRMED = "record_confirmed.txt"


@cache
def _record_confirmed() -> frozenset[str]:
    return frozenset(line.split()[0].upper() for line in _read_lines(RECORD_CONFIRMED))


def roic_years(df: pd.DataFrame) -> pd.Series:
    """How many annual `ROIC_<Y>` values a company has on file."""
    return sum((_num(df, f"ROIC_{y}").notna().astype(int) for y in Settings().annual_years),
               pd.Series(0, index=df.index))  # fmt: skip


def _history_span(df: pd.DataFrame, col: str) -> pd.Series:
    hist = df[col] if col in df.columns else pd.Series(None, index=df.index, dtype=object)
    spans = [(s.index.max() - s.index.min()) if len(s := parse_history(v)) else np.nan for v in hist]
    return pd.Series(spans, index=df.index, dtype=float)


def listed_years(df: pd.DataFrame) -> pd.Series:
    """About how long a company has been listed: the longer of the years its yearly share counts (`Shares_Hist`) and
    its dividends per share (`DPS_Hist`) span, NaN without either. Both: Yahoo's share counts start late in some
    markets (every Vietnamese company about 2023, Vinamilk listed 2006), while a dividend is paid only once listed."""
    return pd.concat([_history_span(df, "Shares_Hist"), _history_span(df, "DPS_Hist")], axis=1).max(axis=1)


def has_record(df: pd.DataFrame) -> pd.Series:
    """At least `RECORD_MIN_YEARS` annual returns on file and a share history spanning at least `LISTED_MIN_YEARS`
    (a blank history passes), or a ticker in `lists/record_confirmed.txt`: what every list asks
    (`highlights.listable`). A frame with no annual ROIC column at all has no record to test, as `in_best_fraction`
    keeps every row of a frame without `Rank`. `highlights.prepare` stores it as `Has_Record` (with `Record_Years` and
    `Listed_Years`) once per report, and a stored column is read as it is."""
    if "Has_Record" in df.columns:
        return df["Has_Record"].fillna(True).astype(bool)
    tickers = df["Ticker"] if "Ticker" in df.columns else pd.Series("", index=df.index)
    confirmed = tickers.astype(str).str.upper().isin(_record_confirmed())
    has_roic = any(f"ROIC_{y}" in df.columns for y in Settings().annual_years)
    enough = (roic_years(df) >= RECORD_MIN_YEARS) if has_roic else pd.Series(True, index=df.index)
    return confirmed | (enough & ~(listed_years(df) < LISTED_MIN_YEARS))


# The top Value and Quality lists add the numbers behind the score (a percentile alone says little at the top).
VALUE_LIST_COLUMNS = [*TOP_LIST_COLUMNS, ("Owner Yield", "Owner_Yield", ".1f", 1, "%"), SHARE_YIELD_COLUMN]
QUALITY_LIST_COLUMNS = [
    ("Moat", "Grade_Moat", ".2f", 1, ""),
    ("Management", "Grade_Mgmt", ".2f", 1, ""),
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
KINDS = ("bank", "insurer", "holding", "real_estate", "utility")


def kind(df: pd.DataFrame) -> pd.Series:
    """The kind a company ranked apart is judged among (`KINDS`), else None (ranked, loss-making, too little data)."""
    group = df["Rank_Group"] if "Rank_Group" in df.columns else pd.Series(None, index=df.index, dtype=object)
    model = df["Business_Model"] if "Business_Model" in df.columns else pd.Series(None, index=df.index, dtype=object)
    out = group.where(group.isin({"holding", "real_estate", "utility"}), model.where(group == "financial"))
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
# x: property is financed long; a REIT above this is a bet on its lenders. 10 (user's choice, Nov 6, 2026; was 8): a
# REIT's depreciation is large, so its EBIT understates what services the debt.
REIT_MAX_NET_DEBT_EBIT = 10.0
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


# "Utilities" (user's choice, Nov 6, 2026): regulated or contracted returns on long-lived assets bought with debt,
# judged against other utilities only, with the debt bounded as the REITs' and the dividend that is their point held.
UTILITY_MIN_INTEREST_COVER = REIT_MIN_INTEREST_COVER
UTILITY_MAX_NET_DEBT_EBIT = 8.0
UTILITY_COLUMNS = REAL_ESTATE_COLUMNS


def utility_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Utilities (`lists/utilities.txt`) in the best fraction of their kind, owner yield positive, interest covered
    `UTILITY_MIN_INTEREST_COVER` times and net debt at most `UTILITY_MAX_NET_DEBT_EBIT` x EBIT (a blank passes), the
    dividend held through the downturns and this year (`_dividend_held`), trusted, in their kind's combined order.
    Display only."""
    keep = (
        (_num(df, "Owner_Yield") > 0)
        & ~(_num(df, "Interest_Coverage") < UTILITY_MIN_INTEREST_COVER)
        & ~(_num(df, "NetDebt_to_EBIT") > UTILITY_MAX_NET_DEBT_EBIT)
        & _dividend_held(df, through_downturns=True)
    )
    return _kind_list(df, "utility", n, keep)


# The two NAV lists (user's design, Nov 14, 2026; they were Balance sheet plays, on tangible book, and Deep discount,
# under 0.66x a checked book: the same idea at two depths, both on book). What a buyer of assets looks for is a
# discount to net asset value: when the net assets compound, to what they are worth as a going concern; when they do
# not, to a fire sale, and then a deep one. The net assets are the book check's estimate (`grading`, the model never
# sees a price), and the price against them orders both lists, the smaller the better. Display only.
NAV_COMPOUNDING_MIN = 8.0  # % a year: book per share trend plus dividends on the book (`NAV_Compounding`)
NAV_MAX_PRICE = 1.0  # Compounding below NAV: price under this multiple of the going-concern estimate
FIRE_SALE_MAX_PRICE = 0.66  # Below fire-sale value: price at most this multiple of the fire-sale estimate
# Who gets an estimate (`nav_candidates`): under this multiple of book and tangible book, any business; under the
# second, the kinds whose assets are worth more than their book (`NAV_INDUSTRIES`, REITs); a holding at any price.
NAV_CHECK_MAX_PRICE_TO_BOOK = 1.0
NAV_CHECK_ASSET_MAX_PRICE_TO_BOOK = 1.5
NAV_INDUSTRIES = "nav_industries.txt"
NAV_COLUMNS = [
    ("NAV Compounding / yr", "NAV_Compounding", "+.1f", 1, "%"),
    ("Price / Book", "Price_to_Book", ".2f", 1, "x"),
    ("Net Cash / Cap", "Net_Cash_to_Cap", "+.0f", 1, "%"),
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
]  # fmt: skip
ASSET_PLAY_COLUMNS = [
    ("Price / NAV", "Price_to_NAV", ".2f", 1, "x"),
    ("Price / Fire-Sale NAV", "Price_to_Fire_NAV", ".2f", 1, "x"),
    *NAV_COLUMNS,
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def nav_multiple(book_to_price, book, nav):
    """Price / NAV, for numbers or Series: today's price over book (`BV_to_Price`, book / price in %) times the book
    the estimate was compared with (`Check_Book_Book`) over the estimate, both in the same money. The one formula of
    the lists and the company card (`cards.nav_figures`)."""
    return (100.0 / book_to_price) * book / nav


def price_to_nav(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(`Price_to_NAV`, `Price_to_Fire_NAV`): the share price over the book check's going-concern estimate and over
    its fire-sale estimate (`nav_multiple`); blank without an estimate above zero or a positive book.
    `highlights.prepare` stores both, and stored columns are read as they are."""
    if "Price_to_NAV" in df.columns and "Price_to_Fire_NAV" in df.columns:
        return _num(df, "Price_to_NAV"), _num(df, "Price_to_Fire_NAV")
    to_price, book = _num(df, "BV_to_Price"), _num(df, "Check_Book_Book")
    to_price, book = to_price.where(to_price > 0), book.where(book > 0)
    base, low = _num(df, "Check_Book_NAV_Base"), _num(df, "Check_Book_NAV_Low")
    return nav_multiple(to_price, book, base.where(base > 0)), nav_multiple(to_price, book, low.where(low > 0))


def nav_holds(df: pd.DataFrame) -> pd.Series:
    """`grading.nav_verdict` per company (True, False, or None with no estimate): an estimate above zero that
    reaches a minority shareholder and is not being eaten. Stored as `NAV_Holds` by `highlights.prepare`."""
    if "NAV_Holds" in df.columns:
        return df["NAV_Holds"]
    blank = pd.Series(None, index=df.index, dtype=object)
    topics, base = df.get("Check_Book_Topics", blank), df.get("Check_Book_NAV_Base", blank)
    return pd.Series(
        [grading.nav_verdict(t, b) for t, b in zip(topics, base, strict=True)], index=df.index, dtype=object
    )


def not_loss_making(df: pd.DataFrame) -> pd.Series:
    """No loss in a normal year (`Rank_Group` not "unprofitable") nor in the latest annual report or the last twelve
    months (clean net income; a blank passes, as in every list)."""
    group = df["Rank_Group"] if "Rank_Group" in df.columns else pd.Series(None, index=df.index, dtype=object)
    annual = _num(df, f"Net_Income_Clean_{Settings().latest}")
    return (group != "unprofitable") & ~(annual <= 0) & ~(_num(df, "Net_Income_Clean_TTM") <= 0)


def _lender(df: pd.DataFrame) -> pd.Series:
    """A bank or an insurer: its book is loans and reserves, which the book check does not read."""
    model = df["Business_Model"] if "Business_Model" in df.columns else pd.Series(None, index=df.index, dtype=object)
    return model.isin(("bank", "insurer"))


def nav_candidates(df: pd.DataFrame) -> pd.Series:
    """The companies whose net assets the book check estimates, on a list or not (`Highlights.check_pool`; user's
    choice, Nov 14, 2026: the wide pool). Every deep discount, as since Nov 11, 2026 (`deep_discount`, loss-making or
    not); and, not loss-making (`not_loss_making`) and no bank or insurer: a price under
    `NAV_CHECK_MAX_PRICE_TO_BOOK` of both book and tangible book; a holding at any price (its stakes may be worth far
    more than their book: BN.TO at 2.2x); a REIT or an industry of `NAV_INDUSTRIES` (property owners and developers,
    conglomerates: land and stakes at old cost, ISUR.MC at 1.1x) under `NAV_CHECK_ASSET_MAX_PRICE_TO_BOOK`. Stored as
    `NAV_Candidate` by `highlights.prepare`."""
    if "NAV_Candidate" in df.columns:
        return df["NAV_Candidate"].fillna(False).astype(bool)
    book, tangible = _num(df, "BV_to_Price"), _num(df, "NAV_to_Price")
    below = (book > 100 / NAV_CHECK_MAX_PRICE_TO_BOOK) & (tangible > 100 / NAV_CHECK_MAX_PRICE_TO_BOOK)
    kinds = kind(df)
    asset = ((kinds == "real_estate") | _in_industries(df, NAV_INDUSTRIES)) & (
        book > 100 / NAV_CHECK_ASSET_MAX_PRICE_TO_BOOK
    )
    return deep_discount(df) | (not_loss_making(df) & ~_lender(df) & (below | (kinds == "holding") | asset))


def nav_compounds(df: pd.DataFrame) -> pd.Series:
    """The net assets compound: `NAV_Compounding` at least `NAV_COMPOUNDING_MIN` % a year (a blank does not)."""
    return _num(df, "NAV_Compounding") >= NAV_COMPOUNDING_MIN


def _nav_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The rows either NAV list may hold, with `Price_to_NAV`, `Price_to_Fire_NAV`, `Price_to_Book` and `NAV_Holds`
    (True on every row, the check these lists count): an estimate that holds (`nav_holds`), not loss-making, no bank
    or insurer, no asset manager unless a holding (`Settings.asset_play_excluded_industries`: its value is fees no
    book records; a holding of stakes is its book), trusted (`ranking.trusted`)."""
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    manager = industry.isin(Settings().asset_play_excluded_industries) & (kind(df) != "holding")
    holds = nav_holds(df).map(lambda v: v is True or v == True).astype(bool)  # noqa: E712 - numpy bool
    base, fire = price_to_nav(df)
    d = df.assign(Price_to_NAV=base, Price_to_Fire_NAV=fire, Price_to_Book=100 / _num(df, "BV_to_Price"))
    return d[holds & not_loss_making(df) & ~_lender(df) & ~manager & trusted(df)].assign(NAV_Holds=True)


def asset_play_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The "Compounding below NAV" list (stored as Below book; user's design, Nov 14, 2026): net assets that compound
    (`nav_compounds`) priced under `NAV_MAX_PRICE` of what they are worth as a going concern (`Price_to_NAV`), among
    `_nav_rows`, not cyclical (user's choice, Nov 10, 2026: ships, oil fields and car-parts plants are no liquid
    margin of safety), the smallest price / NAV first (`Book_Order`, 1 = best). Operating companies, holdings and
    REITs alike; no return floor, no best-fraction cut and no share-price yield bar (as it had on book until Nov 14,
    2026): The List asks those of its members."""
    d = _nav_rows(df)
    d = d[nav_compounds(d) & (d["Price_to_NAV"] < NAV_MAX_PRICE) & ~_flag(d, "Cyclical")]
    d = d.assign(Book_Order=d["Price_to_NAV"].rank(method="first"))
    return d.sort_values("Book_Order").head(n)


# The "Deep discount" tab, shown as "Below fire-sale value" since Nov 14, 2026. Its first rule (user's choice, Nov 11,
# 2026, prompted by Swire Pacific B at 0.40x book and 0.44x tangible book with no book check, being on no list) still
# decides who is always checked: price under this multiple of both book and tangible book (`nav_candidates` adds to
# it). Banks and insurers are left out: the book check reads assets carried high and hidden liabilities, not loans
# and reserves.
DEEP_DISCOUNT_MAX_PRICE_TO_BOOK = 0.66
DEEP_DISCOUNT_COLUMNS = [
    ("Price / Fire-Sale NAV", "Price_to_Fire_NAV", ".2f", 1, "x"),
    ("Price / NAV", "Price_to_NAV", ".2f", 1, "x"),
    *NAV_COLUMNS,
    ("Management", "Grade_Mgmt", ".2f", 1, ""),
    SHARE_YIELD_COLUMN,
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def deep_discount(df: pd.DataFrame) -> pd.Series:
    """Priced under `DEEP_DISCOUNT_MAX_PRICE_TO_BOOK` of both book and tangible book per share (`BV_to_Price` and
    `NAV_to_Price`, book / price in %, both present), not a bank or an insurer. `highlights.prepare` stores it as
    `Deep_Discount`, and a stored column is read as it is."""
    if "Deep_Discount" in df.columns:
        return df["Deep_Discount"].fillna(False).astype(bool)
    floor = 100 / DEEP_DISCOUNT_MAX_PRICE_TO_BOOK
    return (_num(df, "BV_to_Price") >= floor) & (_num(df, "NAV_to_Price") >= floor) & ~_lender(df)


def deep_discount_list(df: pd.DataFrame) -> pd.DataFrame:
    """The candidates of "Below fire-sale value" (stored as Deep discount; user's design, Nov 14, 2026), in its
    order: net assets that do not compound (`nav_compounds` false: those that do are measured as a going concern, on
    Compounding below NAV) priced at most `FIRE_SALE_MAX_PRICE` of what they would fetch if the company closed
    (`Price_to_Fire_NAV`), among `_nav_rows`, the smallest price / fire-sale NAV first (`Discount_Order`, 1 =
    first). `highlights.deep_discount_walk` walks it and asks the management grade."""
    d = _nav_rows(df)
    d = d[~nav_compounds(d) & (d["Price_to_Fire_NAV"] <= FIRE_SALE_MAX_PRICE)]
    d = d.assign(Discount_Order=d["Price_to_Fire_NAV"].rank(method="first"))
    return d.sort_values("Discount_Order")


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


# The "Shortlist 100" list (user's choice, Oct 29, 2026): the hand-off for research in depth, every diagnostic the
# cache already holds applied at once. Display only; the thresholds live here, not in `Settings`, so the method
# fingerprint (`evaluate.METHOD_MODULES`) and every snapshot's method group stay as they are. The top 20 by combined
# rank since Nov 5, 2026 (user's choice: about 20 picks before the grade, more cannot be followed; it was 100).
SHORTLIST_N = 20
# Boring businesses (the Shortlist's reader name since Nov 6, 2026): only industries that do not attract new capital.
BORING_INDUSTRIES = "boring_industries.txt"
NOTHING_BREAKING = "nothing breaking now"  # the funnel label of `nothing_breaking`
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
# Compounders' steady route (user's choice, Nov 6, 2026, with the merge: were 15% and 6%, 11 of the benchmark passed)
OPERATING_MIN_ROIC = 12.0  # normalised ROIC, %
OPERATING_MIN_YIELD = 5.0  # owner earnings yield, %
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
COMPOUNDER_COLUMNS = [
    ("ROIC Norm", "ROIC_Norm", ".1f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Per-Share Compounding", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Owner Earnings Compounding", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Incremental ROIC", "Incremental_ROIC", ".0f", 1, "%"),
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


# Money figures the year-by-year tests read in US terms (`common.us_terms`; Nov 6, 2026: stored figures are the
# statements' currency at today's rate, so a high-inflation currency's revenue rose every year whatever the volume).
# Ratios, returns (ROIC is in US terms already) and share counts are read as stored.
US_TERMS_CONCEPTS = frozenset({"Total Revenue"})


def _annual(df: pd.DataFrame, concept: str, periods: list[str]) -> np.ndarray:
    """`concept` over the annual `periods` (newest first), NaN where missing; `US_TERMS_CONCEPTS` in US terms."""
    values = pd.DataFrame(
        {p: pd.to_numeric(df[f"{concept}_{p}"], errors="coerce") if f"{concept}_{p}" in df.columns else np.nan
         for p in periods}, index=df.index,
    )  # fmt: skip
    if concept in US_TERMS_CONCEPTS:
        values = us_terms(values, df, Settings())
    return values.to_numpy(dtype=float)


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
    """Owner earnings per share in US terms (`common.us_terms`: the company's own currency at today's rate, net of
    its inflation beyond US; Nov 6, 2026), one column `OE_PS_<year>` per annual report."""
    owner = us_terms(pd.DataFrame({p: _num(df, f"Owner_Earnings_{p}") for p in years}, index=df.index), df, Settings())
    shares = pd.DataFrame({p: _num(df, f"Ordinary Shares Number_{p}").where(lambda x: x > 0) for p in years})
    return (owner / shares).rename(columns=lambda p: f"OE_PS_{p}")


# AI screen for every highlight list (user's choice, Nov 4, 2026; the three score rankings Cheap and good, Cheapest
# and Best businesses too since Nov 9, 2026): industries whose business model AI can substitute or disrupt
# (`AI_FRAGILE`) stay out so the lists focus on boring, durable businesses. Display only; constants here, as the
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
    lists (Compounders, Income) ask before they look at growth. Where a recession
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
        & _debt_bounded(df)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


def compounder_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The "Compounders" list (user's choice, Nov 6, 2026: Operating companies and Fast growers merged, 11 and 4 of the
    benchmark): the companies of either route, `operating_list` (steady, paying well) or `fast_grower_list` (very
    fast, already paying), once each, in combined rank order. Display only."""
    both = pd.concat([operating_list(df, len(df)), fast_grower_list(df, len(df))])
    both = both[~both.index.duplicated()]
    return both.sort_values("Rank", kind="stable").head(n)


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
        & _debt_bounded(df)
        & in_best_fraction(df)
    )
    return top_n(df[keep], "Rank", n)


# The "Income" list (user's choice, Nov 7, 2026: Dividend growers and Untaxed dividends merged): dividends a foreign
# investor keeps, held for a decade, ordered by what they pay now plus how fast they can grow. Stability is a gate,
# not a score: a dividend either has the record or it does not. Display only; constants here, as the Shortlist's.
INCOME_MAX_WHT = 10.0  # statutory dividend withholding on a non-resident, % (before treaties: one yardstick)
INCOME_MIN_YEARS = 10  # complete years back to the latest gap or dip deeper than `INCOME_HELD` allows
# A year may pay down to this share of the year before (user's choice, Nov 8, 2026; the other lists keep
# `dividend_record.HELD`, 95%): most companies at 10% withholding or less (China, Thailand, the Gulf, Malaysia) pay a
# fixed share of earnings, so the dividend moves with them; a dip in a soft year is the policy working, not distress.
# The dividend no lower than ten years ago, the recession test and the cover catch a real decline.
INCOME_HELD = 0.80
INCOME_MIN_YIELD = 2.5  # dividend yield after withholding, %
INCOME_MIN_ROIC = 10.0  # normalised ROIC (ROE for financials), %
# The dividend at most this share of what a share earns after interest at zero growth (`EPV_Per_Share_Yield`; Nov 7,
# 2026): the rest grows the next dividend.
INCOME_MAX_PAYOUT = 0.8
# Growth counted in the order, %/yr at most: past it a company is a growth stock with a dividend, not income.
INCOME_GROWTH_CAP = 10.0
INCOME_COLUMNS = [
    ("Income Return", "Income_Return", ".1f", 1, "%"),
    ("Yield after WHT", "Net_Dividend_Yield", ".1f", 1, "%"),
    ("Growth Counted / yr", "Income_Growth", "+.1f", 1, "%"),
    ("Dividend Growth / yr", "Regular_DPS_Growth_10Y_US", "+.1f", 1, "%"),
    ("Years Paid, No Deep Dip", "Income_Unbroken_Years", ".0f", 1, ""),
    ("Spread over Loan", "Loan_Spread", "+.1f", 1, "%"),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    RECESSION_COLUMN,
    *TOP_LIST_COLUMNS,
]  # fmt: skip


def with_dividend_record(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with the `RECORD_COLUMNS` of `dividend_record.dividend_record` it lacks (`highlights.prepare` adds them
    once per report) and `Regular_DPS_Growth_10Y_US` (`dividend_growth_us`); a column already there is kept."""
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        df = df.assign(**dividend_record(df)[missing])
    if "Regular_DPS_Growth_10Y_US" not in df.columns:
        df = df.assign(Regular_DPS_Growth_10Y_US=dividend_growth_us(df))
    return df


INCOME_RECORD = {"Income_Unbroken_Years": "Dividend_Unbroken_Years", "Income_Cut_Recent": "Dividend_Cut_Recent"}


def with_income_record(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with `Income_Unbroken_Years` and `Income_Cut_Recent`: the dividend record judged with Income's tolerance
    (`dividend_record(df, held=INCOME_HELD)`), only for rows at most `INCOME_MAX_WHT` withholding (the others never
    reach the list; blank and False); columns already there are kept (`highlights.prepare` adds them once)."""
    if all(c in df.columns for c in INCOME_RECORD):
        return df
    out = pd.DataFrame({"Income_Unbroken_Years": np.nan, "Income_Cut_Recent": False}, index=df.index)
    asked = _num(df, "Dividend_WHT") <= INCOME_MAX_WHT
    if asked.any():
        record = dividend_record(df[asked], held=INCOME_HELD)
        for col, source in INCOME_RECORD.items():
            out.loc[asked, col] = record[source]
    return df.assign(**out.astype({"Income_Unbroken_Years": float, "Income_Cut_Recent": bool}))


def dividend_growth_us(df: pd.DataFrame) -> pd.Series:
    """`Regular_DPS_Growth_10Y` (declared currency, nominal) in US terms: (1 + g) / (1 + `Excess_Inflation_10Y`) - 1,
    the declared currency's inflation beyond US over the same ten years (`metrics.inflation`), so the lists' bars
    mean the same in rupees as in dollars (Nov 6, 2026: 3% a year is a real fall where prices rise 6%). An unknown
    excess counts as 0: a US company, or a frame without it, keeps its figure."""
    g = _num(df, "Regular_DPS_Growth_10Y") / 100
    excess = _num(df, "Excess_Inflation_10Y").fillna(0.0).clip(lower=-0.99)
    return ((1 + g) / (1 + excess) - 1) * 100


def income_growth(df: pd.DataFrame) -> pd.Series:
    """The dividend growth the Income list counts, %/yr: the regular dividend's 10-year growth in US terms
    (`dividend_growth_us`), no faster than the business can fund from what it keeps (`ROIC_Norm` x (1 - payout),
    payout = `Dividend_Yield` / `EPV_Per_Share_Yield`; financial review, Nov 7, 2026: past growth is not future
    growth, and a company paying out 80% of its earnings cannot keep raising its dividend 8% a year), and never above
    `INCOME_GROWTH_CAP`. A blank anywhere leaves it blank."""
    payout = _num(df, "Dividend_Yield") / _num(df, "EPV_Per_Share_Yield")
    fundable = _num(df, "ROIC_Norm") * (1 - payout)
    past = _num(df, "Regular_DPS_Growth_10Y_US")
    return pd.concat([past, fundable], axis=1).min(axis=1, skipna=False).clip(upper=INCOME_GROWTH_CAP)


def income_list(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The Income list (user's choice, Nov 7, 2026). Gates: statutory withholding at most `INCOME_MAX_WHT`
    (`Dividend_WHT`); paid in each of the last `INCOME_MIN_YEARS` complete years, no year under `INCOME_HELD` of the
    one before, nor the last 12 months under it of the 12 before (`with_income_record`: specials and payment timing
    judged payment by payment, in the declared currency); the regular
    dividend no lower than 10 years ago in US terms (`dividend_growth_us`); yield after withholding at least
    `INCOME_MIN_YIELD`; normalised ROIC (ROE for financials) at least `INCOME_MIN_ROIC`; the dividend at most
    `INCOME_MAX_PAYOUT` of the earnings power a share keeps after interest (`EPV_Per_Share_Yield`; a blank fails); a
    lasting business (`_lasting_business`: not cyclical, not at peak margins, its dividend not cut more than
    `RECESSION_MAX_FALL` % in a past recession) and a sound balance sheet (`_sound_balance_sheet`); `in_best_fraction`,
    or for a bank or insurer (ranked apart, no `Rank`) the best fraction of its own kind (`kind_order`).

    Order (`Income_Order`, 1 = best): `Income_Return`, the dividend's own expected return, yield after withholding
    (`Net_Dividend_Yield`) plus the growth counted (`income_growth`); one number a reader can check by hand.
    `Loan_Spread` (that yield less the euro margin loan's cost, `Borrow_Rate`) is shown for readers who borrow for
    it, never a gate. Trusted as `ranking.top_n`. Display only."""
    df = with_income_record(with_dividend_record(df))
    kind_best = kind_order(df)[1]
    financial = kind(df).isin(("bank", "insurer"))
    net_yield = _num(df, "Dividend_Yield") * (1 - _num(df, "Dividend_WHT") / 100)
    growth = income_growth(df)
    total = net_yield + growth
    loan = _num(df, "Borrow_Rate").fillna(FALLBACK_BORROW_RATE)
    keep = (
        (_num(df, "Dividend_WHT") <= INCOME_MAX_WHT)
        & (_num(df, "Income_Unbroken_Years") >= INCOME_MIN_YEARS)  # 0 when a deep dip was paid this year
        & ~_flag(df, "Income_Cut_Recent")
        & (_num(df, "Regular_DPS_Growth_10Y_US") >= 0)
        & (net_yield >= INCOME_MIN_YIELD)
        & (_num(df, "ROIC_Norm") >= INCOME_MIN_ROIC)
        & (_num(df, "Dividend_Yield") <= INCOME_MAX_PAYOUT * _num(df, "EPV_Per_Share_Yield"))
        & total.notna()
        & _lasting_business(df)
        & _sound_balance_sheet(df)
        & (in_best_fraction(df) | (financial & kind_best))
    )
    out = df[keep].assign(Net_Dividend_Yield=net_yield[keep], Income_Growth=growth[keep], Income_Return=total[keep],
                          Loan_Spread=(net_yield - loan)[keep])  # fmt: skip
    out = out.assign(Income_Order=out["Income_Return"].rank(method="first", ascending=False))
    return top_n(out, "Income_Order", n)


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


# Figures whose blank lets a test pass untested (user's choice, Nov 7, 2026: shown, not capped): data holes, not
# blanks by design (`Incremental_ROIC` when capital barely grew, `EPS_Rev_90d` for a company no analyst follows).
UNTESTED_FIGURES = {
    "Beneish_M": "Beneish M-score",
    "Accruals": "accruals",
    "FCF_Positive_Share": "years with positive free cash flow",
    "Gross_Margin_Held": "gross margin since the inflation peak",
}
UNTESTED_TABS = ("shortlist", "compounders")  # `Tab.key`s whose rows carry the "untested" pill


def untested(row: pd.Series) -> list[str]:
    """The names of the `UNTESTED_FIGURES` missing for one company: the clean-accounts, cash and pricing-power tests
    it passed because the figure is blank (a blank passes, as in every list)."""
    return [name for col, name in UNTESTED_FIGURES.items() if pd.isna(pd.to_numeric(row.get(col), errors="coerce"))]


def nothing_breaking(df: pd.DataFrame) -> pd.Series:
    """Nothing breaking now: analysts' EPS estimates not cut more than `Settings.estimate_cut_pct` in 90 days, no
    dividend cut paid in the last 12 months (`_dividend_held`), recent revenue and EBIT not down more than the
    Dislocation minimums. A blank passes. Boring businesses asks it, and The List
    asks it in place of a price floor (user's choice, Nov 10, 2026: the business, not the share price, says whether
    something is breaking)."""
    s = Settings()
    return (
        ~(_num(df, "EPS_Rev_90d") < s.estimate_cut_pct)
        & _dividend_held(df)
        & ~(_num(df, "Revenue_Change_Recent") < s.dislocation_revenue_change_min)
        & ~(_num(df, "EBIT_Change_Recent") < s.dislocation_ebit_change_min)
    )


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
        (NOTHING_BREAKING, nothing_breaking(df)),
        ("AI: no fragile or exposed industry", ai_resilient(df)),
        ("a boring industry", _in_industries(df, BORING_INDUSTRIES)),
        ("recession: leaders only", ~(recession_prone | bank) | (leader & (bank | (net_debt <= 0)))),
        ("net debt within half of equity", net_debt_ok | financial),
        ("kept its margin through inflation", _margin_held(df)),
        # Earnings power at zero growth beats the Treasury and what more the company's own government pays to borrow
        # dollars (`Sovereign_Spread`, `universe.sovereign_spread`; user's choice, Nov 7, 2026; a frame without it: the
        # US's): `Owner_Yield`, no growth credited, so growth is the margin of safety, not part of the hurdle (Graham,
        # Greenwald; Nov 7, 2026).
        ("beats the bond", ~(num("Owner_Yield") < hurdle + num("Sovereign_Spread").fillna(0.0))),
        ("durable returns",
         (num("ROIC_Norm") >= SHORTLIST_MIN_ROIC) & (num("Value_Score") >= SHORTLIST_MIN_VALUE)
         & (num("Quality_Score") >= SHORTLIST_MIN_QUALITY)),
    ]  # fmt: skip


def shortlist_list(df: pd.DataFrame, n: int = SHORTLIST_N, hurdle: float = BOND_HURDLE_FLOOR) -> pd.DataFrame:
    """The companies worth researching in depth: `in_best_fraction` of the combined ranking and every condition of
    `_shortlist_gates` (per-share growth without dilution, owner earnings not shrinking, cash-backed and clean
    earnings, growth that earns its keep, a sound balance sheet, nothing breaking now, not fragile to AI, recession
    leaders only, net debt within half of equity, the gross margin kept through inflation, earnings power at zero
    growth above the bond `hurdle`, durable returns); trusted
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
