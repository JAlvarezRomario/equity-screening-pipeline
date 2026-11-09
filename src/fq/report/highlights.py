"""Which companies each highlight tab shows: the one registry of tabs (`TABS`) and their members, computed once per
report (`Highlights.compute`). No markup here: `report.html` renders the tabs, `fq evaluate` and the forward test follow
`Highlights.published`, and `pipeline.grade_companies` grades `Highlights.grade_pool`, so what is measured is what was shown.

To add a list: its screen and columns in `report/screens.py`, a `Tab` in `TABS` (its place is its tab position), and
a macro named after `Tab.key` in `templates/list_rules.html.j2` (its sentence and exact rules). To follow it in the
forward test, `forward.LATE_LISTS`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

from fq import grading
from fq.fx import FALLBACK_BORROW_RATE, get_borrow_rate, get_treasury_yield
from fq.metrics.capital import NCAV_EXCLUDED_INDUSTRIES
from fq.ranking import top_n
from fq.report import screens as sc
from fq.settings import Settings
from fq.universe import dividend_wht, industry_key, sovereign_spread

TOP_N = 20  # rows of each list on the index page (about 10 in view, the rest scroll in the card)
SHORTLIST = "Shortlist"
# The Shortlist's bar, stricter than the other lists' (`grading.passes_shortlist`); its pills say so.
SHORTLIST_BAR = (f"both at least {grading.MIN_BOTH:g} and either at least {grading.SHORTLIST_MIN_TOP:g}, stricter "
                 "than the other lists")  # fmt: skip
Pick = Callable[[pd.DataFrame, int], pd.DataFrame]  # (rows a list may show, rows wanted) -> the list, in its order


@dataclass(frozen=True)
class Tab:
    """One highlight tab. `screen(df, n, hurdle)` returns its members in order; `name` is the `List` value `fq evaluate` and the forward test store, so it never changes once a list is
    followed; `label` is what a reader sees when it differs (user's choice, Nov 3, 2026: "Nectar" and "Dividend carry"
    said nothing to a newcomer, but the frozen cohort keeps their names; `label()`); `panel` is its HTML id (kept
    stable: links and `main.js`); `key` names its prose macro; `group` the row of tabs it sits in (`GROUPS`)."""

    name: str
    key: str
    panel: str
    columns: list
    screen: Callable[..., pd.DataFrame]
    group: str = "style"
    label: str | None = None
    n: int = TOP_N
    sort: str = "Rank"  # the order the page shows (below the grade bar last)
    heading: str | None = None  # the section heading, when not the name
    ranked: bool = False  # a pure ranking: half the market would "qualify"
    funnel: Callable[[pd.DataFrame, float], list[tuple[str, int]]] | None = None  # counts after each test
    ai_screen: bool = True  # drawn only from companies past `screens.ai_resilient` (not the pure scores)
    checked: bool = True  # its rows carry the check of its kind (`with_check`); off for a list of checked lists
    # Its "No dividend tax" rows: the tab re-run among zero-withholding companies (True), or its own members at 0%
    # withholding (False: The List, whose sources' zero-withholding re-runs would hold companies nobody graded).
    zero_rerun: bool = True
    # Drawn only from shares not down more than `LIST_MIN_MOMENTUM` over 6 months (False: Temporary trouble, whose
    # point is a business in trouble now, which a fallen price usually shows; user's choice, Nov 9, 2026).
    price_floor: bool = True
    # A ranked walk's eligible companies (the count above it), when not what the walk returns (it stops at a row cap).
    eligible: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def _plain(fn):
    return lambda df, n, hurdle: fn(df, n)


def _hurdled(fn):
    return lambda df, n, hurdle: fn(df, n, hurdle=hurdle)


TROUBLE = "Temporary trouble"
TROUBLE_BAR = (f"both at least {grading.TROUBLE_MIN_BOTH:g} and either at least {grading.TROUBLE_MIN_TOP:g}, graded "
               "before it counts")  # fmt: skip


CHEAPEST_BAR = f"both at least {grading.CHEAPEST_MIN_BOTH:g}, graded before it counts"
BEST_BAR = f"both at least {grading.BEST_MIN_BOTH:g} and either above it, graded before it counts"
WALK_WAITING = TOP_N  # ungraded companies a graded walk holds at once: the next grading run's batch
# The score lists (Cheap and good, Cheapest, Best businesses; user's rule, Nov 9, 2026): at least this many of the rows
# shown pass every test, the list's moat and management bar and the check of its numbers; while fewer do, the list
# walks further down its ranking, a company not yet graded or checked counting as a pass until it is (so a walk
# never runs ahead of the grading), and never past `WALK_MAX_ROWS` rows.
MIN_ALL_PASS = 10
WALK_MAX_ROWS = 60


def _check_verdict(row_value) -> bool | None:
    return None if row_value is None or (isinstance(row_value, float) and pd.isna(row_value)) else bool(row_value)


def _graded_walk(df: pd.DataFrame, n: int, rank_col: str, verdict_of, bar: str) -> pd.DataFrame:
    """Down a ranking (`ranking.top_n` on `rank_col`), the companies whose moat and management pass `verdict_of`
    (True, False, or None when ungraded) until `n` of them and, of those, at least `MIN_ALL_PASS` whose earnings
    check is not a fail (`Check_Earnings_Pass`; a check not made yet counts as a pass until it is): a company graded
    below is skipped and the next one taken; one that fails its check is kept, shaded. Ungraded companies met on the
    way (at most `WALK_WAITING`) stay in the list, marked not graded, so the grading (`Highlights.grade_pool`) reaches
    them and the list fills within a few runs. At most `WALK_MAX_ROWS` rows. `Grade_Bar` is `bar`."""
    ordered = top_n(df, rank_col, len(df))
    blank = pd.Series(np.nan, index=ordered.index)
    moat = ordered["Grade_Moat"] if "Grade_Moat" in ordered.columns else blank
    mgmt = ordered["Grade_Mgmt"] if "Grade_Mgmt" in ordered.columns else blank
    check = ordered["Check_Earnings_Pass"] if "Check_Earnings_Pass" in ordered.columns else blank
    keep, verdicts, passed, sound, waiting = [], [], 0, 0, 0
    for i, m, g, c in zip(ordered.index, moat, mgmt, check, strict=True):
        if (passed >= n and sound >= min(MIN_ALL_PASS, n)) or len(keep) >= WALK_MAX_ROWS:
            break
        verdict = verdict_of(m, g)
        if verdict is None and waiting < WALK_WAITING:
            waiting += 1
            sound += 1  # it may pass both: counted until it is graded and checked
            keep.append(i)
            verdicts.append(None)  # not graded yet: keeps its place, marked, as on every list
        elif verdict:
            passed += 1
            sound += _check_verdict(c) is not False
            keep.append(i)
            verdicts.append(True)
    out = ordered.loc[keep]
    return out.assign(Grade_Pass=pd.Series(verdicts, index=out.index, dtype=object), Grade_Bar=bar)


def cheapest_list(df: pd.DataFrame, n: int, hurdle: float = 0.0) -> pd.DataFrame:
    """Cheapest (user's choice, Nov 7, 2026): down the Value ranking, moat and management both at least
    `grading.CHEAPEST_MIN_BOTH` (`grading.passes_cheapest`; `_graded_walk`), owner yield on the share price above
    `screens.CHEAP_MIN_SHARE_YIELD` (Nov 8, 2026), until at least `MIN_ALL_PASS` pass the check too (Nov 9, 2026)."""
    return _graded_walk(sc.cheap_enough(df), n, "Value_Rank", grading.passes_cheapest, CHEAPEST_BAR)


def cheap_and_good_list(df: pd.DataFrame, n: int, hurdle: float = 0.0) -> pd.DataFrame:
    """Cheap and good: the top `n` of the combined ranking (`Rank`) whose owner yield on the share price is above
    `screens.CHEAP_MIN_SHARE_YIELD` (user's choice, Nov 8, 2026), then further down while fewer than `MIN_ALL_PASS`
    of them pass every test (Nov 9, 2026): the moat and management bar (`grading.passes`) and the earnings check, a
    company not yet graded or checked counting as a pass until it is; at most `WALK_MAX_ROWS` rows. Those that fail
    stay on the list, shaded."""
    ordered = top_n(sc.cheap_enough(df), "Rank", len(df))
    blank = pd.Series(np.nan, index=ordered.index)
    moat = ordered["Grade_Moat"] if "Grade_Moat" in ordered.columns else blank
    mgmt = ordered["Grade_Mgmt"] if "Grade_Mgmt" in ordered.columns else blank
    check = ordered["Check_Earnings_Pass"] if "Check_Earnings_Pass" in ordered.columns else blank
    sound = [grading.passes(m, g) is not False and _check_verdict(c) is not False
             for m, g, c in zip(moat, mgmt, check, strict=True)]  # fmt: skip
    count, rows = 0, 0
    for ok in sound:
        if (rows >= n and count >= min(MIN_ALL_PASS, n)) or rows >= WALK_MAX_ROWS:
            break
        rows += 1
        count += ok
    return ordered.iloc[:rows]


def best_business_list(df: pd.DataFrame, n: int, hurdle: float = 0.0) -> pd.DataFrame:
    """Best businesses (user's choice, Nov 7, 2026: it absorbed Best grades): down the Quality ranking, moat and
    management both at least `grading.BEST_MIN_BOTH` and either above it (`grading.passes_best_walk`; `_graded_walk`),
    until at least `MIN_ALL_PASS` pass the check too (Nov 9, 2026). Price plays no part: a raw score; Cheap and good is
    where price and quality meet."""
    return _graded_walk(df, n, "Quality_Rank", grading.passes_best_walk, BEST_BAR)


def _eligible_cheap(rank_col: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """The companies a cheap walk may draw from: past the share-price bar, trusted and ranked on `rank_col`."""
    return lambda df: top_n(sc.cheap_enough(df), rank_col, len(df))


def _eligible_quality(df: pd.DataFrame) -> pd.DataFrame:
    """The companies Best businesses may draw from: trusted and ranked on Quality."""
    return top_n(df, "Quality_Rank", len(df))


def _trouble(df: pd.DataFrame, n: int, hurdle: float) -> pd.DataFrame:
    """Temporary trouble with `Grade_Pass` re-judged by its own bar (`grading.passes_trouble`: ungraded fails)."""
    out = sc.trouble_list(df, n, hurdle=hurdle)
    moat = out["Grade_Moat"] if "Grade_Moat" in out.columns else pd.Series(np.nan, index=out.index)
    mgmt = out["Grade_Mgmt"] if "Grade_Mgmt" in out.columns else pd.Series(np.nan, index=out.index)
    verdict = [grading.passes_trouble(m, g) for m, g in zip(moat, mgmt, strict=True)]
    return out.assign(Grade_Pass=pd.Series(verdict, index=out.index, dtype=object), Grade_Bar=TROUBLE_BAR)


def _shortlist(df: pd.DataFrame, n: int, hurdle: float) -> pd.DataFrame:
    """The Shortlist with `Grade_Pass` re-judged by its stricter gate and `Grade_Bar` naming it."""
    out = sc.shortlist_list(df, n, hurdle=hurdle)
    if "Grade_Moat" not in out.columns:
        return out
    verdict = [grading.passes_shortlist(m, g) for m, g in zip(out["Grade_Moat"], out["Grade_Mgmt"], strict=True)]
    return out.assign(Grade_Pass=pd.Series(verdict, index=out.index, dtype=object), Grade_Bar=SHORTLIST_BAR)


# The List (user's choice, Nov 9, 2026): the companies that passed every test. Each source list's rows as shown, those
# past both that list's moat and management bar and its check (both True: a missing grade or check is not a pass),
# once each, each list's best in turn (round robin; the share-price yield breaking ties).
# Nov 9, 2026 review (user's choices): Boring businesses joins the sources; one grade floor for every member (both
# at least `grading.CHEAPEST_MIN_BOTH`: a source's "either at least 4.25" route no longer lets weak management in);
# a check failed on any source keeps the company out; a member from Balance sheet plays alone (checked on its book)
# also needs its earnings checked, as The List is ordered by them; holdings left out with banks, insurers and REITs;
# the best 25 shown (user's choice), the count of all that qualify above them; ordered by each company's best place on its lists,
# so each idea gets its best few (round robin), the share-price yield breaking ties.
# Later the same day (user's choices): five sources only (Boring businesses and Income out: 62 -> 53 qualified), and
# both grades at least `THE_LIST_MIN_GRADE` (3.25: 53 -> 43; a 3.0 on either is "holds up", not more). A rule that
# any caution of the check fails was rejected: it left 4, none of them cheap.
THE_LIST = "The List"
THE_LIST_SOURCES = ("Cheap and good", "Cheapest", "Best businesses", "Below book", "Compounders")
THE_LIST_N = 25  # user's choice, Nov 9, 2026: five turns of the five source lists; the page counts every qualifier
THE_LIST_MIN_GRADE = 3.25  # moat and management both at least this, whatever the source list's own bar
# Net cash at least `Settings.net_cash_min_pct` of the market cap (the "net cash" pill's bar) needs management graded at
# least this (user's choice, Nov 9, 2026, any country: cash a management never returns earns nothing for minority
# holders, the cash-rich Japanese trap). 7 of the 43 left out on Nov 9, 2026.
THE_LIST_CASH_MGMT = 4.0
# Kinds (`screens.kind`) kept off The List: their earnings and books read differently from an operating company's.
THE_LIST_LEFT_OUT = frozenset({"bank", "insurer", "real_estate", "holding"})


def _is_true(values) -> pd.Series:
    return pd.Series([v is True or v == True for v in values], index=values.index, dtype=bool)  # noqa: E712


def _is_false(values) -> pd.Series:
    return pd.Series([v is False or v == False for v in values], index=values.index, dtype=bool)  # noqa: E712


def _graded_floor(rows: pd.DataFrame) -> pd.Series:
    """Moat and management both at least `THE_LIST_MIN_GRADE`, False when ungraded: the floor every member of The
    List clears, whatever its source's bar."""
    blank = pd.Series(np.nan, index=rows.index)
    moat = pd.to_numeric(rows.get("Grade_Moat", blank), errors="coerce")
    mgmt = pd.to_numeric(rows.get("Grade_Mgmt", blank), errors="coerce")
    return ((moat >= THE_LIST_MIN_GRADE) & (mgmt >= THE_LIST_MIN_GRADE)).astype(bool)


def the_list(df: pd.DataFrame, n: int, hurdle: float = 0.0) -> pd.DataFrame:
    """The List: the members of `THE_LIST_SOURCES` (each tab's rows as shown, `Tab.n`) past both its moat and
    management bar (`Grade_Pass`) and its check (`Check_Pass`), and past the common grade floor (`_graded_floor`),
    once each, with `The_List_From` naming the lists they came from (as readers see them). Order (`The_List_Order`,
    1 = first; user's choice, Nov 9, 2026: every idea gets its best few, not one yardstick for all): each company's
    best place among the qualifying members of the lists it is on, in each list's own order, so each list's first
    comes before any list's second; a tie goes to the higher owner yield on the share price. `The_List_Why` names
    each list with that place ("Cheapest #2 · Balance sheet plays #1"). Left out: a company whose check failed on any
    source; one that comes only from Balance sheet plays (checked on its book) without a passing earnings check
    (`Check_Earnings_Pass`); the kinds of `THE_LIST_LEFT_OUT`; net cash at least `Settings.net_cash_min_pct` % of market cap with management under
    `THE_LIST_CASH_MGMT`."""
    parts = []
    for name in THE_LIST_SOURCES:
        tab = TAB[name]
        shown = _screened(tab)(df, tab.n, hurdle)
        if not shown.empty:
            if tab.sort in shown.columns:  # the list's own order, as its page shows it
                shown = shown.sort_values(tab.sort, kind="stable", na_position="last")
            parts.append(shown.assign(_From=tab.label or tab.name, _Book=grading.check_kind(name) == grading.BOOK))

    def empty() -> pd.DataFrame:
        return df.iloc[0:0].assign(
            The_List_From=None, The_List_Why=None, The_List_Order=None, Grade_Pass=None, Check_Pass=None
        )

    if not parts:
        return empty()
    rows = pd.concat(parts)
    none = pd.Series(None, index=rows.index, dtype=object)  # a frame without grades or checks: nobody passes
    failed = set(rows.loc[_is_false(rows.get("Check_Pass", none)), "Ticker"])  # a failed check on any source
    both = rows[_is_true(rows.get("Grade_Pass", none)) & _is_true(rows.get("Check_Pass", none)) & _graded_floor(rows)]
    both = both[~both["Ticker"].isin(failed)]
    # Only from a book-checked list: its earnings, which order The List, must be checked too.
    book_only = both.groupby("Ticker", sort=False)["_Book"].transform("all").astype(bool)
    earnings = both.get("Check_Earnings_Pass", pd.Series(None, index=both.index, dtype=object))
    both = both[~book_only | _is_true(earnings)]
    # No banks, insurers, REITs or holdings (user's choices, Nov 9, 2026); fee businesses such as Visa or Moody's stay.
    model = both.get("Business_Model", pd.Series(None, index=both.index, dtype=object))
    both = both[~sc.kind(both).isin(THE_LIST_LEFT_OUT) & ~model.isin(THE_LIST_LEFT_OUT)]
    hoard = sc._num(both, "Net_Cash_to_Cap") >= Settings().net_cash_min_pct
    both = both[~hoard | (sc._num(both, "Grade_Mgmt") >= THE_LIST_CASH_MGMT)]
    if both.empty:
        return empty()
    # Place in its list among the members that qualify here (a list whose first rows fail keeps its seats).
    both = both.assign(_Place=both.groupby("_From", sort=False).cumcount())
    sources = both.groupby("Ticker", sort=False)["_From"].agg(" · ".join)
    why = (
        both.assign(_Why=both["_From"] + " #" + (both["_Place"] + 1).astype(str))
        .groupby("Ticker", sort=False)["_Why"]
        .agg(" · ".join)
    )
    place = both.groupby("Ticker", sort=False)["_Place"].min()
    out = both.drop_duplicates("Ticker").drop(columns=["_From", "_Book", "_Place"])
    out = out.assign(The_List_From=out["Ticker"].map(sources), The_List_Why=out["Ticker"].map(why),
                     _Place=out["Ticker"].map(place),
                     _Yield=-pd.to_numeric(out["EPV_Per_Share_Yield"], errors="coerce"))  # fmt: skip
    out = out.sort_values(["_Place", "_Yield"], kind="stable", na_position="last").drop(columns=["_Place", "_Yield"])
    # Counted over every qualifier before the cut: the "No dividend tax" rows (its members at 0% withholding,
    # `Tab.zero_rerun`) come from the same uncut order, so the two frames interleave right.
    return out.assign(The_List_Order=np.arange(1, len(out) + 1)).head(n)


# The front page's rows of tabs (user's choice, Nov 3, 2026: 14 tabs side by side did not say where to start): key ->
# (heading, the one line that says what such a list is for).
# The rows by idea (user's choice, Nov 6, 2026; were Start here, By style, By kind, Raw scores).
# Nov 7, 2026 (user's choice): Growth, Income and Balance sheet became one row, By style.
GROUPS = {
    # Nov 9, 2026 (user's choice): the answer, first and shown by default.
    "list": (
        "The List",
        "The best 25 companies on Cheap and good, Cheapest, Best businesses, Balance sheet plays or Compounders that "
        "also passed both AI checks: its moat and management, and its numbers.",
    ),
    "start": (
        "Start here",
        "The answer: cheap and good at once, and boring businesses worth research in depth (sound now, or in "
        "temporary trouble).",
    ),
    "style": (
        "By style",
        "Pick the kind of investment: a business compounding per share, a dividend a foreign investor keeps, a "
        "steady business whose share moves little, or a price below what the balance sheet holds; each from the best "
        "{top} of the combined ranking.",
    ),
    # Nov 5, 2026 (user's choice): the rankings hold profitable operating companies only; the other kinds here.
    "kind": (
        "By kind",
        "Banks, insurers, REITs and utilities judged against their own kind, outside the rankings.",
    ),
    "raw": (
        "Raw scores",
        "One score alone, whatever the other: the top of Value with a moat and management that hold up, and the top "
        "of Quality with the best graded moats and managers, whatever its price.",
    ),
}

# The tabs in the order the index page shows them, group by group; a company page lists its tabs in this order too.
# fmt: off
TABS: tuple[Tab, ...] = (
    Tab(THE_LIST, "the_list", "hl-the-list", sc.THE_LIST_COLUMNS, the_list, group="list", n=THE_LIST_N,
        sort="The_List_Order", checked=False, zero_rerun=False),
    Tab("Cheap and good", "cheap_and_good", "hl-top-1", sc.CHEAP_AND_GOOD_COLUMNS, cheap_and_good_list, group="start",
        ranked=True, eligible=_eligible_cheap("Rank")),
    Tab(SHORTLIST, "shortlist", "hl-shortlist", sc.SHORTLIST_COLUMNS, _shortlist, group="start", n=sc.SHORTLIST_N,
        funnel=sc.shortlist_funnel, label="Boring businesses"),
    Tab(TROUBLE, "trouble", "hl-trouble", sc.SHORTLIST_COLUMNS, _trouble, group="start", n=sc.SHORTLIST_N,
        price_floor=False),
    Tab("Compounders", "compounders", "hl-compounders", sc.COMPOUNDER_COLUMNS, _plain(sc.compounder_list)),
    Tab("Income", "income", "hl-income", sc.INCOME_COLUMNS, _plain(sc.income_list), sort="Income_Order"),
    Tab("Low beta", "low_beta", "hl-low-beta", sc.LOW_BETA_COLUMNS, _plain(sc.low_beta_list), sort="Low_Beta_Order"),
    Tab("Below book", "below_book", "hl-asset-plays", sc.ASSET_PLAY_COLUMNS, _plain(sc.asset_play_list),
        sort="Book_Order", label="Balance sheet plays"),
    Tab("Banks and lenders", "banks", "hl-banks", sc.FINANCIAL_KIND_COLUMNS, _plain(sc.bank_list), group="kind",
        sort="Kind_Rank"),
    Tab("Insurers", "insurers", "hl-insurers", sc.FINANCIAL_KIND_COLUMNS, _plain(sc.insurer_list), group="kind",
        sort="Kind_Rank"),
    Tab("Real estate", "real_estate", "hl-real-estate", sc.REAL_ESTATE_COLUMNS, _plain(sc.real_estate_list),
        group="kind", sort="Kind_Rank"),
    Tab("Utilities", "utilities", "hl-utilities", sc.UTILITY_COLUMNS, _plain(sc.utility_list), group="kind",
        sort="Kind_Rank"),
    Tab("Cheapest", "cheapest", "hl-top-2", sc.VALUE_LIST_COLUMNS, cheapest_list, group="raw",
        sort="Value_Rank", ranked=True, eligible=_eligible_cheap("Value_Rank")),
    Tab("Best businesses", "best_businesses", "hl-top-3", sc.QUALITY_LIST_COLUMNS, best_business_list, group="raw",
        sort="Quality_Rank", ranked=True, eligible=_eligible_quality),
)
# fmt: on
TAB = {t.name: t for t in TABS}
TAB_ORDER = tuple(t.name for t in TABS)
RANKED_TABS = tuple(t.name for t in TABS if t.ranked)
# Retired tabs the forward test still follows under their stored names (Nov 4, 2026: Nectar, Compounders and Better
# every year gave way to Operating companies and Fast growers; Net-nets merged into Below book; Temporary trouble
# retired, user's choice; Nov 7, 2026: Repeaters, Cannibals, Off the radar, Dividend growers, Dividend carry and Best
# grades retired before the freeze; Nov 8, 2026: Net cash, now a pill; Nov 9, 2026: Energy, user's choice, as the
# check of the numbers left none of it standing; so none is stored), with the reader's name
# where it differed.
RETIRED_LABELS = {"Nectar": "Steady growers, net cash"}


def label(name: str) -> str:
    """What a reader sees for the stored list `name` (`Tab.label`, else the name; a retired tab its `RETIRED_LABELS`
    name; a list no tab has, such as the forward test's portfolio line, keeps its own)."""
    tab = TAB.get(name)
    return (tab.label or tab.name) if tab else RETIRED_LABELS.get(name, name)


def _memo(pick: Pick) -> Pick:
    """`pick` remembering its result per (frame, n): the page, the totals and the company pages' places ask the same
    list many times. The frame is kept with its result, so its `id` cannot be reused while cached."""
    seen: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def picked(df: pd.DataFrame, n: int) -> pd.DataFrame:
        key = (id(df), n)
        if key not in seen:
            seen[key] = (df, pick(df, n))
        return seen[key][1]

    return picked


def with_check(out: pd.DataFrame, list_name: str) -> pd.DataFrame:
    """A tab's rows with `Check_Kind` (the check that judges a member of `list_name`, `grading.check_kind`) and
    `Check_Pass` (that check's verdict: True, False, or None when not made). `Grade_Pass` is left as it is: the moat
    and management bar and the check are two bars, and only an explicit False of either shades the row (`fails`)."""
    kind = grading.check_kind(list_name)
    col = f"Check_{kind.capitalize()}_Pass"
    values = out[col] if col in out.columns else pd.Series(None, index=out.index, dtype=object)
    verdict = [None if v is None or (isinstance(v, float) and pd.isna(v)) else bool(v) for v in values]
    return out.assign(Check_Kind=kind, Check_Pass=pd.Series(verdict, index=out.index, dtype=object))


def _screened(tab: Tab) -> Callable[..., pd.DataFrame]:
    """`tab.screen`, drawn only from companies past the AI screen (`screens.ai_resilient`) unless `tab.ai_screen` is
    off (every tab since Nov 9, 2026, user's choice: the score lists too; they skipped it from Nov 4); every tab's rows
    then carry the check of its kind (`with_check`)."""
    if not tab.checked:
        return tab.screen

    def pool(df: pd.DataFrame) -> pd.DataFrame:
        return df[sc.ai_resilient(df)] if tab.ai_screen else df

    if tab.ranked:  # the score lists walk on their own (`_graded_walk`, `cheap_and_good_list`)
        return lambda df, n, hurdle: with_check(tab.screen(pool(df), n, hurdle), tab.name)
    return lambda df, n, hurdle: _fill(with_check(tab.screen(pool(df), len(df), hurdle), tab.name), n)


def _fill(out: pd.DataFrame, n: int) -> pd.DataFrame:
    """A screened list's rows in its order, down to the `n`-th that does not fail (`fails`: a moat and management grade
    or a check below the bar), at most `WALK_MAX_ROWS` (user's choice, Nov 9, 2026: one rule for every list, as the
    walks had; Insurers showed 17 shaded of 20 while 16 more qualified). The failures met on the way stay, folded
    under the list; ungraded and unchecked companies keep their place and count until they are judged."""
    if n >= len(out):  # every qualifier asked for (`Highlights.total`): no row cap
        return out
    ok = (~fails(out)).to_numpy().cumsum()
    reach = int(np.searchsorted(ok, n)) + 1 if len(ok) and ok[-1] >= n else len(out)
    return out.iloc[: min(reach, WALK_MAX_ROWS)]


def highlight_picks(hurdle: float) -> dict[str, tuple[Pick, int]]:
    """Every tab: name -> (pick(df, n), rows shown), in `TAB_ORDER`, each remembering its results (`_memo`).
    `hurdle` is the owner yield the Shortlist asks for (`screens.bond_hurdle`)."""
    return {t.name: (_memo(partial(_screened(t), hurdle=hurdle)), t.n) for t in TABS}


def _false(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].map(lambda v: v is False or v == False).astype(bool)  # noqa: E712


def fails(df: pd.DataFrame) -> pd.Series:
    """True where the moat and management grade is below the bar (`grading.passes`) or the check of the list's kind
    failed (`Check_Pass`); an ungraded or unchecked company is not."""
    return _false(df, "Grade_Pass") | _false(df, "Check_Pass")


def prepare(df: pd.DataFrame, bench: pd.DataFrame | None, cfg: Settings | None = None) -> pd.DataFrame:
    """The columns the lists read beyond the ranking: withholding, the run's best-fraction cut, the dividend record
    (`screens.with_dividend_record`, Income's own `screens.with_income_record`), industry leaders and (with `cfg`) the stored moat and management grades
    (`grading.attach`)."""
    if cfg is not None:
        df = grading.attach(df, cfg.data_dir / "grades.parquet")
        df = grading.attach_checks(df, cfg.data_dir / "checks.parquet")
    if "NCAV_to_Price" in df.columns and "industry" in df.columns:
        land = df["industry"].map(lambda i: industry_key(i) in NCAV_EXCLUDED_INDUSTRIES)
        df = df.assign(NCAV_to_Price=df["NCAV_to_Price"].where(~land, np.nan))
    countries = df["country"] if "country" in df.columns else [None] * len(df)
    df = df.assign(Dividend_WHT=[dividend_wht(t, c) for t, c in zip(df["Ticker"], countries, strict=True)])
    df = df.assign(Sovereign_Spread=[sovereign_spread(c) for c in countries])  # the Boring businesses bond hurdle
    df = df.assign(In_Best_Fraction=sc.in_best_fraction(df))  # the run's cut, kept by every subset a list is drawn from
    kind_rank, kind_best = sc.kind_order(df)  # the same for the kinds ranked apart
    df = df.assign(Kind_Rank=kind_rank, In_Kind_Best=kind_best)
    df = sc.with_income_record(sc.with_dividend_record(df))  # judged once per report, not once per list call
    # Leaders of their industry among the benchmark's companies (the Shortlist's recession test), the same in every run.
    leaders_among = df
    if bench is not None and not bench.empty:  # the benchmark, plus the run's companies outside it
        leaders_among = pd.concat([bench, df[~df["Ticker"].isin(bench["Ticker"])]])
    df = df.assign(Industry_Leader=sc.industry_leader(df, leaders_among))
    if "Compounder" in df.columns:
        comp_idx = sc.compounder_list(df[sc.ai_resilient(df)], len(df)).index
        df = df.assign(Compounder=df["Compounder"].fillna(False).astype(bool) | df.index.isin(comp_idx))
    return df


# A share that fell more than this % over 6 months (`Momentum_6M`) is on no list (user's choice, Nov 9, 2026: first on
# The List, the same day on every list: a falling price can be news the figures do not show yet). Display only: the
# second place the price path is read, after Low beta's beta, never a score, rank or flag; a blank passes. On The
# List, 8 of 43 on Nov 9, 2026.
LIST_MIN_MOMENTUM = -20.0


def listable(df: pd.DataFrame, cfg: Settings, price_floor: bool = True) -> pd.DataFrame:
    """The rows a highlight list may show: median daily traded value (`Traded_Value_USD`) at least
    `list_min_traded_usd`, a slice nobody can buy beats nothing, and (`price_floor`) a share price not down more than
    `LIST_MIN_MOMENTUM` % over 6 months (`Momentum_6M`); a missing figure passes. The full ranking keeps all."""
    traded = pd.to_numeric(df["Traded_Value_USD"], errors="coerce") if "Traded_Value_USD" in df.columns else None
    keep = pd.Series(True, index=df.index) if traded is None else ~(traded < cfg.list_min_traded_usd)
    return df[keep & ~(price_floor & (sc._num(df, "Momentum_6M") < LIST_MIN_MOMENTUM))]


@dataclass
class Highlights:
    """Every tab's members for one report. `frame` is the run with the columns the lists add (`prepare`), which the
    full ranking and the company pages read too; `lists` its rows a list may show (`listable`); `zero` those of
    zero-withholding countries (the "No dividend tax" switch's extra rows)."""

    frame: pd.DataFrame
    lists: pd.DataFrame
    liquid: pd.DataFrame  # `lists` without the price floor (`Tab.price_floor` False)
    treasury_yield: float | None
    hurdle: float
    picks: dict[str, tuple[Pick, int]]
    borrow_rate: float = FALLBACK_BORROW_RATE  # what a euro margin loan costs, % (`fx.get_borrow_rate`)
    zero: pd.DataFrame = field(init=False)
    liquid_zero: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.zero = self.lists[self.lists["Dividend_WHT"] == 0]
        self.liquid_zero = self.liquid[self.liquid["Dividend_WHT"] == 0]

    def _rows(self, name: str, zero: bool = False) -> pd.DataFrame:
        """The rows tab `name` may draw from: with or without the price floor (`Tab.price_floor`), zero-withholding
        only when `zero`."""
        floor = TAB[name].price_floor
        return (self.zero if floor else self.liquid_zero) if zero else (self.lists if floor else self.liquid)

    @classmethod
    def compute(cls, df: pd.DataFrame, cfg: Settings, bench: pd.DataFrame | None = None) -> Highlights:
        frame = prepare(df, bench, cfg)
        treasury = get_treasury_yield(cfg.data_dir / "treasury_yield.json")
        hurdle = sc.bond_hurdle(treasury)
        borrow = get_borrow_rate(cfg.data_dir / "borrow_rate.json")
        frame = frame.assign(Borrow_Rate=borrow)  # Income's loan spread (shown), kept by every subset
        liquid = listable(frame, cfg, price_floor=False)
        return cls(frame, listable(frame, cfg), liquid, treasury, hurdle, highlight_picks(hurdle), borrow)

    def shown(self, name: str, zero: bool = False) -> pd.DataFrame:
        """The rows tab `name` shows (`zero`: among zero-withholding companies only), in the list's own order."""
        pick, n = self.picks[name]
        if zero and not TAB[name].zero_rerun:
            rows = self._rows(name)
            members = pick(rows, len(rows))
            return members[members["Dividend_WHT"] == 0].head(n)
        return pick(self._rows(name, zero), n)

    def total(self, name: str) -> int:
        """How many companies qualify for tab `name`, however many rows it shows (a ranked walk: how many it may draw
        from, `Tab.eligible`, past the AI screen)."""
        tab, rows = TAB[name], self._rows(name)
        if tab.eligible is not None:
            return len(tab.eligible(rows[sc.ai_resilient(rows)] if tab.ai_screen else rows))
        return len(self.picks[name][0](rows, len(rows)))

    def funnel(self, name: str) -> list[tuple[str, int]] | None:
        tab = TAB[name]
        return None if tab.funnel is None else tab.funnel(self._rows(name), self.hurdle)

    def listed(self) -> set[str]:
        """Every company a tab shows (the zero-withholding extras left out)."""
        return {t for name in TAB_ORDER for t in self.shown(name)["Ticker"]}

    def published(self) -> pd.DataFrame:
        """(Ticker, List, Grade_Pass, Check_Pass): the companies each tab shows, for `fq evaluate` and the forward test
        to follow and for the grading (`pipeline.grade_companies`) to grade. `Grade_Pass` is the moat and management
        gate (`grading.passes`; None ungraded), `Check_Pass` the check of the tab's kind (None unchecked): a company
        failing either stays on the tab, shaded."""
        rows = []
        for name in TAB_ORDER:
            shown = self.shown(name)
            checks = shown["Check_Pass"] if "Check_Pass" in shown.columns else [None] * len(shown)
            rows += list(zip(shown["Ticker"], [name] * len(shown), shown["Grade_Pass"], checks, strict=True))
        return pd.DataFrame(rows, columns=["Ticker", "List", "Grade_Pass", "Check_Pass"])

    def places(self) -> dict[str, str]:
        """Ticker -> "Cheap and good #4 · Compounders #2": every tab that shows the company and its row there, as the
        index page shows it (below the grade bar last), for the company page's "Why it's here" strip."""
        out: dict[str, list[str]] = {}
        for tab in TABS:
            shown = self.shown(tab.name)
            shown = shown.assign(_fails=fails(shown)).sort_values(["_fails", tab.sort], kind="stable")
            for i, ticker in enumerate(shown["Ticker"], 1):
                out.setdefault(ticker, []).append(f"{tab.label or tab.name} #{i}")
        return {t: " · ".join(p) for t, p in out.items()}

    def grade_pool(self) -> list[str]:
        """The companies the grading must grade: every one a tab shows."""
        return list(dict.fromkeys(self.published()["Ticker"]))

    def check_pool(self) -> list[tuple[str, str]]:
        """The (ticker, kind) checks the lists need: every company a tab shows that clears that tab's moat and
        management bar (`Grade_Pass` True), checked on its book on Balance sheet plays, on its earnings elsewhere; a
        Balance sheet plays member on its earnings too, which The List orders by (`the_list`)."""
        pub = self.published()
        passed = pub[pub["Grade_Pass"].map(lambda v: v is True or v == True)]  # noqa: E712
        kept_off = set(self.frame.loc[sc.kind(self.frame).isin(THE_LIST_LEFT_OUT), "Ticker"])  # never on The List
        pairs = []
        for t, name in zip(passed["Ticker"], passed["List"], strict=True):
            kind = grading.check_kind(name)
            if kind is not None:
                pairs.append((t, kind))
            if kind == grading.BOOK and name in THE_LIST_SOURCES and t not in kept_off:
                pairs.append((t, grading.EARNINGS))
        return list(dict.fromkeys(pairs))


def published_lists(df: pd.DataFrame, cfg: Settings, bench: pd.DataFrame | None = None) -> pd.DataFrame:
    """`Highlights.published` for a ranked run (computes every list; reuse a `Highlights` when one is at hand)."""
    return Highlights.compute(df, cfg, bench).published()
