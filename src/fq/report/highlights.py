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
from fq.ranking import top_n, trusted
from fq.report import screens as sc
from fq.settings import Settings
from fq.universe import dividend_wht, industry_key, sovereign_spread

TOP_N = 20  # rows of each list on the index page (about 10 in view, the rest scroll in the card)
SHORTLIST = "Shortlist"
REPEATERS = "Repeaters"
BEST_GRADES = "Best grades"
DERIVED = (REPEATERS, BEST_GRADES)  # tabs that follow the other lists, never counted by Repeaters as a style
# The Shortlist's bar, stricter than the other lists' (`grading.passes_shortlist`); its pills say so.
SHORTLIST_BAR = (f"both at least {grading.MIN_BOTH:g} and either at least {grading.SHORTLIST_MIN_TOP:g}, stricter "
                 "than the other lists")  # fmt: skip
_REPEAT_STEP = 1e6  # Repeaters' order: more lists first, then the combined rank (Rank stays far below this step)
# Two screens of the same thesis family count once when deciding whether a company is a repeater across styles,
# though `Repeat_Lists` and `Repeat_Count` still show every list it qualifies for.
_REPEAT_FAMILIES = {
    "Compounders": "compounder",
    "Below book": "balance_sheet",
    "Net cash": "balance_sheet",
    "Dividend carry": "income",
    "Dividend growers": "income",
    "Banks and lenders": "financial",
    "Insurers": "financial",
}

Pick = Callable[[pd.DataFrame, int], pd.DataFrame]  # (rows a list may show, rows wanted) -> the list, in its order


@dataclass(frozen=True)
class Tab:
    """One highlight tab. `screen(df, n, hurdle)` returns its members in order (None for Repeaters, built from the
    others); `name` is the `List` value `fq evaluate` and the forward test store, so it never changes once a list is
    followed; `label` is what a reader sees when it differs (user's choice, Nov 3, 2026: "Nectar" and "Dividend carry"
    said nothing to a newcomer, but the frozen cohort keeps their names; `label()`); `panel` is its HTML id (kept
    stable: links and `main.js`); `key` names its prose macro; `group` the row of tabs it sits in (`GROUPS`)."""

    name: str
    key: str
    panel: str
    columns: list
    screen: Callable[..., pd.DataFrame] | None
    group: str = "growth"
    label: str | None = None
    n: int = TOP_N
    sort: str = "Rank"  # the order the page shows (below the grade bar last)
    heading: str | None = None  # the section heading, when not the name
    ranked: bool = False  # a pure ranking: half the market would "qualify", so Repeaters count only its rows shown
    funnel: Callable[[pd.DataFrame, float], list[tuple[str, int]]] | None = None  # counts after each test
    ai_screen: bool = True  # drawn only from companies past `screens.ai_resilient` (not the pure scores)


def _top(col: str):
    return lambda df, n, hurdle: top_n(df, col, n)


def _plain(fn):
    return lambda df, n, hurdle: fn(df, n)


def _hurdled(fn):
    return lambda df, n, hurdle: fn(df, n, hurdle=hurdle)


TROUBLE = "Temporary trouble"
TROUBLE_BAR = (f"both at least {grading.TROUBLE_MIN_BOTH:g} and either at least {grading.TROUBLE_MIN_TOP:g}, graded "
               "before it counts")  # fmt: skip


CHEAPEST_BAR = f"both at least {grading.CHEAPEST_MIN_BOTH:g}, graded before it counts"
CHEAPEST_WAITING = TOP_N  # ungraded companies the Cheapest walk holds at once: the next grading run's batch


def cheapest_list(df: pd.DataFrame, n: int, hurdle: float = 0.0) -> pd.DataFrame:
    """Cheapest (user's choice, Nov 7, 2026): down the Value ranking (`ranking.top_n` on `Value_Rank`), the companies
    whose moat and management are both at least `grading.CHEAPEST_MIN_BOTH` (`grading.passes_cheapest`) until `n`
    of them; a company graded below is skipped and the next one taken. Ungraded companies met on the way (at most
    `CHEAPEST_WAITING`) stay in the list, marked not graded, so the grading (`Highlights.grade_pool`) reaches them and the list
    fills to `n` within a few runs."""
    ordered = top_n(df, "Value_Rank", len(df))
    moat = ordered["Grade_Moat"] if "Grade_Moat" in ordered.columns else pd.Series(np.nan, index=ordered.index)
    mgmt = ordered["Grade_Mgmt"] if "Grade_Mgmt" in ordered.columns else pd.Series(np.nan, index=ordered.index)
    keep, verdicts, passed, waiting = [], [], 0, 0
    for i, m, g in zip(ordered.index, moat, mgmt, strict=True):
        if passed >= n:
            break
        verdict = grading.passes_cheapest(m, g)
        if verdict is None and waiting < CHEAPEST_WAITING:
            waiting += 1
            keep.append(i)
            verdicts.append(None)  # not graded yet: keeps its place, marked, as on every list
        elif verdict:
            passed += 1
            keep.append(i)
            verdicts.append(True)
    out = ordered.loc[keep]
    return out.assign(Grade_Pass=pd.Series(verdicts, index=out.index, dtype=object), Grade_Bar=CHEAPEST_BAR)


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


# The front page's rows of tabs (user's choice, Nov 3, 2026: 14 tabs side by side did not say where to start): key ->
# (heading, the one line that says what such a list is for).
# The rows by idea (user's choice, Nov 6, 2026; were Start here, By style, By kind, Raw scores).
GROUPS = {
    "start": (
        "Start here",
        "The answer: cheap and good at once, boring businesses worth research in depth (sound now, or in temporary "
        "trouble), and the best graded.",
    ),
    "growth": ("Growth", "Businesses compounding per share, from the best {top} of the combined ranking."),
    "income": ("Income", "Dividends to count on: held for a decade and growing, or untaxed for a non-resident."),
    "balance": ("Balance sheet", "Value held on the balance sheet: a pile of net cash, or a price at or below book."),
    # Nov 5, 2026 (user's choice): the rankings hold profitable operating companies only; the other kinds here.
    "kind": (
        "By kind",
        "Banks, insurers, REITs and utilities judged against their own kind, outside the rankings; "
        "and energy, ranked, seen apart.",
    ),
    "raw": (
        "Raw scores",
        "One score alone, whatever the other: the top of Value (with a moat and management that hold up), the top of "
        "Quality.",
    ),
}

# The tabs in the order the index page shows them, group by group; a company page lists its tabs in this order too.
# fmt: off
TABS: tuple[Tab, ...] = (
    Tab("Cheap and good", "cheap_and_good", "hl-top-1", sc.TOP_LIST_COLUMNS, _top("Rank"), group="start",
        ranked=True, ai_screen=False),
    Tab(SHORTLIST, "shortlist", "hl-shortlist", sc.SHORTLIST_COLUMNS, _shortlist, group="start", n=sc.SHORTLIST_N,
        funnel=sc.shortlist_funnel, label="Boring businesses"),
    Tab(TROUBLE, "trouble", "hl-trouble", sc.SHORTLIST_COLUMNS, _trouble, group="start", n=sc.SHORTLIST_N),
    Tab(REPEATERS, "repeaters", "hl-repeaters", sc.REPEATER_COLUMNS, None, group="start", sort="Repeat_Key"),
    Tab(BEST_GRADES, "best_grades", "hl-best-grades", sc.BEST_GRADE_COLUMNS,
        lambda df, n, hurdle: best_grade_list(df, n), group="start",
        sort="Best_Order"),
    Tab("Compounders", "compounders", "hl-compounders", sc.COMPOUNDER_COLUMNS, _plain(sc.compounder_list),
        group="growth"),
    Tab("Cannibals", "cannibals", "hl-cannibals", sc.CANNIBAL_COLUMNS, _plain(sc.cannibal_list), group="growth",
        heading="The Cannibals"),
    Tab("Off the radar", "off_radar", "hl-radar", sc.OFF_RADAR_COLUMNS, _plain(sc.off_radar_list), group="growth"),
    Tab("Dividend growers", "dividend_growers", "hl-dividend-growers", sc.DIVIDEND_GROWER_COLUMNS,
        _plain(sc.dividend_grower_list), group="income"),
    Tab("Dividend carry", "carry", "hl-carry", sc.CARRY_COLUMNS, _plain(sc.dividend_carry_list), group="income",
        label="Untaxed dividends", sort="Carry_Order"),
    Tab("Net cash", "net_cash", "hl-net-cash", sc.NET_CASH_COLUMNS, _plain(sc.net_cash_list), group="balance"),
    Tab("Below book", "below_book", "hl-asset-plays", sc.ASSET_PLAY_COLUMNS, _plain(sc.asset_play_list),
        group="balance", sort="Book_Order"),
    Tab("Banks and lenders", "banks", "hl-banks", sc.FINANCIAL_KIND_COLUMNS, _plain(sc.bank_list), group="kind",
        sort="Kind_Rank"),
    Tab("Insurers", "insurers", "hl-insurers", sc.FINANCIAL_KIND_COLUMNS, _plain(sc.insurer_list), group="kind",
        sort="Kind_Rank"),
    Tab("Real estate", "real_estate", "hl-real-estate", sc.REAL_ESTATE_COLUMNS, _plain(sc.real_estate_list),
        group="kind", sort="Kind_Rank"),
    Tab("Utilities", "utilities", "hl-utilities", sc.UTILITY_COLUMNS, _plain(sc.utility_list), group="kind",
        sort="Kind_Rank"),
    Tab("Energy", "energy", "hl-energy", sc.ENERGY_COLUMNS, _plain(sc.energy_list), group="kind"),
    Tab("Cheapest", "cheapest", "hl-top-2", sc.VALUE_LIST_COLUMNS, cheapest_list, group="raw",
        sort="Value_Rank", ranked=True, ai_screen=False),
    Tab("Best businesses", "best_businesses", "hl-top-3", sc.QUALITY_LIST_COLUMNS, _top("Quality_Rank"), group="raw",
        sort="Quality_Rank", ranked=True, ai_screen=False),
)
# fmt: on
TAB = {t.name: t for t in TABS}
TAB_ORDER = tuple(t.name for t in TABS)
RANKED_TABS = tuple(t.name for t in TABS if t.ranked)
# Retired tabs the forward test still follows under their stored names (Nov 4, 2026: Nectar, Compounders and Better
# every year gave way to Operating companies and Fast growers; Net-nets merged into Below book; Temporary trouble
# retired, user's choice), with the reader's name where it differed.
RETIRED_LABELS = {"Nectar": "Steady growers, net cash"}


def label(name: str) -> str:
    """What a reader sees for the stored list `name` (`Tab.label`, else the name; a retired tab its `RETIRED_LABELS`
    name; a list no tab has, such as the forward test's portfolio line, keeps its own)."""
    tab = TAB.get(name)
    return (tab.label or tab.name) if tab else RETIRED_LABELS.get(name, name)


def _memo(pick: Pick) -> Pick:
    """`pick` remembering its result per (frame, n): the page, the totals, the company pages' places and Repeaters ask
    the same list many times. The frame is kept with its result, so its `id` cannot be reused while cached."""
    seen: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def picked(df: pd.DataFrame, n: int) -> pd.DataFrame:
        key = (id(df), n)
        if key not in seen:
            seen[key] = (df, pick(df, n))
        return seen[key][1]

    return picked


def repeater_candidates(df: pd.DataFrame, picks: dict[str, tuple[Pick, int]]) -> pd.DataFrame:
    """Companies on two or more screened lists across at least two distinct styles (`_REPEAT_FAMILIES`), with
    `Repeat_Count`, `Repeat_Lists` (in `TAB_ORDER`) and `Repeat_Key`, most lists first, then the combined rank;
    `Repeat_Lists` names them as the page does (`label`). A screened list counts every company that qualifies and does
    not fail that list's own grade bar (`fails`); the pure rankings (`RANKED_TABS`) are left out."""
    member: dict[str, list[str]] = {}
    for name in TAB_ORDER:
        if name in DERIVED or name in RANKED_TABS or name not in picks:
            continue
        pick, _ = picks[name]
        qual = pick(df, len(df))
        for ticker in qual[~fails(qual)]["Ticker"]:
            member.setdefault(ticker, []).append(name)
    many = {t: names for t, names in member.items() if len({_REPEAT_FAMILIES.get(n, n) for n in names}) >= 2}
    out = df[df["Ticker"].isin(many)]
    count = out["Ticker"].map(lambda t: len(many[t])).astype(float)  # float even when empty (else text)
    out = out.assign(Repeat_Count=count, Repeat_Lists=out["Ticker"].map(lambda t: ", ".join(map(label, many[t]))),
                     Repeat_Key=-count * _REPEAT_STEP + out["Rank"].fillna(_REPEAT_STEP - 1))  # fmt: skip
    return out.sort_values("Repeat_Key", kind="stable")


def repeater_list(df: pd.DataFrame, n: int, picks: dict[str, tuple[Pick, int]]) -> pd.DataFrame:
    """The Repeaters tab: `repeater_candidates` that pass the moat and management bar (`Grade_Pass`); ungraded or
    below it, a company is left out (user's rule: they must pass), so the tab has no shaded rows."""
    out = repeater_candidates(df, picks)
    if "Grade_Pass" not in out.columns:  # nothing graded: nothing passes
        return out.head(0)
    passed = out["Grade_Pass"].map(lambda v: v is True or v == True).astype(bool)  # noqa: E712 (empty: rows, not columns)
    return out[passed].head(n)


def best_grade_list(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The Best grades tab (user's choice, Nov 7, 2026): every graded company (each was graded because a list showed
    it) with trustworthy figures (`ranking.trusted`) whose moat and management grades are both at least
    `grading.BEST_MIN_BOTH` and either above it (`grading.passes_best`), highest owner yield first (`Best_Order`).
    It shows its companies, so `Highlights.grade_pool` keeps their grades current."""
    if "Grade_Moat" not in df.columns:  # nothing graded: nothing passes
        return df.head(0).assign(Best_Order=pd.Series(dtype=float))
    best = [grading.passes_best(m, g) for m, g in zip(df["Grade_Moat"], df["Grade_Mgmt"], strict=True)]
    out = df[pd.Series(best, index=df.index, dtype=bool) & trusted(df)]
    order = -pd.to_numeric(out["Owner_Yield"], errors="coerce").fillna(-np.inf)
    return out.assign(Best_Order=order).sort_values("Best_Order", kind="stable").head(n)


def _screened(tab: Tab) -> Callable[..., pd.DataFrame]:
    """`tab.screen`, drawn only from companies past the AI screen (`screens.ai_resilient`) unless `tab.ai_screen` is
    off (the pure score rankings: Cheap and good, Cheapest and Best businesses; user's choice, Nov 4, 2026)."""
    if not tab.ai_screen:
        return tab.screen
    return lambda df, n, hurdle: tab.screen(df[sc.ai_resilient(df)], n, hurdle)


def highlight_picks(hurdle: float) -> dict[str, tuple[Pick, int]]:
    """Every tab: name -> (pick(df, n), rows shown), in `TAB_ORDER`, each remembering its results (`_memo`).
    `hurdle` is the owner yield the Shortlist asks for (`screens.bond_hurdle`)."""
    picks = {t.name: (_memo(partial(_screened(t), hurdle=hurdle)), t.n) for t in TABS if t.screen is not None}
    repeaters = (_memo(partial(repeater_list, picks=dict(picks))), TAB[REPEATERS].n)
    return {name: repeaters if name == REPEATERS else picks[name] for name in TAB_ORDER}


def fails(df: pd.DataFrame) -> pd.Series:
    """True where the moat and management grade is below the bar (`grading.passes`); an ungraded company is not."""
    if "Grade_Pass" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["Grade_Pass"].map(lambda v: v is False or v == False).astype(bool)  # noqa: E712


def prepare(df: pd.DataFrame, bench: pd.DataFrame | None, cfg: Settings | None = None) -> pd.DataFrame:
    """The columns the lists read beyond the ranking: withholding, the run's best-fraction cut, the dividend record
    (`screens.with_dividend_record`), industry leaders and (with `cfg`) the stored moat and management grades
    (`grading.attach`)."""
    if cfg is not None:
        df = grading.attach(df, cfg.data_dir / "grades.parquet")
    if "NCAV_to_Price" in df.columns and "industry" in df.columns:
        land = df["industry"].map(lambda i: industry_key(i) in NCAV_EXCLUDED_INDUSTRIES)
        df = df.assign(NCAV_to_Price=df["NCAV_to_Price"].where(~land, np.nan))
    countries = df["country"] if "country" in df.columns else [None] * len(df)
    df = df.assign(Dividend_WHT=[dividend_wht(t, c) for t, c in zip(df["Ticker"], countries, strict=True)])
    df = df.assign(Sovereign_Spread=[sovereign_spread(c) for c in countries])  # the Boring businesses bond hurdle
    df = df.assign(In_Best_Fraction=sc.in_best_fraction(df))  # the run's cut, kept by every subset a list is drawn from
    kind_rank, kind_best = sc.kind_order(df)  # the same for the kinds ranked apart
    df = df.assign(Kind_Rank=kind_rank, In_Kind_Best=kind_best)
    df = sc.with_dividend_record(df)  # judged once per report, not once per list call
    # Leaders of their industry among the benchmark's companies (the Shortlist's recession test), the same in every run.
    leaders_among = df
    if bench is not None and not bench.empty:  # the benchmark, plus the run's companies outside it
        leaders_among = pd.concat([bench, df[~df["Ticker"].isin(bench["Ticker"])]])
    return df.assign(Industry_Leader=sc.industry_leader(df, leaders_among))


def listable(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """The rows a highlight list may show: median daily traded value (`Traded_Value_USD`) at least
    `list_min_traded_usd`, a slice nobody can buy beats nothing (a missing figure passes). The full ranking keeps all."""
    if "Traded_Value_USD" not in df.columns:
        return df
    return df[~(pd.to_numeric(df["Traded_Value_USD"], errors="coerce") < cfg.list_min_traded_usd)]


@dataclass
class Highlights:
    """Every tab's members for one report. `frame` is the run with the columns the lists add (`prepare`), which the
    full ranking and the company pages read too; `lists` its rows a list may show (`listable`); `zero` those of
    zero-withholding countries (the "No dividend tax" switch's extra rows)."""

    frame: pd.DataFrame
    lists: pd.DataFrame
    treasury_yield: float | None
    hurdle: float
    picks: dict[str, tuple[Pick, int]]
    borrow_rate: float = FALLBACK_BORROW_RATE  # what a euro margin loan costs, % (`fx.get_borrow_rate`)
    zero: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.zero = self.lists[self.lists["Dividend_WHT"] == 0]

    @classmethod
    def compute(cls, df: pd.DataFrame, cfg: Settings, bench: pd.DataFrame | None = None) -> Highlights:
        frame = prepare(df, bench, cfg)
        treasury = get_treasury_yield(cfg.data_dir / "treasury_yield.json")
        hurdle = sc.bond_hurdle(treasury)
        borrow = get_borrow_rate(cfg.data_dir / "borrow_rate.json")
        frame = frame.assign(Borrow_Rate=borrow)  # the Untaxed dividends hurdle, kept by every subset
        return cls(frame, listable(frame, cfg), treasury, hurdle, highlight_picks(hurdle), borrow)

    def shown(self, name: str, zero: bool = False) -> pd.DataFrame:
        """The rows tab `name` shows (`zero`: among zero-withholding companies only), in the list's own order."""
        pick, n = self.picks[name]
        return pick(self.zero if zero else self.lists, n)

    def total(self, name: str) -> int:
        """How many companies qualify for tab `name`, however many rows it shows."""
        return len(self.picks[name][0](self.lists, len(self.lists)))

    def funnel(self, name: str) -> list[tuple[str, int]] | None:
        tab = TAB[name]
        return None if tab.funnel is None else tab.funnel(self.lists, self.hurdle)

    def candidates(self) -> pd.DataFrame:
        """Every company on two or more lists, graded or not (`repeater_candidates`)."""
        return repeater_candidates(self.lists, {k: v for k, v in self.picks.items() if k not in DERIVED})

    def listed(self) -> set[str]:
        """Every company a tab shows (the zero-withholding extras left out)."""
        return {t for name in TAB_ORDER for t in self.shown(name)["Ticker"]}

    def published(self) -> pd.DataFrame:
        """(Ticker, List, Grade_Pass): the companies each tab shows, for `fq evaluate` and the forward test to follow
        and for the grading (`pipeline.grade_companies`) to grade. `Grade_Pass` is the moat and management gate (`grading.passes`;
        None ungraded): a company below it stays on the tab, shaded."""
        rows = [(t, name, g) for name in TAB_ORDER
                for t, g in zip(self.shown(name)["Ticker"], self.shown(name)["Grade_Pass"], strict=True)]  # fmt: skip
        return pd.DataFrame(rows, columns=["Ticker", "List", "Grade_Pass"])

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
        """The companies the grading must grade: every one a tab shows and every Repeaters candidate (Repeaters
        shows only those that pass, so they are graded before they can enter)."""
        return list(dict.fromkeys([*self.published()["Ticker"], *self.candidates()["Ticker"]]))


def published_lists(df: pd.DataFrame, cfg: Settings, bench: pd.DataFrame | None = None) -> pd.DataFrame:
    """`Highlights.published` for a ranked run (computes every list; reuse a `Highlights` when one is at hand)."""
    return Highlights.compute(df, cfg, bench).published()
