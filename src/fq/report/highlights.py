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

import pandas as pd

from fq import grading
from fq.fx import get_treasury_yield
from fq.ranking import top_n
from fq.report import screens as sc
from fq.settings import Settings
from fq.universe import dividend_wht

TOP_N = 20  # rows of each list on the index page (about 10 in view, the rest scroll in the card)
SHORTLIST = "Shortlist"
REPEATERS = "Repeaters"
# The Shortlist's bar, stricter than the other lists' (`grading.passes_shortlist`); its pills say so.
SHORTLIST_BAR = (f"both at least {grading.MIN_BOTH:g} and either at least {grading.SHORTLIST_MIN_TOP:g}, stricter "
                 "than the other lists")  # fmt: skip
_REPEAT_STEP = 1e6  # Repeaters' order: more lists first, then the combined rank (Rank stays far below this step)

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
    group: str = "style"
    label: str | None = None
    n: int = TOP_N
    sort: str = "Rank"  # the order the page shows (below the grade bar last)
    heading: str | None = None  # the section heading, when not the name
    ranked: bool = False  # a pure ranking: half the market would "qualify", so Repeaters count only its rows shown
    funnel: Callable[[pd.DataFrame, float], list[tuple[str, int]]] | None = None  # counts after each test


def _top(col: str):
    return lambda df, n, hurdle: top_n(df, col, n)


def _plain(fn):
    return lambda df, n, hurdle: fn(df, n)


def _hurdled(fn):
    return lambda df, n, hurdle: fn(df, n, hurdle=hurdle)


def _shortlist(df: pd.DataFrame, n: int, hurdle: float) -> pd.DataFrame:
    """The Shortlist with `Grade_Pass` re-judged by its stricter gate and `Grade_Bar` naming it."""
    out = sc.shortlist_list(df, n, hurdle=hurdle)
    if "Grade_Moat" not in out.columns:
        return out
    verdict = [grading.passes_shortlist(m, g) for m, g in zip(out["Grade_Moat"], out["Grade_Mgmt"], strict=True)]
    return out.assign(Grade_Pass=pd.Series(verdict, index=out.index, dtype=object), Grade_Bar=SHORTLIST_BAR)


# The front page's rows of tabs (user's choice, Nov 3, 2026: 14 tabs side by side did not say where to start): key ->
# (heading, the one line that says what such a list is for).
GROUPS = {
    "start": ("Start here", "The answer: cheap and good at once, then the candidates for research in depth."),
    "style": ("By style", "Lenses on the best {top} of the combined ranking, each asking one question of its own."),
    "raw": ("Raw scores", "One score alone, whatever the other: the top of Value, the top of Quality."),
}

# The tabs in the order the index page shows them, group by group; a company page lists its tabs in this order too.
# fmt: off
TABS: tuple[Tab, ...] = (
    Tab("Cheap and good", "cheap_and_good", "hl-top-1", sc.TOP_LIST_COLUMNS, _top("Rank"), group="start", ranked=True),
    Tab(SHORTLIST, "shortlist", "hl-shortlist", sc.SHORTLIST_COLUMNS, _shortlist, group="start", n=sc.SHORTLIST_N,
        funnel=sc.shortlist_funnel),
    Tab(REPEATERS, "repeaters", "hl-repeaters", sc.REPEATER_COLUMNS, None, group="start", sort="Repeat_Key"),
    Tab("Operating companies", "operating", "hl-operating", sc.OPERATING_COLUMNS, _plain(sc.operating_list)),
    Tab("Fast growers", "fast_growers", "hl-fast", sc.FAST_COLUMNS, _plain(sc.fast_grower_list)),
    Tab("Cannibals", "cannibals", "hl-cannibals", sc.CANNIBAL_COLUMNS, _plain(sc.cannibal_list),
        heading="The Cannibals"),
    Tab("Below book", "below_book", "hl-asset-plays", sc.ASSET_PLAY_COLUMNS, _plain(sc.asset_play_list)),
    Tab("Net-nets", "net_nets", "hl-net-nets", sc.NET_NET_COLUMNS, _plain(sc.net_net_list)),
    Tab("Net cash", "net_cash", "hl-net-cash", sc.NET_CASH_COLUMNS, _plain(sc.net_cash_list)),
    Tab("Off the radar", "off_radar", "hl-radar", sc.OFF_RADAR_COLUMNS, _plain(sc.off_radar_list)),
    Tab("Dividend carry", "carry", "hl-carry", sc.CARRY_COLUMNS, _plain(sc.dividend_carry_list),
        label="Untaxed dividends"),
    Tab("Temporary trouble", "trouble", "hl-trouble", sc.TROUBLE_COLUMNS, _hurdled(sc.trouble_list),
        funnel=sc.trouble_funnel),
    Tab("Cheapest", "cheapest", "hl-top-2", sc.VALUE_LIST_COLUMNS, _top("Value_Rank"), group="raw",
        sort="Value_Rank", ranked=True),
    Tab("Best businesses", "best_businesses", "hl-top-3", sc.QUALITY_LIST_COLUMNS, _top("Quality_Rank"), group="raw",
        sort="Quality_Rank", ranked=True),
)
# fmt: on
TAB = {t.name: t for t in TABS}
TAB_ORDER = tuple(t.name for t in TABS)
RANKED_TABS = tuple(t.name for t in TABS if t.ranked)
# Retired tabs the forward test still follows under their stored names (Nov 4, 2026: Nectar, Compounders and Better
# every year gave way to Operating companies and Fast growers), with the reader's name where it differed.
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
    """Companies on two or more lists, whatever their grade, with `Repeat_Count`, `Repeat_Lists` (in `TAB_ORDER`) and
    `Repeat_Key`, most lists first, then the combined rank; `Repeat_Lists` names them as the page does (`label`). A
    screened list counts every company that qualifies, not only the rows it shows; the ranked lists (`RANKED_TABS`)
    only the rows they show."""
    member: dict[str, list[str]] = {}
    for name in TAB_ORDER:
        if name == REPEATERS or name not in picks:
            continue
        pick, n = picks[name]
        for ticker in pick(df, n if name in RANKED_TABS else len(df))["Ticker"]:
            member.setdefault(ticker, []).append(name)
    many = {t: names for t, names in member.items() if len(names) >= 2}
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


def highlight_picks(hurdle: float) -> dict[str, tuple[Pick, int]]:
    """Every tab: name -> (pick(df, n), rows shown), in `TAB_ORDER`, each remembering its results (`_memo`).
    `hurdle` is the owner yield the Shortlist and Temporary trouble ask for (`screens.bond_hurdle`)."""
    picks = {t.name: (_memo(partial(t.screen, hurdle=hurdle)), t.n) for t in TABS if t.screen is not None}
    repeaters = (_memo(partial(repeater_list, picks=dict(picks))), TAB[REPEATERS].n)
    return {name: repeaters if name == REPEATERS else picks[name] for name in TAB_ORDER}


def fails(df: pd.DataFrame) -> pd.Series:
    """True where the moat and management grade is below the bar (`grading.passes`); an ungraded company is not."""
    if "Grade_Pass" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Grade_Pass"].map(lambda v: v is False or v == False)  # noqa: E712


def prepare(df: pd.DataFrame, bench: pd.DataFrame | None, cfg: Settings | None = None) -> pd.DataFrame:
    """The columns the lists read beyond the ranking: withholding, the run's best-fraction cut, industry leaders and
    (with `cfg`) the stored moat and management grades (`grading.attach`)."""
    if cfg is not None:
        df = grading.attach(df, cfg.data_dir / "grades.parquet")
    countries = df["country"] if "country" in df.columns else [None] * len(df)
    df = df.assign(Dividend_WHT=[dividend_wht(t, c) for t, c in zip(df["Ticker"], countries, strict=True)])
    df = df.assign(In_Best_Fraction=sc.in_best_fraction(df))  # the run's cut, kept by every subset a list is drawn from
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
    zero: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.zero = self.lists[self.lists["Dividend_WHT"] == 0]

    @classmethod
    def compute(cls, df: pd.DataFrame, cfg: Settings, bench: pd.DataFrame | None = None) -> Highlights:
        frame = prepare(df, bench, cfg)
        treasury = get_treasury_yield(cfg.data_dir / "treasury_yield.json")
        hurdle = sc.bond_hurdle(treasury)
        return cls(frame, listable(frame, cfg), treasury, hurdle, highlight_picks(hurdle))

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
        return repeater_candidates(self.lists, {k: v for k, v in self.picks.items() if k != REPEATERS})

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
