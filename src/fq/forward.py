"""Backward and forward test: the highlight lists frozen on one date, held equal-weight, followed monthly against
two ETFs.

The lists on the site move with every price and every report, so they cannot say on their own whether a list
worked. The lists as published at `START` are frozen (`freeze_cohort`, never rewritten), each bought in equal
amounts per company and held without trading (buy and hold: the weights drift with prices). At every completed
month-end the USD total return (dividends reinvested on the ex-date, gross of withholding tax, as Yahoo's adjusted
closes) of each member is stored (`ledger`), with its price-only return beside it (`Price_Return`, from the closes
adjusted for splits only: the gap is what the dividends added). Each list and the portfolio is compared with
`BENCHMARKS` (S&P 500 and total world ETFs), measured the same way. Pure functions except the parquet I/O and
`update`, which takes the price and FX downloads as arguments.

The ledger runs from `BACK_MONTHS` before `START` (`back_start`): the months up to `START` are the backward test
(the frozen lists were picked knowing those prices, so it is hindsight, not evidence), the months after it the
forward test (`phases`).

Yahoo back-adjusts history on every dividend and forgets delisted tickers. A ratio of two adjusted closes survives
the back-adjustment, so each month's return is stored once and never recomputed: whatever Yahoo forgets later stays
in the ledger. A member without a price after a mark (delisted, a Yahoo gap) is held at its last value (`Stale`).

Only the members that pass the moat and management grade are held (user's choice, Oct 31, 2026): the gate as graded
on the day the cohort is frozen (`FROZEN_ON`) is frozen once in `GATE_FILE` (`freeze_gate`) and applied to the lists, not to the portfolio
(`apply_gate`). The ledger still measures every member, so the record stays whole.

Because a stored month is final, what is stored is guarded (Nov 1, 2026). A download that failed is not stored and is
retried (`update`, `MAX_STALE_SHARE`); a member whose price came back catches up in its next month from its last priced
mark (`_anchors`). A jump of `UNIT_BREAK` in a day is a change of quote unit and is left out (`price_breaks`). Yahoo's
later corrections (a split or dividend recorded late) are found by measuring the last stored month again and carried
into the new month, never by rewriting a row (`reconcile`; one month back only). A month whose dividends look
implausible (`DIVIDEND_JUMP`) is kept but checked. Whatever these touch is marked `Check` and named on the page.

Lists born after the freeze are added once each (`LATE_LISTS`): the first in `ADDED_FILE`, each later publication date in a
file of its own (`ADDED_GLOB`), every one written once, bought at the same close: allowed only when no
price after it reached the list's choice (ranked from a cache written before the main markets traded again).
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fq.fetch import PRICE_BREAK
from fq.fx import MINOR_UNITS
from fq.grading import passes_shortlist

log = logging.getLogger(__name__)

# The pivot (user's choice, Nov 6, 2026: the test restarts on the lists as refined by then): November's month-end mark,
# i.e. the Nov 30, 2026 close, with the lists published on Dec 1 (`FROZEN_ON`). Backward before it, forward after it.
# The first test (Oct 30 close) is kept in `archive_2026-10-30_cohort/`, not shown.
START = pd.Timestamp("2026-11-30")
BENCHMARKS = {"SPY": "S&P 500 (SPY)", "VT": "Total world (VT)"}  # USD ETFs, total return
WORLD = "VT"  # the market each list is ranked against
PORTFOLIO_LIST = "Portfolio (equal weight)"
# Tabs renamed since the cohort was frozen: today's name -> the cohort's (none since the Nov 2026 re-freeze).
RENAMED: dict[str, str] = {}
SHORTLIST_COHORT = "Shortlist"  # the Shortlist's name in the frozen cohort: the stricter grade gate applies to it
BACK_MONTHS = 6  # the backward test's span (user's choice), month-end to month-end
CHUNK = 400  # tickers per price request
LOOKBACK_DAYS = 10  # the download starts this long before the first mark, so a holiday still has a close
GATE_FILE = "grade_gate.parquet"  # the moat and management gate, frozen once (`freeze_gate`)
ADDED_FILE = "cohort_added.parquet"  # the lists added after the freeze, written once (`freeze_late_lists`)
ADDED_GLOB = "cohort_added_*.parquet"  # later additions, one file per publication date, each written once
# List -> the date of its saved tabs. Repeaters (born Oct 31) was first saved on Sunday Nov 1, before any main market
# traded after the pivot's close. It admits only companies that pass the grade, so the frozen gate does not apply.
# Nectar on its final rules (user's choice, Nov 2, 2026), ranked from the same cache as the re-frozen cohort, so no later
# price reached it. Its first freeze, on earlier rules, is kept in `archive_nectar_first_rules/`.
# A list born after the freeze can be added (`LATE_LISTS`: name -> the date of its saved tabs), bought at `START` or at
# a pivot of its own (`LATE_PIVOTS`: name -> a later month-end), measured apart from the others (`pivot_groups`).
# None since the restart.
LATE_PIVOTS: dict[str, pd.Timestamp] = {}
LATE_LISTS: dict[str, pd.Timestamp] = {}
# The cohort is frozen only from the tabs saved on this day (`pipeline.forward_step`; a day keeps its last report's
# tabs, so every benchmark report of Dec 1 must rank the cache written before the markets opened that day), at the
# first online run after it. Until those tabs exist the site shows a placeholder (`forward_page.build_placeholder`).
FROZEN_ON = pd.Timestamp("2026-12-01")
MAX_STALE_SHARE = 0.2  # above this share of tickers without a price, a month is a failed download: not stored, retried
UNIT_BREAK = (
    20.0  # a one-day move this large (either way) is a change of quote unit (BCG.L, pence to euros), not a return
)
# A stored month differing this much from the same month measured again: Yahoo corrected something. Small (Nov 3,
# 2026; was 0.05, sized for splits), so a dividend Yahoo records after the month-end run (0.3-3% typically) is not lost.
RECONCILE_GAP = 0.002
DIVIDEND_JUMP = 0.15  # a month's dividends above this share of the price (or below -1%): a Yahoo dividend to check
DIVIDEND_SLACK = 0.01  # adjusted closes a little below the plain ones: rounding, not worth a check
COHORT_COLUMNS = ("List", "Ticker", "Position", "Name", "Country", "Currency", "Value_Score", "Quality_Score")

# `fetch.total_and_price_history` (columns (Total|Price, ticker)); a plain frame of adjusted closes also works, with
# `Price_Return` left unknown.
PriceFetch = Callable[[list[str], pd.Timestamp], pd.DataFrame]
FxHistory = Callable[[str], pd.Series | None]  # `fx._usd_history`


def freeze_cohort(
    published: pd.DataFrame,
    ranked: pd.DataFrame,
    held: frozenset[str] | set[str],
    start: pd.Timestamp,
    published_on: pd.Timestamp,
    method: str = "",
) -> pd.DataFrame:
    """The lists to follow: each published tab (`highlights.published_lists`: Ticker, List, in the order shown), the
    portfolio's holdings, with what was known of each member that day."""
    ranked = ranked.drop_duplicates("Ticker").set_index("Ticker")
    order = ranked["Rank"].sort_values(kind="stable").index if "Rank" in ranked.columns else ranked.index
    held_sorted = [t for t in order if t in held] + sorted(set(held) - set(order))
    parts = [published[["List", "Ticker"]]]
    parts.append(pd.DataFrame({"List": PORTFOLIO_LIST, "Ticker": held_sorted}))
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["List", "Ticker"])
    out["Position"] = out.groupby("List", sort=False).cumcount() + 1
    known = {"Name": "shortName", "Country": "country", "Currency": "currency",
             "Value_Score": "Value_Score", "Quality_Score": "Quality_Score"}  # fmt: skip
    for col, src in known.items():
        out[col] = out["Ticker"].map(ranked[src]) if src in ranked.columns else None
    if "financialCurrency" in ranked.columns:  # Yahoo leaves the quote currency out for a few (1325.SR)
        out["Currency"] = out["Currency"].fillna(out["Ticker"].map(ranked["financialCurrency"]))
    return out.assign(Start=start, Published_On=published_on, Method=method).reset_index(drop=True)


def save_cohort(cohort: pd.DataFrame, path: Path) -> Path:
    """Write the cohort once: an existing one is never replaced (the test is only honest frozen)."""
    if path.exists():
        raise FileExistsError(f"{path} exists: the forward test's cohort is frozen and is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(path, index=False)
    return path


def freeze_gate(grades: pd.DataFrame, path: Path) -> Path:
    """Write the grade gate once (Ticker, Moat, Management, Passes, Graded_At): a later re-grade never moves the
    test's membership. `grades` is `grading.load`'s frame with `Passes` added by the caller."""
    if path.exists():
        raise FileExistsError(f"{path} exists: the forward test's grade gate is frozen and is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    grades[["Ticker", "Moat", "Management", "Passes", "Graded_At"]].to_parquet(path, index=False)
    return path


def apply_gate(cohort: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    """The cohort less the list members below the moat and management bar (the Shortlist's stricter one on its list,
    `grading.passes_shortlist`, from the frozen grades); ungraded members and the portfolio stay.
    `attrs["gated_out"]` counts the companies left out, `attrs["gate_date"]` is the last grade's date."""
    failed = set(gate.loc[gate["Passes"].eq(False), "Ticker"])
    strict = [passes_shortlist(m, g) for m, g in zip(gate["Moat"], gate["Management"], strict=True)]
    failed_strict = set(gate.loc[pd.Series(strict, index=gate.index, dtype=object).eq(False), "Ticker"])
    on_shortlist = cohort["List"].isin([SHORTLIST_COHORT, RENAMED.get(SHORTLIST_COHORT, SHORTLIST_COHORT)])
    out_rows = (
        cohort["List"].ne(PORTFOLIO_LIST)
        & ~cohort["List"].isin(LATE_LISTS)
        & (cohort["Ticker"].isin(failed) | (on_shortlist & cohort["Ticker"].isin(failed_strict)))
    )
    out = cohort.loc[~out_rows].reset_index(drop=True)
    out.attrs["gated_out"] = cohort.loc[out_rows, "Ticker"].nunique()
    out.attrs["gate_date"] = pd.to_datetime(gate["Graded_At"]).max()
    return out


def freeze_late_lists(
    folder: Path,
    late: dict[str, pd.DataFrame],
    ranked: pd.DataFrame,
    start: pd.Timestamp,
    method: str = "",
    grades: pd.DataFrame | None = None,
) -> None:
    """Freeze each list of `LATE_LISTS` no added file holds yet, a publication date at a time once `late` holds every
    list of that date (its saved tabs: Ticker, List, in the order shown): the cohort's pivot `start`, or the list's own
    (`LATE_PIVOTS`), the list's own publication date, less the members `grades` (with `Passes`) fail, as the frozen gate does not cover them. The first
    addition goes to `ADDED_FILE`, each later date to a file of its own (`ADDED_GLOB`), so no stored file is ever
    rewritten; `cohort.parquet` is not touched."""
    have = {name for path in _added_files(folder) for name in pd.read_parquet(path, columns=["List"])["List"]}
    due = {name: when for name, when in LATE_LISTS.items() if name not in have}
    failed = set() if grades is None else set(grades.loc[grades["Passes"].eq(False), "Ticker"])
    for when in sorted(set(due.values())):
        names = [name for name, w in due.items() if w == when]
        if not set(names) <= set(late):
            continue
        path = folder / ADDED_FILE
        if path.exists():
            path = folder / ADDED_GLOB.replace("*", f"{when:%Y-%m-%d}")
        held = {
            name: late[name].loc[(late[name]["List"] == name) & ~late[name]["Ticker"].isin(failed)] for name in names
        }
        parts = [freeze_cohort(held[name], ranked, set(), LATE_PIVOTS.get(name, start), when, method) for name in names]
        save_cohort(pd.concat(parts, ignore_index=True), path)
        log.info("Forward test: %s added", ", ".join(names))


def _added_files(folder: Path) -> list[Path]:
    """The stored additions to the cohort, oldest first."""
    first = [folder / ADDED_FILE] if (folder / ADDED_FILE).exists() else []
    return first + sorted(folder.glob(ADDED_GLOB))


def load_cohort(folder: Path) -> pd.DataFrame:
    """The frozen cohort plus the lists added after it."""
    parts = [pd.read_parquet(folder / "cohort.parquet"), *(pd.read_parquet(p) for p in _added_files(folder))]
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def pivot_groups(cohort: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """(pivot, its lists) for each pivot the cohort holds, the cohort's own (`START`, the portfolio's) first: a list
    bought later is measured from its own close, against the benchmarks over the same months, never beside the others."""
    return [(pd.Timestamp(when), part) for when, part in cohort.groupby("Start", sort=True)]


def back_start(start: pd.Timestamp) -> pd.Timestamp:
    """The backward test's first mark: the month-end `BACK_MONTHS` before `start`."""
    return start - pd.DateOffset(months=BACK_MONTHS) + pd.offsets.MonthEnd(0)


def last_close(mark: pd.Timestamp) -> pd.Timestamp:
    """The weekday on or before a month-end mark: the close it stands for (holidays aside), as the page names it."""
    return pd.offsets.BDay().rollback(mark)


def phases(ledger: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(backward, forward): the ledger's marks up to `start` and after it."""
    after = ledger["Date"] > start
    return ledger[~after], ledger[after]


def month_ends(start: pd.Timestamp, today: pd.Timestamp) -> list[pd.Timestamp]:
    """Every month-end after `start` that is over by `today` (its close is known), oldest first."""
    ends = pd.date_range(start, today.normalize(), freq="ME")
    return [d for d in ends if d > start.normalize() and d < today.normalize()]


def to_usd(prices: pd.DataFrame, currencies: dict[str, str | None], fx_history: FxHistory) -> tuple[pd.DataFrame, list]:
    """Quote-currency closes in USD at each day's rate (minor units such as GBp and KWF divided out), and the
    tickers left in their own currency because the rate has no history (reported, never silent)."""
    out = prices.sort_index().copy()
    missing = []
    for ticker in out.columns:
        ccy = currencies.get(ticker)
        if not isinstance(ccy, str):  # unknown: never taken for dollars
            missing.append(ticker)
            continue
        base, units = MINOR_UNITS.get(ccy, (ccy, 1))
        if base == "USD":
            out[ticker] = out[ticker] / units
            continue
        hist = fx_history(base)
        if hist is None or hist.empty:
            missing.append(ticker)
            continue
        rate = hist.sort_index().reindex(out.index.union(hist.index)).ffill().reindex(out.index)
        out[ticker] = out[ticker] * rate / units
    if missing:
        log.warning("No FX history for %s: their returns stay in quote currency", ", ".join(missing))
    return out, missing


def price_breaks(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closes, breaks) from quote-currency closes. A one-day move of at least `UNIT_BREAK` is a change of unit: the
    closes before it are put on the new basis, so that day earns nothing. One of `PRICE_BREAK` up to that may be real (a
    collapse), so it stays. `breaks` (Date, Ticker) lists both kinds, to be checked."""
    out = prices.sort_index().copy()
    found = []
    for ticker in out.columns:
        close = out[ticker].dropna()
        step = (close / close.shift()).where(lambda v: v > 0)
        size = step.where(step >= 1, 1 / step)
        found += [(day, ticker) for day in size.index[size >= PRICE_BREAK]]
        for day in size.index[size >= UNIT_BREAK]:
            out.loc[out.index < day, ticker] *= step[day]
    return out, pd.DataFrame(found, columns=["Date", "Ticker"])


def _anchors(ledger: pd.DataFrame, start: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Tickers without a price at the last stored mark -> the mark to take their entry price from: their last priced
    one (none yet: the ledger's first), or `start` once the ledger is past it (the forward test never earns what came
    before it)."""
    last = ledger["Date"].max()
    priced = ledger.loc[~ledger["Stale"].astype(bool)].groupby("Ticker")["Date"].max()
    out = {}
    for ticker in _stale_now(ledger):
        anchor = priced.get(ticker, back_start(start))
        out[ticker] = max(anchor, start) if last > start else anchor
    return out


def period_returns(
    prices: pd.DataFrame, marks: list[pd.Timestamp], anchors: dict[str, pd.Timestamp] | None = None
) -> pd.DataFrame:
    """(Date, Ticker, Return, Stale) for each step between consecutive `marks`: the last close on or before each mark.
    A ticker with no close after the earlier mark (delisted, a gap) or none on or before it (no entry price) earns
    0 for that step and is `Stale`: held at its last value. In the first step a ticker of `anchors` enters at its anchor
    date's close instead, so a month it missed is earned then."""
    prices = prices.sort_index()
    rows = []
    for i, (a, b) in enumerate(zip(marks[:-1], marks[1:], strict=True)):
        before, until = prices.loc[:a], prices.loc[:b]
        pa = before.ffill().iloc[-1].copy() if len(before) else pd.Series(np.nan, index=prices.columns)
        pb = until.ffill().iloc[-1] if len(until) else pd.Series(np.nan, index=prices.columns)
        fresh = prices.loc[(prices.index > a) & (prices.index <= b)].notna().any()
        for ticker, anchor in (anchors or {}).items() if i == 0 else ():
            if ticker in prices.columns:
                entry = prices.loc[:anchor, ticker].ffill()
                pa[ticker] = entry.iloc[-1] if len(entry) else np.nan
                fresh[ticker] = prices.loc[(prices.index > anchor) & (prices.index <= b), ticker].notna().any()
        ok = fresh & (pa > 0) & (pb > 0)
        ret = (pb / pa.where(pa > 0) - 1).where(ok, 0.0)
        rows.append(pd.DataFrame({"Date": b, "Ticker": prices.columns, "Return": ret.values, "Stale": ~ok.values}))
    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "Return", "Stale"])
    return pd.concat(rows, ignore_index=True)


def extend_ledger(ledger: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Stored marks are kept as they are; only new (Date, Ticker) pairs are added."""
    if ledger is None or ledger.empty:
        return new.reset_index(drop=True)
    both = pd.concat([ledger, new], ignore_index=True)
    return both.drop_duplicates(["Date", "Ticker"], keep="first").sort_values(["Date", "Ticker"], ignore_index=True)


def growth(ledger: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Each ticker's value of 1 invested at `start`, at `start` and every mark (index Date, a column per ticker)."""
    if ledger.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([start], name="Date"))
    wide = ledger.pivot(index="Date", columns="Ticker", values="Return").sort_index().fillna(0.0)
    value = (1 + wide).cumprod()
    first = pd.DataFrame(1.0, index=pd.DatetimeIndex([start]), columns=value.columns)
    return pd.concat([first, value]).rename_axis("Date")


def series_values(cohort: pd.DataFrame, ledger: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Cumulative USD total return (%) of every list (equal weight at entry, then held: the mean of its members'
    values) and of each benchmark, at `start` (0) and every mark. Columns: lists in cohort order, then benchmarks.
    A member with no ledger row at all (never priced, so no entry price) is left out of its list's mean, not held as
    cash; `summary`'s `Members` still counts it."""
    value = growth(ledger, start)
    out = {}
    for name, members in cohort.groupby("List", sort=False)["Ticker"]:
        have = [t for t in members if t in value.columns]
        out[name] = value[have].mean(axis=1) if have else pd.Series(np.nan, index=value.index)
    for ticker, label in BENCHMARKS.items():
        out[label] = value[ticker] if ticker in value.columns else pd.Series(np.nan, index=value.index)
    return (pd.DataFrame(out) - 1) * 100


def price_values(cohort: pd.DataFrame, ledger: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame | None:
    """`series_values` on the price-only returns (dividends left out), or None when a row lacks them (months stored
    before Nov 3, 2026, or a download without plain closes)."""
    if ledger.empty or "Price_Return" not in ledger.columns or ledger["Price_Return"].isna().any():
        return None
    return series_values(cohort, ledger.assign(Return=ledger["Price_Return"]), start)


def _stale_now(ledger: pd.DataFrame) -> set[str]:
    """Tickers without a price at the last stored mark."""
    if ledger.empty:
        return set()
    last = ledger[ledger["Date"] == ledger["Date"].max()]
    return set(last.loc[last["Stale"].astype(bool), "Ticker"])


def _max_drawdown(cum_pct: pd.Series) -> float:
    wealth = 1 + cum_pct.dropna() / 100
    return float(((wealth / wealth.cummax()) - 1).min() * 100) if len(wealth) else np.nan


def summary(
    values: pd.DataFrame, cohort: pd.DataFrame, ledger: pd.DataFrame, live: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per series: return to date, the part of it the dividends added (points: total minus price-only return;
    only when every row has its price-only return), excess over each benchmark (points), annualised return (once a
    year has passed), worst fall from a high (monthly marks), share of months ahead of the world ETF, share of members
    ahead of it, stale members and, with `live` (today's published lists), how many members are still on their list."""
    start, last = values.index[0], values.index[-1]
    months = (last.year - start.year) * 12 + (last.month - start.month)
    years = (last - start).days / 365.25
    world = values.get(BENCHMARKS[WORLD])
    step = (1 + values / 100) / (1 + values.shift() / 100) - 1  # each month's return
    member_value = growth(ledger, start).iloc[-1] - 1 if not ledger.empty else pd.Series(dtype=float)
    stale_now = _stale_now(ledger)
    live_pairs = set(zip(live["List"].replace(RENAMED), live["Ticker"], strict=True)) if live is not None else None
    world_member = member_value.get(WORLD, np.nan)
    price = price_values(cohort, ledger, start)
    rows = []
    for name in values.columns:
        ret = values[name].iloc[-1]
        members = cohort.loc[cohort["List"] == name, "Ticker"]
        row = {"Series": name, "Kind": "benchmark" if name in BENCHMARKS.values() else "list",
               "Members": len(members) or np.nan, "Return": ret}  # fmt: skip
        if price is not None:
            row["Dividends"] = ret - price[name].iloc[-1]
        for ticker, label in BENCHMARKS.items():
            row[f"vs {ticker}"] = ret - values[label].iloc[-1] if label in values.columns else np.nan
        row["Annualised"] = ((1 + ret / 100) ** (1 / years) - 1) * 100 if years >= 1 and pd.notna(ret) else np.nan
        row["Max_Drawdown"] = _max_drawdown(values[name])
        if world is not None and months and name != BENCHMARKS[WORLD]:  # VT against itself says nothing
            ahead = (step[name] > step[BENCHMARKS[WORLD]]).iloc[1:]
            row["Months_Ahead"] = ahead.mean() * 100
        if len(members) and pd.notna(world_member):
            row["Members_Ahead"] = (member_value.reindex(members) > world_member).mean() * 100
        row["Stale"] = int(members.isin(stale_now).sum()) if len(members) else np.nan
        if live_pairs is not None and row["Kind"] == "list" and name != PORTFOLIO_LIST:
            row["Still_Listed"] = sum((name, t) in live_pairs for t in members)
        rows.append(row)
    return pd.DataFrame(rows).assign(Months=months, Start=start, Last=last)


def member_table(cohort: pd.DataFrame, ledger: pd.DataFrame, live: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every frozen member: its list, entry position, backward and forward return (%; forward from its list's own
    pivot, `pivot_groups`), stale and still on its list today."""
    start = cohort["Start"].min()
    back, _ = phases(ledger, start)
    out = cohort.assign(Stale=cohort["Ticker"].isin(_stale_now(ledger)))
    value = growth(back, back_start(start)).iloc[-1] - 1 if not back.empty else pd.Series(dtype=float)
    out["Return_Back"] = out["Ticker"].map(value * 100)
    out["Return_Fwd"] = np.nan
    for when, part in pivot_groups(out):
        _, fwd = phases(ledger, when)
        value = growth(fwd, when).iloc[-1] - 1 if not fwd.empty else pd.Series(dtype=float)
        out.loc[part.index, "Return_Fwd"] = part["Ticker"].map(value * 100)
    if live is not None:
        pairs = set(zip(live["List"].replace(RENAMED), live["Ticker"], strict=True))
        out["Still_Listed"] = [(a, t) in pairs for a, t in zip(out["List"], out["Ticker"], strict=True)]
    return out


def download(tickers: list[str], since: pd.Timestamp, fetch: PriceFetch) -> pd.DataFrame:
    """Adjusted closes of `tickers` since `since`, in chunks of `CHUNK` (one request each), joined on date."""
    frames = []
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i : i + CHUNK]
        try:
            frames.append(fetch(part, since))
        except Exception as exc:  # a failed chunk: its tickers are stale this month, the rest go on
            log.warning("Forward test: prices for %d tickers failed (%s)", len(part), exc)
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.loc[:, ~out.columns.duplicated()]


def _currencies(cohort: pd.DataFrame) -> dict[str, str | None]:
    out = dict(zip(cohort["Ticker"], cohort["Currency"], strict=True))
    return out | dict.fromkeys(BENCHMARKS, "USD")


def _download_usd(cohort: pd.DataFrame, since: pd.Timestamp, fetch: PriceFetch, fx_history: FxHistory):
    """(USD total-return closes of every member and benchmark, unit breaks left out; the same for the closes adjusted
    for splits only, None when the fetch gave none; `price_breaks`' breaks of the total-return closes; tickers
    without FX)."""
    tickers = sorted(set(cohort["Ticker"]) | set(BENCHMARKS))
    raw = download(tickers, since - pd.Timedelta(days=LOOKBACK_DAYS), fetch)
    kinds = set(raw.columns.get_level_values(0)) if isinstance(raw.columns, pd.MultiIndex) else set()
    total = (raw["Total"] if "Total" in kinds else raw).reindex(columns=tickers)
    total, breaks = price_breaks(total)
    total, no_fx = to_usd(total, _currencies(cohort), fx_history)
    plain = None
    if "Price" in kinds:
        plain, _ = price_breaks(raw["Price"].reindex(columns=tickers))
        plain, _ = to_usd(plain, _currencies(cohort), fx_history)
    return total, plain, breaks, no_fx


def _price_returns(
    plain: pd.DataFrame | None, marks: list[pd.Timestamp], anchors: dict[str, pd.Timestamp] | None = None
) -> pd.DataFrame | None:
    """(Date, Ticker, Price_Return) over the same steps as the total return; unknown where the plain close is stale."""
    if plain is None:
        return None
    out = period_returns(plain, marks, anchors)
    return out.assign(Price_Return=out["Return"].where(~out["Stale"].astype(bool)))[["Date", "Ticker", "Price_Return"]]


def _rows(
    returns: pd.DataFrame,
    marks: list[pd.Timestamp],
    breaks: pd.DataFrame,
    no_fx: list,
    price: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ledger rows: `Measured` keeps the month as measured (`Return` may carry a correction), `Price_Return` the
    price-only return (`_price_returns`; 0 for a stale member, unknown without plain closes), `Check` marks a month
    with a price break or with dividends outside [-`DIVIDEND_SLACK`, `DIVIDEND_JUMP`] of the price (a dividend Yahoo
    recorded in the wrong unit or size)."""
    index = pd.DatetimeIndex(marks)
    flagged = {(marks[index.searchsorted(d)], t) for d, t in zip(breaks["Date"], breaks["Ticker"], strict=True)
               if marks[0] < d <= marks[-1]}  # fmt: skip
    check = [(d, t) in flagged for d, t in zip(returns["Date"], returns["Ticker"], strict=True)]
    out = returns.assign(No_FX=returns["Ticker"].isin(no_fx), Check=check, Measured=returns["Return"])
    if price is None:
        return out.assign(Price_Return=np.nan)
    out = out.merge(price, on=["Date", "Ticker"], how="left")
    out["Price_Return"] = out["Price_Return"].where(~out["Stale"].astype(bool), 0.0)
    dividends = (1 + out["Return"]) / (1 + out["Price_Return"]) - 1
    out["Check"] = out["Check"] | dividends.gt(DIVIDEND_JUMP) | dividends.lt(-DIVIDEND_SLACK)
    return out


def _with_guards(ledger: pd.DataFrame) -> pd.DataFrame:
    """A ledger stored before the guards: old rows gain the columns (`Price_Return` unknown), their returns stay."""
    if "Check" not in ledger.columns:
        ledger = ledger.assign(Check=False)
    if "Price_Return" not in ledger.columns:
        ledger = ledger.assign(Price_Return=np.nan)
    return ledger if "Measured" in ledger.columns else ledger.assign(Measured=ledger["Return"])


def _marks(ledger: pd.DataFrame, start: pd.Timestamp) -> list[pd.Timestamp]:
    """The first mark and every stored one."""
    return [back_start(start), *(pd.Timestamp(d) for d in sorted(ledger["Date"].unique()))]


def reconcile(ledger: pd.DataFrame, prices: pd.DataFrame, start: pd.Timestamp) -> dict[str, float]:
    """Ticker -> factor still owed on the last stored month: that month measured again over as stored, where the two
    differ by more than `RECONCILE_GAP`. A ratio of adjusted closes survives a later dividend, so a gap means Yahoo
    changed something (a split or a dividend recorded late, a corrected close). Not for a month without prices, then
    or now, or one that was catching up (it spans more than a month). Compares with `Measured`, so a correction carried
    once is not carried again. Only the last stored month is measured again: a correction found later is not carried."""
    prev, last = _marks(ledger, start)[-2:]
    stored = ledger[ledger["Date"] == last].set_index("Ticker")
    caught_up = set(ledger.loc[(ledger["Date"] == prev) & ledger["Stale"].astype(bool), "Ticker"])
    again = period_returns(prices, [prev, last]).set_index("Ticker")
    out = {}
    for ticker in again.index[~again["Stale"].astype(bool)].intersection(stored.index):
        if stored.at[ticker, "Stale"] or ticker in caught_up:
            continue
        factor = (1 + again.at[ticker, "Return"]) / (1 + stored.at[ticker, "Measured"])
        if abs(factor - 1) > RECONCILE_GAP:
            out[ticker] = float(factor)
    return out


def _failed_download(new: pd.DataFrame) -> bool:
    """A month with a benchmark unpriced, or too many members: Yahoo failed, nothing may be stored."""
    for day, part in new.groupby("Date"):
        share = part["Stale"].astype(bool).mean()
        no_bench = part.loc[part["Ticker"].isin(BENCHMARKS), "Stale"].astype(bool).any()
        if no_bench or share > MAX_STALE_SHARE:
            log.warning("Forward test: %s not stored (%.0f%% without a price%s); the next run retries", f"{day:%Y-%m-%d}",
                        share * 100, ", a benchmark among them" if no_bench else "")  # fmt: skip
            return True
    return False


def _backfill(ledger: pd.DataFrame, cohort: pd.DataFrame, fetch: PriceFetch, fx_history: FxHistory) -> pd.DataFrame:
    """Add the stored months of members with no row at all (a list added later). A ticker with no price is left to
    the next run rather than stored flat."""
    missing = sorted(set(cohort["Ticker"]) - set(ledger["Ticker"]))
    if not missing:
        return ledger
    marks = _marks(ledger, cohort["Start"].iloc[0])
    prices, plain, breaks, no_fx = _download_usd(cohort[cohort["Ticker"].isin(missing)], marks[0], fetch, fx_history)
    new = _rows(period_returns(prices, marks), marks, breaks, no_fx, _price_returns(plain, marks))
    new = new[new["Ticker"].isin(missing)]
    priced = new.loc[~new["Stale"].astype(bool), "Ticker"].unique()
    if not len(priced):
        return ledger
    log.info("Forward test: stored months added for %s", ", ".join(priced))
    return extend_ledger(ledger, new[new["Ticker"].isin(priced)])


def update(folder: Path, today: pd.Timestamp, fetch: PriceFetch, fx_history: FxHistory) -> pd.DataFrame:
    """Add every month-end over since the last stored mark (the first time: since `back_start`) to
    `<folder>/ledger.parquet` (nothing when none passed, or when the download failed); returns the ledger. The cohort
    must exist. Members new to the ledger get its stored months first."""
    cohort = load_cohort(folder)
    start = cohort["Start"].iloc[0]
    path = folder / "ledger.parquet"
    ledger = _with_guards(pd.read_parquet(path)) if path.exists() else None
    stored = ledger is not None and len(ledger) > 0
    if stored:
        ledger = _backfill(ledger, cohort, fetch, fx_history)
        ledger.to_parquet(path, index=False)
    last = ledger["Date"].max() if stored else back_start(start)
    due = month_ends(last, today)
    if not due:
        return ledger if ledger is not None else period_returns(pd.DataFrame(), [])
    anchors = _anchors(ledger, start) if stored else {}
    since = min([*anchors.values(), _marks(ledger, start)[-2] if stored else last])
    prices, plain, breaks, no_fx = _download_usd(cohort, since, fetch, fx_history)
    marks = [last, *due]
    new = _rows(period_returns(prices, marks, anchors), marks, breaks, no_fx, _price_returns(plain, marks, anchors))
    if _failed_download(new):
        return ledger if ledger is not None else period_returns(pd.DataFrame(), [])
    owed = reconcile(ledger, prices, start) if stored else {}
    if owed:
        log.warning("Forward test: Yahoo changed the last stored month of %s: carried into %s", ", ".join(owed),
                    f"{due[0]:%Y-%m-%d}")  # fmt: skip
        first = (new["Date"] == due[0]) & new["Ticker"].isin(owed)
        factor = new.loc[first, "Ticker"].map(owed)
        new.loc[first, "Return"] = (1 + new.loc[first, "Return"]) * factor - 1
        new.loc[first, "Check"] = True
    ledger = extend_ledger(ledger, new)
    ledger.to_parquet(path, index=False)
    log.info("Forward test: %d month-end(s) added, last %s", len(due), f"{due[-1]:%Y-%m-%d}")
    return ledger


def step(
    folder: Path,
    published: pd.DataFrame,
    ranked: pd.DataFrame,
    held: frozenset[str] | set[str],
    today: pd.Timestamp,
    fetch: PriceFetch,
    fx_history: FxHistory,
    method: str = "",
    published_on: pd.Timestamp | None = None,
    offline: bool = False,
    grades: pd.DataFrame | None = None,
    late: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The test's work for one benchmark report: (cohort, ledger), None when there is nothing to show (offline
    before any data).

    Without a cohort, it is frozen from `published` (the lists of the report on or just before `START` when the
    caller has them, `published_on` their date), then the month-ends passed are added. `offline` reads what is
    stored and downloads nothing. `grades` (with `Passes`) freezes the grade gate when none is stored; the cohort
    returned is gated (`apply_gate`), the ledger covers every member. `late` (the saved tabs of `LATE_LISTS`) adds
    those lists once."""
    cohort_path = folder / "cohort.parquet"
    if not cohort_path.exists():
        if offline:
            return None
        when = published_on if published_on is not None else today.normalize()
        if abs(when - START) > pd.Timedelta(days=7):
            log.warning("Forward test frozen from lists published %s, not near %s", f"{when:%Y-%m-%d}",
                        f"{START:%Y-%m-%d}")  # fmt: skip
        save_cohort(freeze_cohort(published, ranked, held, START, when, method), cohort_path)
        shutil.rmtree(folder / "preview", ignore_errors=True)  # the former phantom preview
        log.info("Forward test cohort frozen: %s", cohort_path)
    freeze_late_lists(folder, late or {}, ranked, pd.read_parquet(cohort_path)["Start"].iloc[0], method, grades)
    cohort = load_cohort(folder)
    ledger_path = folder / "ledger.parquet"
    if offline:
        ledger = pd.read_parquet(ledger_path) if ledger_path.exists() else period_returns(pd.DataFrame(), [])
    else:
        ledger = update(folder, today, fetch, fx_history)
    gate_path = folder / GATE_FILE
    if not gate_path.exists() and grades is not None and len(grades):
        freeze_gate(grades, gate_path)
        log.info("Forward test grade gate frozen: %s", gate_path)
    if gate_path.exists():
        cohort = apply_gate(cohort, pd.read_parquet(gate_path))
    return cohort, ledger
