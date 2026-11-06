"""The dividend record (user's design, Nov 4, 2026, for the retired Unbroken dividends; read by Dividend growers,
Untaxed dividends, the lists by kind and the 20-year tests): dividends per share raised or held
every year over the whole record Yahoo keeps, judged payment by payment (`DPS_Payments`, from `fetch.dividend_payments`;
`DPS_Payments_FX` in the declared currency when the quote differs, so an exchange rate reads as neither a raise nor a cut).

Yearly sums alone cannot tell a special dividend from a raise later cut back, nor a payment that slipped past New Year
from a cut; the payments can. The rules, in order:
1. Usual count of a year: the median number of payments over the five years around it (a switch from annual to
   semi-annual payments follows).
2. Extraordinary payment: at least `SPECIAL_RATIO` x the median of the same-season payments in the two years either
   side (at least two of them), and above both the same-season payment a year before and the one a year after (a
   spike, not a raise that stays). An extra payment (its year has more than usual) is dropped (Costco's specials); one
   in the regular slot is capped at that median (a special folded into an annual payment).
3. Regular dividend of a complete calendar year: the sum of its payments after step 2.
4. Cut: a year under `HELD` x the year before, unless the payment counts explain it: the year has fewer payments than
   the one before and the next year makes up the pair, or the year before had more payments than usual or than the
   year before it (one pulled forward, or an interim added) and the year is held against the year before that. The
   tolerance is for rounding, not for cuts; a raise cut back with the same payments is a cut.
5. Unbroken years: back from the last complete year to the latest gap or cut; the record is whole when that reaches
   the first year Yahoo has a payment, or when the latest gap or cut is more than `JUDGED_YEARS` old (user's choice,
   Nov 5, 2026: old Yahoo records hold lone wrong payments, ED and GD in the 1970s-80s at a quarter of the usual one,
   and a business that has held its dividend for 30 years has shown what the list asks).
6. This year: the regular dividends of the 12 months to the fetch under `HELD` x the 12 months before, both windows
   holding the same number of payments (a payment late, early or not yet made is not judged), is a cut now
   (`Dividend_Cut_Recent`): the record is broken at once, not next year.
7. Through recessions (`lists/recessions.txt`, user's design, Nov 4, 2026): for each global recession Y the record
   covers (payments from Y-2, Y+2 complete, the business model not exempt), and each downturn of one sector for the
   companies of that sector (the 2015 oil price collapse for Energy, user's choice, Nov 5, 2026: Yahoo's few annual
   reports hold the 2022 spike, the dividend reaches the crash), the lower of the regular dividend's mean
   over (Y, Y+1) and (Y+1, Y+2) against its mean over (Y-2, Y-1), capped at 0: pairs of years, so a payment slipping
   across a year end reads as no fall. `Recession_Dividend_Fall` is the worst, `Recession_Dividend_Note` says each.
   The dividend is the one long window Yahoo gives into a recession (share counts mostly start after 2014); it exists
   only for payers, so it is never scored and never in the `Cyclical` flag, only shown and tested by lists, where a
   blank passes.

Display only; constants here, as the screens', so the method fingerprint stays as it is. No network."""

import json
from functools import cache

import numpy as np
import pandas as pd

from fq.universe import _read_lines

SPECIAL_RATIO = 1.5  # a payment this many times its season's usual size, and a spike, is extraordinary
HELD = 0.95  # a regular dividend under this share of the year before is a cut (rounding and residual FX pass)
SEASON_DAYS = 75  # a payment within this many days of a date one or two years away is the same season
GROWTH_YEARS = 10
JUDGED_YEARS = 30  # a gap or cut older than this many complete years no longer breaks the record (rule 5)
WINDOW_DAYS = 365  # the 12-month windows of rule 6
RECESSIONS = "recessions.txt"
COLUMNS = {"Dividend_Unbroken_Years": float, "Dividend_Record_Whole": bool, "Regular_DPS_Growth_10Y": float,
           "Dividend_Cut_Recent": bool, "Recession_Dividend_Fall": float, "Recession_Dividend_Note": object,
           "Recessions_Covered": float, "Recessions_Due": float}  # fmt: skip


@cache
def recessions() -> tuple[tuple[int, frozenset[str], frozenset[str]], ...]:
    """(year, business models it does not judge, the only Yahoo sectors it judges: empty for a global recession) for
    each line of `RECESSIONS`: `<year>`, then `not: <model>, ...` or `only: sector <sector>, ...`."""
    out = []
    for line in _read_lines(RECESSIONS):
        year, _, rest = line.split("#")[0].strip().partition(" ")
        rest = rest.strip()
        exempt, only = frozenset(), frozenset()
        if rest.startswith("only:"):
            names = rest.removeprefix("only:").strip().removeprefix("sector").strip()
            only = frozenset(s.strip() for s in names.split(",") if s.strip())
        else:
            exempt = frozenset(m.strip() for m in rest.removeprefix("not:").split(",") if m.strip())
        out.append((int(year), exempt, only))
    return tuple(out)


def applies(exempt: frozenset[str], only: frozenset[str], model, sector) -> bool:
    """Whether a downturn of `recessions()` judges a company of `model` in `sector`."""
    if isinstance(model, str) and model in exempt:
        return False
    return not only or (isinstance(sector, str) and sector in only)


def parse_payments(value) -> pd.Series:
    """A stored `{"YYYY-MM-DD": amount}` JSON as a float Series on a date index, oldest first; empty when absent."""
    if not isinstance(value, str) or not value:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    try:
        data = json.loads(value)
    except ValueError:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in data.items() if v is not None}, dtype=float).sort_index()
    return s[s > 0]


AROUND, BEFORE, AFTER = (-730, -365, 365, 730), (-365,), (365,)


def _season(days: np.ndarray, i: int, offsets: tuple[int, ...]) -> np.ndarray:
    """Indices of the payments within `SEASON_DAYS` of payment `i` moved by each of `offsets` (days)."""
    gap = days - days[i]
    return np.flatnonzero(np.any([np.abs(gap - o) <= SEASON_DAYS for o in offsets], axis=0))


def regular_payments(payments: pd.Series) -> pd.Series:
    """`payments` with the extraordinary ones dropped or capped (rules 1 and 2 of the module docstring)."""
    if len(payments) < 3:
        return payments
    values = payments.to_numpy(dtype=float)
    days = ((payments.index - payments.index[0]).days).to_numpy()
    years = payments.index.year.to_numpy()
    counts = pd.Series(years).value_counts()
    span = range(years.min(), years.max() + 1)
    count_of = {y: int(counts.get(y, 0)) for y in span}
    usual = {y: np.median([count_of[x] for x in range(y - 2, y + 3) if x in count_of]) for y in span}
    out = values.copy()
    drop = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        season = _season(days, i, AROUND)
        if len(season) < 2:
            continue
        median = float(np.median(values[season]))
        before, after = _season(days, i, BEFORE), _season(days, i, AFTER)
        if len(before) == 0 or len(after) == 0 or values[i] < SPECIAL_RATIO * median:
            continue
        if values[i] <= values[before].max() or values[i] <= values[after].max():
            continue
        if count_of[years[i]] > usual[years[i]]:
            drop[i] = True
        else:
            out[i] = median
    return pd.Series(out[~drop], index=payments.index[~drop])


def _cut_recent(regular: pd.Series, as_of: pd.Timestamp | None) -> bool:
    """Rule 6: the 12 months to `as_of` under `HELD` x the 12 months before, both with the same number of payments."""
    if as_of is None or pd.isna(as_of):
        return False
    window = pd.Timedelta(days=WINDOW_DAYS)
    now = regular[(regular.index > as_of - window) & (regular.index <= as_of)]
    before = regular[(regular.index > as_of - 2 * window) & (regular.index <= as_of - window)]
    if now.empty or len(now) != len(before):
        return False
    return bool(now.sum() < HELD * before.sum())


def _recession_falls(total: dict[int, float], first: int, last_year: int, model, sector=None) -> tuple[list, int]:
    """Rule 7: [(year, fall %)] for each downturn the record covers, and how many apply to `model` in `sector`."""
    due = [y for y, exempt, only in recessions() if applies(exempt, only, model, sector)]
    falls = []
    for y in due:
        if first > y - 2 or y + 2 > last_year:
            continue
        base = (total[y - 2] + total[y - 1]) / 2
        if base <= 0:
            continue
        low = min(total[y] + total[y + 1], total[y + 1] + total[y + 2]) / 2
        falls.append((y, min(0.0, (low / base - 1) * 100)))
    return falls, len(due)


def _note(falls: list) -> str | None:
    """ "2009 held; 2020 −40%": a fall within the cut tolerance is held."""
    if not falls:
        return None
    tolerance = -(1 - HELD) * 100
    return "; ".join(f"{y} held" if f >= tolerance else f"{y} −{abs(f):.0f}%" for y, f in falls)


EMPTY = {"Dividend_Unbroken_Years": np.nan, "Dividend_Record_Whole": False, "Regular_DPS_Growth_10Y": np.nan,
         "Dividend_Cut_Recent": False, "Recession_Dividend_Fall": np.nan, "Recession_Dividend_Note": None,
         "Recessions_Covered": 0.0, "Recessions_Due": np.nan}  # fmt: skip


def judge(payments: pd.Series, last_year: int, as_of: pd.Timestamp | None = None, model=None, sector=None) -> dict:
    """The record of one company (rules 1-7 of the module docstring), complete years up to `last_year`, this year's
    payments up to `as_of`, downturns as they apply to the business `model` in `sector`: the `COLUMNS`. Without payments the
    record is blank (`Dividend_Unbroken_Years` NaN), and each list's test says whether a blank passes."""
    if payments[payments.index.year <= last_year].empty:
        return dict(EMPTY)
    cut_now = _cut_recent(regular_payments(payments), as_of)
    regular = regular_payments(payments[payments.index.year <= last_year])
    first = int(payments.index.year.min())
    years = range(first, last_year + 1)
    by_year = regular.groupby(regular.index.year)
    total = {y: float(by_year.sum().get(y, 0.0)) for y in years}
    n = {y: int(by_year.size().get(y, 0)) for y in years}
    usual = {y: np.median([n[x] for x in range(y - 2, y + 3) if x in n]) for y in years}

    def broken(y: int) -> bool:
        if total[y] <= 0:
            return True
        if y == first or total[y] >= HELD * total[y - 1]:
            return False
        slipped_out = n[y] < n[y - 1] and y + 1 in total and total[y] + total[y + 1] >= HELD * 2 * total[y - 1]
        # More payments the year before than usual or than the year before it: one pulled forward, or a move to more
        # frequent payments (an interim added) that front-loaded that year (Dom Development, 2023).
        extra = y - 2 in total and (n[y - 1] > usual[y - 1] or n[y - 1] > n[y - 2])
        pulled_in = extra and total[y] >= HELD * total[y - 2]
        return not (slipped_out or pulled_in)

    start = next((y for y in range(last_year, first - 1, -1) if broken(y)), None)
    unbroken = last_year - first + 1 if start is None else last_year - start  # the broken year itself not counted
    then = total.get(last_year - GROWTH_YEARS, 0.0)
    growth = (
        ((total[last_year] / then) ** (1 / GROWTH_YEARS) - 1) * 100 if then > 0 and total[last_year] > 0 else np.nan
    )
    falls, due = _recession_falls(total, first, last_year, model, sector)
    return {"Dividend_Unbroken_Years": 0.0 if cut_now else float(unbroken),
            "Dividend_Record_Whole": (start is None or start <= last_year - JUDGED_YEARS) and not cut_now,
            "Regular_DPS_Growth_10Y": growth,
            "Dividend_Cut_Recent": cut_now,
            "Recession_Dividend_Fall": min(f for _, f in falls) if falls else np.nan,
            "Recession_Dividend_Note": _note(falls), "Recessions_Covered": float(len(falls)),
            "Recessions_Due": float(due)}  # fmt: skip


def dividend_record(df: pd.DataFrame) -> pd.DataFrame:
    """`judge` for every row of `df`: payments from `DPS_Payments_FX`, else `DPS_Payments`; the last complete year is the
    one before `Fetched_At`, this year's payments judged up to it; downturns by `Business_Model` and `sector`. The
    `COLUMNS`."""
    empty = pd.Series([None] * len(df), index=df.index, dtype=object)
    fx = df.get("DPS_Payments_FX", empty)
    raw = df.get("DPS_Payments", empty)
    chosen = fx.where(fx.map(lambda v: isinstance(v, str) and bool(v)), raw)
    fetched = pd.to_datetime(df.get("Fetched_At", empty), errors="coerce")
    models, sectors = df.get("Business_Model", empty), df.get("sector", empty)
    rows = [
        judge(parse_payments(p), when.year - 1, when, m, sec) if pd.notna(when) else dict(EMPTY)
        for p, when, m, sec in zip(chosen, fetched, models, sectors, strict=True)
    ]
    return pd.DataFrame(rows, index=df.index, columns=list(COLUMNS)).astype(COLUMNS)  # columns even when `df` is empty
