"""Normal levels from a short history: growth profiles, normalised ratios and yields."""

import numpy as np
import pandas as pd

from fq.metrics.common import _col, _distinct, _distinct_mean, _num, _row_years, _slot, _text
from fq.settings import Settings

# FCF margin (a fraction): a step under half a point is flat, so a run of rises or falls needs every step above it
# (user's choice, Oct 21, 2026: SJM's margin read "rose every year" on a 0.09-point first step and earning power
# jumped from the median year to the latest).
MARGIN_STEP = 0.005
# A steady improver may hold flat for a year (`_pick_level`): a step down to this share of the year before is flat.
STEADY_FLAT = 0.95


def growth_profile(row: pd.Series, concept: str, years: list[str], step: float = 0.0) -> dict:
    """Classify a metric's history (oldest to newest, TTM last). A rise or fall counts only when it exceeds `step`."""
    annual = sorted((y for y in years if y != "TTM"), key=_slot, reverse=True)  # oldest first
    order = annual + (["TTM"] if "TTM" in years else [])
    kept = [(y, v) for y in order if pd.notna(v := _num(row, f"{concept}_{y}"))]
    vals, used = [v for _, v in kept], [y for y, _ in kept]

    if len(vals) < 3:
        return {"status": "INSUFFICIENT_DATA", "vals": vals, "years": used}
    rises = [b - a > step for a, b in zip(vals, vals[1:], strict=False)]
    rising = all(rises)
    falling = all(a - b > step for a, b in zip(vals, vals[1:], strict=False))
    mean = np.mean(vals)
    cv = np.std(vals) / abs(mean) if mean != 0 else 0.0
    if rising and len(vals) >= 4:
        return {"status": "STRICT_GROWTH", "vals": vals, "years": used}
    if falling:
        return {"status": "DECLINE", "vals": vals, "years": used}
    # Three rising points are treated like a dip followed by two rises (not worse): stable -> recent level.
    if all(rises[-2:]) and cv < 0.20:
        return {"status": "RECENT_GROWTH_STABLE", "vals": vals, "years": used}
    if rising:
        return {"status": "RECENT_GROWTH", "vals": vals, "years": used}
    return {"status": "CYCLICAL", "vals": vals, "years": used, "cv": cv}


def _median_positions(vals: list[float]) -> list[int]:
    """The positions the median is taken from (the middle one, or the middle two)."""
    order = list(np.argsort(vals, kind="stable"))
    n = len(order)
    return [int(order[n // 2])] if n % 2 else [int(order[n // 2 - 1]), int(order[n // 2])]


def _pick_level(
    vals: list[float],
    status: str,
    cyclical: bool = False,
    premium: float = 1.0,
    is_fraction: bool = False,
    overearner_ratio: float = 1.5,
) -> tuple[float, list[int], bool]:
    """(level, positions of `vals` it stands for, capped as an over-earner): see `_normal_level`. The positions date
    the level (`level_age`), so a trend projected from it starts where the level sits, not at the latest year."""
    if not vals:
        return np.nan, [], False
    every = list(range(len(vals)))
    if cyclical:
        median, mean = float(np.median(vals)), float(np.mean(vals))
        return (median, _median_positions(vals), False) if median <= mean else (mean, every, False)

    # Over-earner check: latest > 1.5x anchor (30th percentile). A series that rose every year without a single step of
    # that size is a steady improver, not a peak (FICO's ROIC 41 -> 69, steps under 1.25x, read 48 and lost its trend
    # with it); a climb with any such step is a recovery from a trough and stays capped (FLXS 3.6 -> 7 -> 14.5 -> 21.4),
    # as is a climb out of losses. Since Nov 6, 2026 a steady improver may also hold flat a year (`STEADY_FLAT`) when
    # its last two steps rose (`RECENT_GROWTH_STABLE`): 30, 30, 38, 44, 50 was capped to 38 for the flat year alone.
    floor = 0.02 if is_fraction else 2.0
    anchor = max(float(np.percentile(vals, 30)), floor)
    steady_climb = (
        status in ("STRICT_GROWTH", "RECENT_GROWTH_STABLE")
        and vals[0] > 0
        and vals[-1] > vals[0]
        and all(STEADY_FLAT * a <= b <= overearner_ratio * a for a, b in zip(vals, vals[1:], strict=False))
    )
    if vals[-1] > overearner_ratio * anchor and (vals[-1] - anchor) >= floor and not steady_climb:
        return float(np.median(vals)), _median_positions(vals), True

    last = [len(vals) - 1]
    if status == "STRICT_GROWTH":
        return vals[-1] * premium, last, False
    if status in ("RECENT_GROWTH_STABLE", "DECLINE"):
        return vals[-1], last, False
    if status == "RECENT_GROWTH":
        return float(np.mean(vals[-3:])), every[-3:], False
    if status == "INSUFFICIENT_DATA":
        return float(np.mean(vals)), every, False
    return float(np.median(vals)), _median_positions(vals), False


def _normal_level(
    vals: list[float],
    status: str,
    cyclical: bool = False,
    premium: float = 1.0,
    is_fraction: bool = False,
    overearner_ratio: float = 1.5,
) -> float:
    """Robust normal level of a series (oldest to newest): mid-cycle for cyclicals,
    median-capped for over-earners (boom > 1.5x 30th percentile), otherwise growth profile.
    """
    return _pick_level(vals, status, cyclical, premium, is_fraction, overearner_ratio)[0]


def level_age(row: pd.Series, years: list[str], positions: list[int]) -> float:
    """Years between the newest of `years` and the mean date of the periods at `positions` (0 when the level is the
    latest value, about 1.5 for the median of four years). Periods without a date take Y0's date less their slot."""
    if not years or not positions:
        return 0.0
    y0 = pd.to_datetime(row.get("PeriodEnd_Y0"), errors="coerce")

    def when(y: str) -> pd.Timestamp:
        d = pd.to_datetime(row.get(f"PeriodEnd_{y}"), errors="coerce")
        if pd.isna(d) and y != "TTM" and pd.notna(y0):
            d = y0 - pd.Timedelta(days=365.25 * _slot(y))
        return d

    dates = [when(y) for y in years]
    known = [d for d in dates if pd.notna(d)]
    picked = [dates[i] for i in positions if pd.notna(dates[i])]
    if not known or not picked:
        return 0.0
    newest = max(known)
    return float(np.mean([(newest - d).days / 365.25 for d in picked]))


def normalized_metric(
    df: pd.DataFrame,
    name: str,
    years: list[str],
    cyclical: pd.Series | None = None,
    with_basis: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Robust 'normal' level of a ratio such as ROIC, chosen by growth profile,
    cyclicality and over-earner capping. `with_basis` also returns, per company, the level's `Age` in years
    (`level_age`), whether it was `Capped` as an over-earner and whether the series fell every year (`Declining`).
    """
    cyclical = cyclical if cyclical is not None else pd.Series(False, index=df.index)
    latest = min((y for y in years if y != "TTM"), key=_slot)
    out, basis = [], []
    for idx, row in df.iterrows():
        prof = growth_profile(row, name, _row_years(row, name, years, latest))
        vals, status = prof["vals"], prof["status"]
        is_cyc = bool(cyclical.get(idx, False))
        norm, positions, capped = _pick_level(vals, status, cyclical=is_cyc, premium=1.0, is_fraction=False)
        out.append(norm)
        basis.append((level_age(row, prof["years"], positions), capped, status == "DECLINE"))
    level = pd.Series(out, index=df.index)
    if not with_basis:
        return level
    return level, pd.DataFrame(basis, index=df.index, columns=["Age", "Capped", "Declining"])


def normalized_fcf_yield(
    df: pd.DataFrame,
    years: list[str],
    premium: float,
    cyclical: pd.Series | None = None,
    concept: str = "FCF_Adj",
    denominator: str = "FCF_Base",
    cap: pd.Series | None = None,
    with_basis: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Normalised FCF yield per share = smart FCF margin x smart revenue base / `denominator` (`FCF_Base` or
    `Value_Base`, `_shareholder_base`).

    Margin: latest (x premium) if margins expand every year, latest if recently
    expanding and stable or falling every year, otherwise the median. Revenue base: latest if growing
    or falling every year, otherwise the 3-year average, both judged at today's exchange rate like the price it is divided by
    (historical rates would count a fallen currency's old, larger USD revenue as if it would come
    back, and would call a high-inflation company's local growth flat). FCF is `concept` (`FCF_Adj`: after
    SBC; net income in financial mode).
    `cyclical` companies get a mid-cycle view instead: the lower of the median and mean margin of all
    periods, and the 3-year average revenue, since a run of rising years may be the upswing of a cycle.
    `cap` (a yield on the same denominator; `valuation._value` passes the TTM after-tax profit where
    `earnings_collapse`) bounds the result: earning power is at most what the TTM earned.
    `with_basis` also returns the level's `Age` (years: the mean of the margin's and the revenue base's, `level_age`;
    0 where the TTM cap binds) and whether the margin was `Capped` as an over-earner.
    """
    cyclical = cyclical if cyclical is not None else pd.Series(False, index=df.index)
    latest = min((y for y in years if y != "TTM"), key=_slot)
    out, basis = [], []
    for idx, row in df.iterrows():
        margins = {}
        for y in _row_years(row, concept, years, latest):
            fcf, rev = _num(row, f"{concept}_{y}"), _num(row, f"Total Revenue_{y}")
            if pd.notna(fcf) and pd.notna(rev) and rev > 0:
                margins[f"FCF_Margin_{y}"] = fcf / rev
        if not margins:
            out.append(np.nan)
            basis.append((0.0, False))
            continue

        mp = growth_profile(pd.Series(margins), "FCF_Margin", years, step=MARGIN_STEP)
        mid_cycle = bool(cyclical.get(idx, False))
        margin, m_pos, capped = _pick_level(
            mp["vals"], mp["status"], cyclical=mid_cycle, premium=premium, is_fraction=True
        )

        rp = growth_profile(row, "Total Revenue", _row_years(row, "Total Revenue", years, latest))  # today's rate
        rv = rp["vals"]
        r_pos = list(range(len(rv)))
        if not rv:
            base = np.nan
        elif rp["status"] in ("STRICT_GROWTH", "RECENT_GROWTH_STABLE") and not mid_cycle:
            base, r_pos = rv[-1], r_pos[-1:]
        elif rp["status"] == "DECLINE":  # falling every year: the older, larger revenue is not coming back
            base, r_pos = rv[-1], r_pos[-1:]
        elif rp["status"] == "INSUFFICIENT_DATA":
            base = np.mean(rv)
        else:
            base, r_pos = np.mean(rv[-3:]), r_pos[-3:]
        age = (level_age(row, mp["years"], m_pos) + level_age(row, rp["years"], r_pos)) / 2

        value = _num(row, denominator)
        ok = pd.notna(margin) and pd.notna(base) and pd.notna(value) and value > 0
        normal = margin * base / value * 100 if ok else np.nan
        limit = cap.get(idx, np.nan) if cap is not None else np.nan
        if ok and pd.notna(limit) and limit < normal:
            normal, age = limit, 0.0  # the cut is not assumed to reverse
        out.append(normal)
        basis.append((age, capped))
    level = pd.Series(out, index=df.index)
    if not with_basis:
        return level
    return level, pd.DataFrame(basis, index=df.index, columns=["Age", "Capped"])


QUARTER_GAP_DAYS = (60, 120)  # Q0 and Q1 one quarter apart (a half-yearly reporter's "quarters" are not)


def _quarter_yield(df: pd.DataFrame, q: str) -> pd.Series:
    """One quarter's clean pre-tax profit before interest paid, annualised, as % of `Equity_Value` (the basis of
    `Pretax_PreFin_Yield_<p>`; interest received is not stored by quarter, so it stays in). Financial and float
    mode keep their interest: it is the business."""
    operating = ~(df["Financial_Mode"] | (df["Business_Model"] == "float"))
    pretax = _col(df, f"Pretax Income_{q}") - _col(df, f"Total Unusual Items_{q}").fillna(0)
    paid = _col(df, f"Interest Expense_{q}").abs().fillna(0).where(operating, 0.0)
    return (pretax + paid) * 4 * 100 / df["Equity_Value"]


def _half_year(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(Q0, Q1) annualised clean pre-tax yields, NaN unless both exist and are one quarter apart."""
    gap = (pd.to_datetime(_text(df, "PeriodEnd_Q0"), errors="coerce")
           - pd.to_datetime(_text(df, "PeriodEnd_Q1"), errors="coerce")).dt.days  # fmt: skip
    ok = gap.between(*QUARTER_GAP_DAYS)
    return _quarter_yield(df, "Q0").where(ok), _quarter_yield(df, "Q1").where(ok)


def earnings_collapse(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """True where profit has fallen and stayed down: clean profit before financing over the TTM more than
    `earnings_collapse_pct` below the latest annual report (a profit then, the TTM newer), AND each of the two latest
    quarters below that annual pace by as much. The cut is then not assumed to reverse (user's choice, Oct 20, 2026;
    extends "falling every year = latest"). One bad quarter is not a collapse: TEF.MC's Q4-25 held -EUR 2.9bn of
    charges Yahoo mostly did not tag as unusual, with Q1-Q3 normal. Without stored quarters the TTM test decides."""
    ttm, latest = _col(df, "Pretax_PreFin_Yield_TTM"), _col(df, f"Pretax_PreFin_Yield_{cfg.latest}")
    newer = pd.to_datetime(_text(df, "PeriodEnd_TTM"), errors="coerce") > pd.to_datetime(
        _text(df, f"PeriodEnd_{cfg.latest}"), errors="coerce"
    )
    bar = latest * (1 - cfg.earnings_collapse_pct / 100)
    q0, q1 = _half_year(df)
    lasting = ((q0 < bar) & (q1 < bar)) | q0.isna() | q1.isna()
    return (latest > 0) & (ttm < bar) & newer & lasting


def run_rate(df: pd.DataFrame, cfg: Settings, after_tax: bool) -> pd.Series:
    """The half-year run-rate where profit collapsed (`earnings_collapse`): (Q0 + Q1) x 2 as a yield before
    financing on `Equity_Value`, after the TTM tax rate when `after_tax` (user's choice: recent, and less seasonal
    than one quarter). QFIN: ~$386m against a TTM of $561m. NaN elsewhere or without both quarters."""
    q0, q1 = _half_year(df)
    rate = (q0 + q1) / 2
    if after_tax:
        rate = rate * (1 - _col(df, "Tax_Rate_TTM").fillna(cfg.notional_tax_rate))
    return rate.where(earnings_collapse(df, cfg))


def _pretax_power(
    df: pd.DataFrame, cfg: Settings, name: str = "Pretax_PreFin_Yield", gains_inside: pd.Series | None = None
) -> pd.Series:
    """Pre-tax yield before financing (% of `Equity_Value`; `name` = "Aftertax_PreFin_Yield" for the same after
    tax): the mean of TTM and the last three years (quarters shared by TTM and Y0 counted once), or the latest
    when it fell every year, as `normalized_metric` does: the older, larger profit is not coming back by default.
    `gains_inside` rows (insurers, holdings: investment gains in the profit) take the lower of that mean and the
    median, so one good market year weighs less."""
    mean = _distinct_mean(df, name, cfg.avg_years, cfg)
    if gains_inside is not None:
        median = _distinct(df, name, cfg.avg_years, cfg).median(axis=1)
        mean = mean.where(~gains_inside, np.fmin(mean, median))
    latest = _col(df, f"{name}_TTM").fillna(_col(df, f"{name}_{cfg.latest}"))
    declining = pd.Series(
        [growth_profile(row, name, _row_years(row, name, cfg.avg_years, cfg.latest))["status"] == "DECLINE"
         for _, row in df.iterrows()],
        index=df.index,
    )  # fmt: skip
    collapsed = earnings_collapse(df, cfg)
    recent = np.fmin(latest, run_rate(df, cfg, after_tax=name.startswith("Aftertax")))  # fmin skips a missing side
    return mean.where(~(declining | collapsed), latest.where(~collapsed, recent))
