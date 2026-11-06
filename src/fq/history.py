"""Instant test of Quality on the history Yahoo already serves. Pure functions.

Quality claims that today's numbers foresee whether owner earnings last and grow. Waiting for snapshots to age
takes years, but each company's statements already hold the answer for the recent past: `lag_raw` turns the raw
cache into what was known `lag` years ago (Y{lag} becomes Y0, no TTM), Quality is scored on that, and `quality_ic`
checks it against what happened since (`OUTCOMES`, from the true Y{lag} to the true Y0, both audited reports).

Limits: today's companies only (survivors: index members that failed are gone), restated figures, and only
`n_years - lag` annual reports behind each lagged score. It tests the cross-section (did the better scores hold
up better?), which survivorship biases far less than average returns.
"""

import re

import numpy as np
import pandas as pd

from fq.metrics.common import _col, us_terms
from fq.ranking import rankable
from fq.settings import Settings

LAGS = (1,)  # years back: Yahoo serves ~4 annual reports (Y4 is mostly empty), so 1 leaves three behind the score
PERIOD_COLUMN = re.compile(r"^(?P<concept>.+)_(?P<period>TTM|Y\d+|Q\d+)$")
# What happened by today's Y0 (`outcomes`): revenue per share against the year before, and the normalised levels
# Value and Quality relied on a year earlier against what came: the question is whether that earning power
# was still there, not whether one good year repeated (a single year mean-reverts).
OUTCOMES = ("revenue/share", "owner earnings/share vs normalised", "ROIC vs normalised")


def lag_raw(raw: pd.DataFrame, lag: int, cfg: Settings) -> pd.DataFrame:
    """The raw cache as it stood `lag` years ago: every per-period column moved back (Y{k+lag} -> Y{k}, the oldest
    slots empty) and TTM and the stored quarters emptied, so metrics fall back to the lagged Y0. Everything else (price, info) stays as it
    is today: only what Quality reads is meaningful in the result, not Value."""
    out = raw.copy()
    n = cfg.n_years
    moved = {}
    for col in raw.columns:
        m = PERIOD_COLUMN.match(col)
        if m is None:
            continue
        concept, period = m["concept"], m["period"]
        if period == "TTM" or period.startswith("Q"):  # newer than the lagged Y0
            moved[col] = pd.Series(np.nan, index=raw.index, dtype=object if raw[col].dtype == object else float)
            continue
        source = f"{concept}_Y{int(period[1:]) + lag}"
        if int(period[1:]) + lag < n and source in raw.columns:
            moved[col] = raw[source]
        else:
            moved[col] = pd.Series(np.nan, index=raw.index, dtype=object if raw[col].dtype == object else float)
    out[list(moved)] = pd.DataFrame(moved, index=raw.index)
    return out


def _per_share(df: pd.DataFrame, value: pd.Series, period: str) -> pd.Series:
    shares = _col(df, f"Ordinary Shares Number_{period}")
    return value / shares.where(shares > 0)


def change(before: pd.Series, after: pd.Series, how: str) -> pd.Series:
    """In points, or as a log ratio clipped to +/-1 (a profit turned loss = -1, a loss turned profit = +1; two
    losses say nothing)."""
    a, b = pd.Series(pd.to_numeric(before, errors="coerce")), pd.Series(pd.to_numeric(after, errors="coerce"))
    if how == "points":
        return b - a
    ratio = np.log((b / a).where((a > 0) & (b > 0))).clip(-1, 1)
    return ratio.mask((a > 0) & (b <= 0), -1.0).mask((a <= 0) & (b > 0), 1.0)


def outcomes(now: pd.DataFrame, lagged: pd.DataFrame, lag: int, cfg: Settings) -> pd.DataFrame:
    """`OUTCOMES` per company (index = Ticker); `now` and `lagged` are `calculate_metrics` of today's cache and of
    `lag_raw`, in the same row order.

    - revenue/share: log change from Y{lag} to Y0 in US terms (`common.us_terms`, as `Per_Share_Compounding`
      measures it since Nov 6, 2026: today's rate both, net of the currency's inflation beyond US).
    - owner earnings/share vs normalised: today's Y0 owner FCF before financing per share (`FCF_Owner_Y0`) against
      the earnings power Value priced `lag` years earlier (`EPV_Yield` x `Value_Base`: the same flow, the price
      cancelling out, per lagged Y0 share, carried into US terms the same way), log, +/-1 when the sign changed.
    - ROIC vs normalised: today's Y0 ROIC - the lagged `ROIC_Norm`, in points.
    """
    latest = cfg.latest
    actual = _per_share(now, _col(now, f"FCF_Owner_{latest}"), latest)
    excess = (1 + _col(now, f"Excess_Inflation_{latest}").fillna(0.0)).clip(lower=0.01)
    priced = _per_share(lagged, lagged["EPV_Yield"] / 100 * lagged["Value_Base"], latest) * excess.to_numpy() ** lag
    revenue = us_terms(
        pd.DataFrame({p: _col(now, f"Total Revenue_{p}") for p in (f"Y{lag}", latest)}, index=now.index), now, cfg
    )
    frame = pd.DataFrame(
        {
            OUTCOMES[0]: change(
                _per_share(now, revenue[f"Y{lag}"], f"Y{lag}"),
                _per_share(now, revenue[latest], latest),
                "log",
            ),
            OUTCOMES[1]: change(priced.to_numpy(), actual.to_numpy(), "log"),
            OUTCOMES[2]: change(lagged["ROIC_Norm"].to_numpy(), _col(now, f"ROIC_{latest}").to_numpy(), "points"),
        }
    )
    return frame.set_axis(now["Ticker"])


def _t(ic: float, n: int) -> float:
    return ic * np.sqrt((n - 2) / (1 - ic**2)) if n > 2 and abs(ic) < 1 else np.nan


def quality_ic(lagged: pd.DataFrame, after: pd.DataFrame, lag: int, cfg: Settings) -> pd.DataFrame:
    """One row per (measure, outcome): the rank correlation of the lagged Quality score, each Quality criterion's
    percentile and each candidate's with what happened next (`outcomes`), over the ranked companies (`rankable`:
    profitable operating companies, one scale; Nov 6, 2026: every kind apart is scored among its own, so pooling a
    bank's 90th percentile of banks with an industrial's mixed two yardsticks) with a Quality score;
    t = IC x sqrt((N - 2) / (1 - IC²)), one cross-section."""
    ranked = lagged[rankable(lagged) & lagged["Quality_Score"].notna()].set_index("Ticker")
    criteria = {c[0] for c in (*cfg.quality_criteria, *cfg.financial_quality_criteria)}
    measures = [
        "Quality_Score",
        *(f"Pct_{c}" for c in sorted(criteria)),
        *(f"Cand_Pct_{c}" for c, _ in cfg.quality_candidates),
    ]
    rows = []
    for col in (c for c in measures if c in ranked.columns):
        name = "cand:" + col.removeprefix("Cand_Pct_") if col.startswith("Cand_") else col.removeprefix("Pct_")
        for outcome in after.columns:
            pair = pd.concat([ranked[col], after[outcome].reindex(ranked.index)], axis=1).dropna()
            n = len(pair)
            ic = float(pair.rank().corr().iloc[0, 1]) if n >= 10 else np.nan
            rows.append({"Lag": lag, "Measure": name, "Outcome": outcome, "IC": ic, "N": n, "t": _t(ic, n)})
    return pd.DataFrame(rows, columns=["Lag", "Measure", "Outcome", "IC", "N", "t"])
