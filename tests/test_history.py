"""The instant Quality test on the cache's own history (no network)."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq import history
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking


def test_lag_raw_moves_every_period_back_and_empties_ttm(cfg):
    raw = make_raw(cfg, n=3)
    lagged = history.lag_raw(raw, 1, cfg)
    assert (lagged["Total Revenue_Y0"] == raw["Total Revenue_Y1"]).all()
    assert (lagged["PeriodEnd_Y1"] == raw["PeriodEnd_Y2"]).all()
    assert lagged["Total Revenue_TTM"].isna().all() and lagged["PeriodEnd_TTM"].isna().all()
    assert lagged[f"Total Revenue_Y{cfg.n_years - 1}"].isna().all()  # nothing older to move in
    assert (lagged["marketCap"] == raw["marketCap"]).all()  # info stays


def test_change_is_log_clipped_with_sign_changes():
    out = history.change(pd.Series([1.0, 1.0, -1.0, -1.0, 1.0]), pd.Series([2.0, -1.0, 1.0, -2.0, 100.0]), "log")
    assert out[0] == pytest.approx(np.log(2)) and out[1] == -1.0 and out[2] == 1.0
    assert np.isnan(out[3]) and out[4] == 1.0  # two losses say nothing; clipped at +1


def test_quality_ic_runs_on_the_lagged_cache(cfg):
    raw = make_raw(cfg, n=30)
    now = calculate_metrics(raw, cfg)
    lagged = calculate_metrics(history.lag_raw(raw, 1, cfg), cfg)
    after = history.outcomes(now, lagged, 1, cfg)
    assert list(after.columns) == list(history.OUTCOMES) and after.index.tolist() == raw["Ticker"].tolist()
    ic = history.quality_ic(apply_dual_ranking(lagged, cfg), after, 1, cfg)
    assert {"Quality_Score", "Return_Quality"} <= set(ic["Measure"]) and (ic["Lag"] == 1).all()
    assert ic["IC"].between(-1, 1).all() or ic["IC"].isna().any()


def test_lag_raw_blanks_the_stored_quarters(cfg):
    raw = make_raw(cfg, n=2)
    raw["Pretax Income_Q0"], raw["PeriodEnd_Q0"] = 1.0, "2026-06-30"
    lagged = history.lag_raw(raw, 1, cfg)
    assert lagged["Pretax Income_Q0"].isna().all() and lagged["PeriodEnd_Q0"].isna().all()
