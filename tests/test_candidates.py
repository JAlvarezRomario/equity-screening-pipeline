"""Moat / management candidates: reported and tracked, never part of a score."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.evaluate import evaluate_snapshot, snapshot
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking


def test_gross_margin_spread_and_worst_revenue_year(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw[f"Gross Profit_{p}"] = raw[f"Total Revenue_{p}"] * 0.6  # steady 60%
    raw.loc[1, "Gross Profit_Y1"] = raw.loc[1, "Total Revenue_Y1"] * 0.3  # one bad year
    raw.loc[1, "Total Revenue_Y2"] = raw.loc[1, "Total Revenue_Y3"] * 0.8  # revenue fell 20%
    m = calculate_metrics(raw, cfg)
    assert m.loc[0, "Gross_Margin_StdDev"] == pytest.approx(0.0, abs=1e-9)
    assert m.loc[1, "Gross_Margin_StdDev"] > 10
    assert m.loc[0, "Revenue_Worst_Change"] > 0
    assert m.loc[1, "Revenue_Worst_Change"] == pytest.approx(-20.0)


def test_incremental_roic(cfg):
    raw = make_raw(cfg, n=3)
    raw.loc[1, [f"Operating Income_{p}" for p in cfg.annual_years]] = raw.loc[1, "Operating Income_Y4"]  # flat profit
    for p in cfg.annual_years:  # company 2: capital did not grow
        raw.loc[2, f"Stockholders Equity_{p}"] = raw.loc[2, "Stockholders Equity_Y4"]
        raw.loc[2, f"Total Debt_{p}"] = raw.loc[2, "Total Debt_Y4"]
        raw.loc[2, f"Cash And Cash Equivalents_{p}"] = raw.loc[2, "Cash And Cash Equivalents_Y4"]
    m = calculate_metrics(raw, cfg)
    # Profit and capital grow together: the return on new capital equals the return on all (year-end) capital.
    assert m.loc[0, "Incremental_ROIC"] == pytest.approx(m.loc[0, "NOPAT_Y0"] * 100 / m.loc[0, "Invested_Capital_Y0"])
    assert m.loc[1, "Incremental_ROIC"] == pytest.approx(0.0, abs=1e-9)  # more capital, no more profit
    assert pd.isna(m.loc[2, "Incremental_ROIC"])  # nothing reinvested: nothing to judge

    raw.loc[0, "industry"] = "Banks - Regional"
    raw.loc[0, "sector"] = "Financial Services"
    assert pd.isna(calculate_metrics(raw, cfg).loc[0, "Incremental_ROIC"])


def test_sbc_to_revenue(cfg):
    m = calculate_metrics(make_raw(cfg, n=2), cfg)
    assert m["SBC_to_Revenue"].tolist() == pytest.approx([1.0, 1.0])


def test_asset_growth_and_accruals(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw[f"Total Assets_{p}"] = raw[f"Total Revenue_{p}"] * 2
    m = calculate_metrics(raw, cfg)
    growth = (raw["Total Revenue_Y0"] / raw["Total Revenue_Y1"] - 1) * 100
    assert m["Asset_Growth"].tolist() == pytest.approx(growth.tolist())
    # Net income 15% of revenue, operating cash flow 16% (+1 point per company), assets 2x revenue.
    assert m["Accruals"].tolist() == pytest.approx([-0.5, -1.0])

    raw.loc[0, "industry"] = "Banks - Regional"
    raw.loc[0, "sector"] = "Financial Services"
    m = calculate_metrics(raw, cfg)
    assert pd.isna(m.loc[0, "Asset_Growth"]) and pd.isna(m.loc[0, "Accruals"])


def _metrics(cfg, n=30):
    m = calculate_metrics(make_raw(cfg, n=n, seed=3), cfg)
    rng = np.random.default_rng(7)
    m["heldPercentInsiders"] = rng.uniform(0, 0.3, n)
    m["Gross_Margin_StdDev"] = rng.uniform(0, 10, n)
    return m


def test_candidates_never_change_scores_or_ranks(cfg):
    m = _metrics(cfg)
    with_cand = apply_dual_ranking(m, cfg)
    without = apply_dual_ranking(m, replace(cfg, quality_candidates=()))
    cols = ["Ticker", "Value_Score", "Quality_Score", "Rank", "Value_Rank", "Quality_Rank", "Dislocation"]
    pd.testing.assert_frame_equal(with_cand[cols], without[cols])
    assert with_cand["Cand_Pct_heldPercentInsiders"].between(0, 100).all()
    assert not any(c.startswith("Cand_Pct_") for c in without.columns)


def test_candidate_percentiles_match_the_benchmark(cfg):
    m = _metrics(cfg)
    bench = apply_dual_ranking(m, cfg)
    group = apply_dual_ranking(m.iloc[[0, 3, 7]].reset_index(drop=True), cfg, reference=bench, benchmark="sp500")
    ref = bench.set_index("Ticker")
    for _, r in group.iterrows():
        assert r["Cand_Pct_heldPercentInsiders"] == pytest.approx(ref.at[r["Ticker"], "Cand_Pct_heldPercentInsiders"])


def test_snapshot_and_ic_include_candidates(cfg):
    ranked = apply_dual_ranking(_metrics(cfg), cfg)
    snap = snapshot(ranked)
    assert "Cand_Pct_heldPercentInsiders" in snap.columns
    fwd = pd.Series(ranked["Cand_Pct_heldPercentInsiders"].to_numpy(), index=ranked["Ticker"])
    _, ic = evaluate_snapshot(snap, fwd)
    assert ic["cand:heldPercentInsiders"] == pytest.approx(1.0)
    assert "heldPercentInsiders" not in ic  # kept apart from the ranked criteria


def test_worst_revenue_change_counts_a_later_ttm_fall(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "Total Revenue_TTM"] = raw.loc[0, f"Total Revenue_{cfg.latest}"] * 0.8  # -20% since the annual report
    raw.loc[0, "PeriodEnd_TTM"] = (
        pd.Timestamp(raw.loc[0, f"PeriodEnd_{cfg.latest}"]) + pd.DateOffset(months=6)
    ).strftime("%Y-%m-%d")
    m = calculate_metrics(raw, cfg)
    assert m.loc[0, "Revenue_Worst_Change"] == pytest.approx(-20.0)
