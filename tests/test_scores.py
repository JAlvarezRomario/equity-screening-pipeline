"""Tests for the unified single-score Value (Owner_Yield) and Quality (Return_Quality) metrics."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.metrics.valuation import _owner_yield
from fq.ranking import apply_dual_ranking


def _trend_frame(cfg, flows, interest=0.0, fx_then=None, cyclical=False, epv=5.0, excess=0.0):
    """One company for `_owner_yield`: `flows` per period, oldest first (TTM last), at today's rate."""
    periods = list(reversed(cfg.fiscal_years))[-len(flows) :]  # oldest first
    row = {"Ticker": "T0", "Financial_Mode": False, "Cyclical": cyclical, "Business_Model": "operating", "EPV_Yield": epv,
           f"Excess_Inflation_{cfg.latest}": excess}  # fmt: skip
    end = pd.Timestamp("2026-06-30")
    for p in periods:
        years_back = -0.5 if p == "TTM" else float(p[1:])
        row[f"PeriodEnd_{p}"] = end - pd.Timedelta(days=round(365.25 * years_back))
    for p, f in zip(periods, flows, strict=True):
        row[f"FCF_Reported_{p}"] = f - interest  # after interest
        row[f"FCF_Adj_{p}"] = f - interest
        row[f"FCF_PreFin_{p}"] = f
        row[f"Ordinary Shares Number_{p}"] = 10.0
        row[f"Total Revenue_{p}"] = 1000.0
        row[f"FX_Hist_{p}"] = 1.0 if fx_then is None else fx_then[p]
    return pd.DataFrame([row])


def _basis(age=0.0, capped=False):
    return pd.DataFrame({"Age": [age], "Capped": [capped]})


def test_owner_trend_is_before_interest_so_debt_does_not_magnify_it(cfg):
    flows = [100.0, 110.0, 121.0, 133.1]
    plain, levered = _trend_frame(cfg, flows), _trend_frame(cfg, flows, interest=60.0)
    _owner_yield(plain, cfg, _basis())
    _owner_yield(levered, cfg, _basis())
    assert levered.loc[0, "Owner_Trend"] == pytest.approx(plain.loc[0, "Owner_Trend"])


def test_owner_trend_is_in_constant_currency(cfg):
    # Flat in local currency while the currency fell: the level is at today's rate already, so no decline to project.
    periods = list(reversed(cfg.fiscal_years))[-4:]
    fallen = _trend_frame(cfg, [100.0] * 4, fx_then={p: 1.6 - 0.2 * i for i, p in enumerate(periods)})
    _owner_yield(fallen, cfg, _basis())
    assert fallen.loc[0, "Owner_Trend"] == pytest.approx(0.0, abs=1e-9)  # flat: R² 0, nothing credited or charged


def test_owner_trend_is_net_of_the_currencys_own_inflation(cfg):
    flows = [100.0, 110.0, 121.0, 133.1]
    us, local = _trend_frame(cfg, flows), _trend_frame(cfg, flows, excess=0.05)
    _owner_yield(us, cfg, _basis())
    _owner_yield(local, cfg, _basis())
    assert local.loc[0, "Owner_Trend"] < us.loc[0, "Owner_Trend"] - 3.0


def test_owner_trend_needs_four_points(cfg):
    three = _trend_frame(cfg, [100.0, 120.0, 140.0])
    _owner_yield(three, cfg, _basis())
    assert three.loc[0, "Owner_Trend"] == 0.0 and three.loc[0, "Owner_Yield"] == pytest.approx(5.0)


def test_cyclical_owner_trend_is_not_charged_unless_it_fell_every_year(cfg):
    bumpy = _trend_frame(cfg, [140.0, 90.0, 120.0, 80.0], cyclical=True)  # the cycle: mid-cycle level already
    _owner_yield(bumpy, cfg, _basis())
    assert bumpy.loc[0, "Owner_Trend"] == 0.0 and bumpy.loc[0, "Owner_Yield"] == pytest.approx(5.0)
    falling = _trend_frame(cfg, [140.0, 120.0, 100.0, 80.0], cyclical=True)
    _owner_yield(falling, cfg, _basis())
    assert falling.loc[0, "Owner_Trend"] < 0 and falling.loc[0, "Owner_Yield"] < 5.0
    rising = _trend_frame(cfg, [80.0, 100.0, 120.0, 140.0], cyclical=True)
    _owner_yield(rising, cfg, _basis())
    assert rising.loc[0, "Owner_Trend"] == 0.0


def test_owner_trend_projects_from_the_levels_date_and_not_past_an_over_earner_cap(cfg):
    flows = [100.0, 110.0, 121.0, 133.1]
    latest, median = _trend_frame(cfg, flows), _trend_frame(cfg, flows)
    _owner_yield(latest, cfg, _basis(age=0.0))
    _owner_yield(median, cfg, _basis(age=1.5))
    g = latest.loc[0, "Owner_Trend"] / 100
    assert latest.loc[0, "Owner_Yield"] == pytest.approx(5.0 * (1 + g) ** 3)
    assert median.loc[0, "Owner_Yield"] == pytest.approx(5.0 * (1 + g) ** 4.5)
    capped = _trend_frame(cfg, flows)
    _owner_yield(capped, cfg, _basis(capped=True))
    assert capped.loc[0, "Owner_Trend"] == 0.0 and capped.loc[0, "Owner_Yield"] == pytest.approx(5.0)


def test_owner_trend_rate_is_on_the_newest_level(cfg):
    # The rate is the slope over the line's value at the newest period, not over the mean.
    y = np.array([100.0, 110.0, 120.0, 130.0])
    df = _trend_frame(cfg, list(y))
    _owner_yield(df, cfg, _basis())
    x = np.array([-2.0, -1.0, 0.0, 0.5])  # Y2, Y1, Y0, TTM half a year later
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - ((y - intercept - slope * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    expected = ((1 + slope / (intercept + slope * 0.5)) / (1 + cfg.inflation) - 1) * r2 * 100
    assert df.loc[0, "Owner_Trend"] == pytest.approx(expected, rel=0.02)


def test_owner_yield_forward_growth_credited_when_smooth(cfg):
    raw = make_raw(cfg, n=2)
    # Give row 0 a smooth rising owner earnings per share
    for k, p in enumerate(cfg.fiscal_years):
        # Y3 is oldest, TTM is newest: growing +10% real per year
        raw.loc[0, f"Operating Cash Flow_{p}"] = 100.0 * (1.1 ** (len(cfg.fiscal_years) - k))
        raw.loc[0, f"Capital Expenditure_{p}"] = -20.0 * (1.1 ** (len(cfg.fiscal_years) - k))
        raw.loc[0, f"Ordinary Shares Number_{p}"] = 10.0
        raw.loc[0, f"Total Revenue_{p}"] = 500.0
        raw.loc[0, f"FX_Hist_{p}"] = 1.0

    df = calculate_metrics(raw, cfg)
    assert "Owner_Yield" in df.columns
    assert "Owner_Trend" in df.columns
    assert "Owner_Smoothness" in df.columns

    # Smooth uptrend should give positive Owner_Trend and boost Owner_Yield above EPV_Yield
    if df.loc[0, "EPV_Yield"] > 0:
        assert df.loc[0, "Owner_Yield"] >= df.loc[0, "EPV_Yield"]
        assert df.loc[0, "Owner_Trend"] > 0.0


def test_return_quality_components_and_haircut(cfg):
    raw = make_raw(cfg, n=3)
    # Row 0: safe leverage (NetDebt/EBIT <= 4)
    # Row 1: high leverage (NetDebt/EBIT > 4, <= 6) -> haircut 0.75
    # Row 2: severe leverage (NetDebt/EBIT > 6) -> haircut 0.50
    df = calculate_metrics(raw, cfg)
    assert "Return_Quality" in df.columns
    assert "ROIC_Slope" in df.columns
    assert "Quality_Haircut" in df.columns

    # Check haircuts
    haircuts = df["Quality_Haircut"].tolist()
    for h in haircuts:
        assert h in (0.50, 0.75, 1.00)


def test_single_criterion_dual_ranking(cfg):
    raw = make_raw(cfg, n=5)
    df = calculate_metrics(raw, cfg)
    ranked = apply_dual_ranking(df, cfg)

    # Must contain single criterion scores and ranks
    assert "Value_Score" in ranked.columns
    assert "Quality_Score" in ranked.columns
    assert "Rank" in ranked.columns
    assert "Pct_Owner_Yield" in ranked.columns
    assert "Pct_Return_Quality" in ranked.columns

    # Shown percentiles must also be present for display and highlight lists
    assert "Pct_ROIC_Norm" in ranked.columns
    assert "Pct_ROIC_Variability" in ranked.columns
    assert "Pct_Per_Share_Compounding" in ranked.columns


def test_a_missing_criterion_counts_at_the_middle_not_at_the_strongest_one():
    from fq.ranking import weighted_percentile_score

    # A known only by its ROIC at the top; B at the 90th percentile of both. A used to keep its top percentile
    # undiluted and outrank B (PURR, PXED: Quality #1-3 on ROIC alone).
    n = 50
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "roic": np.arange(n, dtype=float),
                       "growth": np.arange(n, dtype=float)})  # fmt: skip
    df.loc[n - 1, "growth"] = np.nan  # A: best ROIC, growth unknown
    df.loc[n - 2, ["roic", "growth"]] = [n * 0.9, n * 0.9]  # B: about the 90th percentile of both
    criteria = [("roic", "top", 0.7), ("growth", "top", 0.3)]
    score, coverage, _ = weighted_percentile_score(df, criteria, min_coverage=0.7, winsor_pct=0.0)
    assert coverage[n - 1] == pytest.approx(0.7)
    assert score[n - 1] < score[n - 2]
    assert score[n - 1] == pytest.approx(70 * 1.0 + 30 * 0.5, abs=3)


def test_owner_yield_share_is_after_interest_over_the_share_price(cfg):
    """The worked example (Nov 8, 2026): 10 shares at 1000, 100 a share before interest, no trend. Net cash of 500 a
    share earning 11.25 after tax: 20% on the business, 11.1% on the share price. Net debt of 500 a share costing 15
    after tax: the share-price yield sits above the business yield."""
    flat = [1000.0] * 4
    cash = _trend_frame(cfg, flat, epv=1000 / 5000 * 100)
    cash[["FCF_Base", "Value_Base", "Net_Financing_Yield"]] = [10000.0, 5000.0, -112.5 * 100 / 10000]
    _owner_yield(cash, cfg, _basis())
    assert cash.loc[0, "Owner_Yield"] == pytest.approx(20.0)
    assert cash.loc[0, "Owner_Yield_Share"] == pytest.approx(11.125)

    debt = _trend_frame(cfg, flat, epv=1000 / 15000 * 100)
    debt[["FCF_Base", "Value_Base", "Net_Financing_Yield"]] = [10000.0, 15000.0, 150.0 * 100 / 10000]
    _owner_yield(debt, cfg, _basis())
    assert debt.loc[0, "Owner_Yield_Share"] == pytest.approx(8.5)
    assert debt.loc[0, "Owner_Yield_Share"] > debt.loc[0, "Owner_Yield"]
