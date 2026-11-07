"""Nov 6, 2026 review (user's choice): growth in US terms without past exchange rates, the Oct 28 trend rule for
compounding, the financial owner trend on its level's flow, one market for shown percentiles, a steady climber with a
flat year, ranked-only diagnostics, the share-class window, and the dividend lists in US terms."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.metrics.common import local_inflation, us_terms


def _set_revenue(raw, row, cfg, newest, yearly):
    """Revenue of `row` over every period: `newest` at Y0, `yearly` a year (TTM half a year past Y0)."""
    for p in cfg.fiscal_years:
        age = -0.5 if p == "TTM" else int(p[1:])
        raw.loc[row, f"Total Revenue_{p}"] = newest * yearly ** (-age)


def test_us_terms_takes_the_currencys_excess_out_and_restated_statements_get_us_inflation_back(cfg):
    df = pd.DataFrame({
        "Excess_Inflation_Y0": [0.10, 0.0, 1 / 1.03 - 1],
        "Inflation_Basis": ["nominal", "nominal", "restated"],
        "Inflation_US_Y0": [0.03, 0.03, 0.03],
        "PeriodEnd_Y0": ["2025-12-31"] * 3, "PeriodEnd_Y1": ["2024-12-31"] * 3,
    })  # fmt: skip
    values = pd.DataFrame({"Y0": [110.0, 100.0, 100.0], "Y1": [100.0, 100.0, 100.0]})
    out = us_terms(values, df, cfg)
    assert out.loc[0, "Y0"] / out.loc[0, "Y1"] == pytest.approx(1.0, rel=1e-3)  # 10% local = 0 in US terms
    assert out.loc[1].tolist() == [100.0, 100.0]  # a US company is unchanged
    assert out.loc[2, "Y0"] / out.loc[2, "Y1"] == pytest.approx(1.03, rel=1e-3)  # real 0 reads as US inflation
    assert local_inflation(df, cfg, "Y0").tolist() == pytest.approx([1.03 * 1.10 - 1, 0.03, 0.0])


def test_a_yen_growing_in_real_terms_is_not_flagged_for_the_yens_fall(cfg):
    raw = make_raw(cfg, n=3)
    raw["financialCurrency"] = None
    raw.loc[1, "financialCurrency"] = "JPY"  # built-in table: 2.5% a year against 3% in the US
    _set_revenue(raw, 1, cfg, 1e9, 1.025)  # yen revenue up with yen prices: flat in real terms
    no_fx = calculate_metrics(raw, cfg)
    for k, p in enumerate(cfg.annual_years):
        raw.loc[1, f"FX_Hist_{p}"] = 1.4**k  # the yen lost 29% a year against the dollar
    df = calculate_metrics(raw, cfg)
    assert abs(df.loc[1, "Revenue_Trend_Real"]) < 0.5 and not df.loc[1, "Revenue_Decline"]
    for col in ("Per_Share_Compounding", "Revenue_Growth_USD", "Growth_Used", "Owner_Yield"):
        assert df.loc[1, col] == pytest.approx(no_fx.loc[1, col], nan_ok=True)


def test_a_euro_falling_in_real_terms_is_flagged_though_the_euro_rose(cfg):
    raw = make_raw(cfg, n=3)
    raw["financialCurrency"] = None
    raw.loc[1, "financialCurrency"] = "EUR"  # 2.5% a year
    _set_revenue(raw, 1, cfg, 1e9, 1.025 * 0.95)  # 5% a year less volume
    for k, p in enumerate(cfg.annual_years):
        raw.loc[1, f"FX_Hist_{p}"] = 0.8**k  # the euro rose: in dollars of the time revenue grew ~19% a year
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Revenue_Trend_Real"] == pytest.approx(-5.0, abs=0.3)
    assert df.loc[1, "Revenue_Decline"]


def test_restated_statements_are_not_read_as_a_collapse(cfg):
    raw = make_raw(cfg, n=3)
    raw["financialCurrency"] = None
    raw.loc[1, "financialCurrency"] = "TRY"  # an operating company there restates (IAS 29): today's money
    _set_revenue(raw, 1, cfg, 1e9, 1.0)  # flat in today's lira: flat in real terms
    for k, p in enumerate(cfg.annual_years):
        raw.loc[1, f"FX_Hist_{p}"] = 2.0**k  # the old rates, which once multiplied today's lira
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Inflation_Basis"] == "restated"
    assert not df.loc[1, "Revenue_Decline"] and abs(df.loc[1, "Revenue_Trend_Real"]) < 0.5
    assert df.loc[1, "Per_Share_Compounding"] > 0  # flat real plus US inflation, as a US company's would read


def test_compounding_needs_the_trend_points_of_every_scored_trend(cfg):
    raw = make_raw(cfg, n=2)
    for p in ("Y2", "Y3", "Y4"):
        raw.loc[1, f"Total Revenue_{p}"] = np.nan  # TTM, Y0 and Y1 left: three points
    df = calculate_metrics(raw, cfg)
    assert np.isnan(df.loc[1, "Per_Share_Compounding"]) and df.loc[0, "Per_Share_Compounding"] > 0


def test_a_banks_owner_trend_is_fitted_on_its_clean_flow(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "industry"] = "Banks - Regional"
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Assets_{p}"] = raw.loc[0, f"Stockholders Equity_{p}"] * 10
    base = calculate_metrics(raw, cfg)
    gain = 0.5 * raw.loc[0, "Net Income_Y0"]  # a one-off gain in the latest year, after tax
    tax = raw.loc[0, "Tax Provision_Y0"] / raw.loc[0, "Pretax Income_Y0"]
    raw.loc[0, "Net Income_Y0"] += gain
    raw["Total Unusual Items_Y0"] = np.nan
    raw.loc[0, "Total Unusual Items_Y0"] = gain / (1 - tax)
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Business_Model"] == "bank"
    assert df.loc[0, "FCF_Reported_Y0"] > base.loc[0, "FCF_Reported_Y0"]  # raw net income holds the gain
    assert df.loc[0, "Owner_Trend"] == pytest.approx(base.loc[0, "Owner_Trend"])  # the trend does not


def test_asset_heavy_shown_percentiles_are_taken_on_the_operating_market(cfg):
    from fq.ranking import _scores

    df = pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(6)],
        "Business_Model": ["operating"] * 4 + ["asset_heavy"] * 2,
        "ROIC_Norm": [10.0, 20.0, 30.0, 40.0, 5.0, 6.0],
        "Owner_Yield": 5.0, "Return_Quality": [10.0, 20.0, 30.0, 40.0, 5.0, 6.0],
    })  # fmt: skip
    out = _scores(df, cfg, df, own_kind=False)
    assert out.loc[5, "Pct_ROIC_Norm"] < 50  # 6% is the second lowest of six, not the best of two


def test_a_steady_climber_with_one_flat_year_is_not_capped_but_a_trough_recovery_is():
    from fq.metrics.normalise import _pick_level

    level, _, capped = _pick_level([30, 30, 38, 44, 50], "RECENT_GROWTH_STABLE")
    assert (level, capped) == (50, False)
    _, _, capped = _pick_level([3.6, 7, 14.5, 21.4], "STRICT_GROWTH")  # FLXS: a step of 2x
    assert capped
    _, _, capped = _pick_level([30, 26, 38, 44, 50], "RECENT_GROWTH_STABLE")  # a real fall is no steady climb
    assert capped


def test_quality_history_correlates_the_ranked_companies_only(cfg):
    from fq.history import quality_ic

    n = 30
    lagged = pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)],
        "Rank_Group": ["ranked"] * 20 + ["financial"] * 10,
        "Quality_Score": np.linspace(1, 99, n),
    })  # fmt: skip
    after = pd.DataFrame({"outcome": np.linspace(0, 1, n)}, index=lagged["Ticker"])
    out = quality_ic(lagged, after, 1, cfg)
    assert out.loc[out["Measure"] == "Quality_Score", "N"].tolist() == [20]


def test_a_share_class_pair_on_both_sides_of_the_window_is_merged():
    from fq.filters import merge_share_classes

    df = pd.DataFrame({
        "Ticker": ["A", "B1", "B2"], "shortName": ["Alpha", "Beta", "Beta"],
        "Total Revenue_Y0": [1000.0, 1005.0, 1005.06], "Net Income_Y0": [50.0, 80.0, 80.0],
        "PeriodEnd_Y0": ["2025-12-31"] * 3, "sharesOutstanding": [1e8, 2e8, 1e8],
        "currency": ["USD"] * 3, "financialCurrency": ["USD"] * 3, "Fin_Rate": [1.0] * 3,
    })  # fmt: skip
    _, removed = merge_share_classes(df, verbose=False)
    assert removed == [("B2", "second listing or share class of B1")]


def test_screens_read_revenue_and_owner_earnings_in_us_terms_and_ratios_as_stored():
    from fq.report.screens import _annual, _owner_per_share

    df = pd.DataFrame({
        "Excess_Inflation_Y0": [0.10],
        "PeriodEnd_Y0": ["2025-12-31"], "PeriodEnd_Y1": ["2024-12-31"],
        "Total Revenue_Y0": [110.0], "Total Revenue_Y1": [100.0],
        "operatingMargins_Y0": [0.2], "operatingMargins_Y1": [0.2],
        "Owner_Earnings_Y0": [11.0], "Owner_Earnings_Y1": [10.0],
        "Ordinary Shares Number_Y0": [1.0], "Ordinary Shares Number_Y1": [1.0],
    })  # fmt: skip
    revenue = _annual(df, "Total Revenue", ["Y0", "Y1"])[0]
    assert revenue[0] / revenue[1] == pytest.approx(1.0, rel=1e-3)  # 10% local, all of it the currency's excess
    assert _annual(df, "operatingMargins", ["Y0", "Y1"])[0].tolist() == [0.2, 0.2]
    owner = _owner_per_share(df, ["Y0", "Y1"]).iloc[0]
    assert owner["OE_PS_Y0"] / owner["OE_PS_Y1"] == pytest.approx(1.0, rel=1e-3)


def test_dividend_growth_is_judged_in_us_terms():
    from fq.report.screens import dividend_growth_us, with_dividend_record

    df = pd.DataFrame({"Regular_DPS_Growth_10Y": [20.0, 5.0, 5.0], "Excess_Inflation_10Y": [0.25, 0.0, np.nan]})
    assert dividend_growth_us(df).tolist() == pytest.approx([1.2 / 1.25 * 100 - 100, 5.0, 5.0])
    assert with_dividend_record(df)["Regular_DPS_Growth_10Y_US"].iloc[0] < 0  # +20% in lira at 25% excess: a fall


def test_a_rupee_dividend_growing_slower_than_its_inflation_counts_its_growth_in_us_terms():
    from fq.report.screens import income_list
    from test_compounder import _twenty_year_row

    def r(ticker, rank, **over):
        row = _twenty_year_row(ticker, rank, roic=(15, 15, 14, 14))
        return row | {"Dividend_Unbroken_Years": 12.0, "Dividend_Cut_Recent": False, "Regular_DPS_Growth_10Y": 5.0,
                      "Recession_Dividend_Fall": -3.0, "Dividend_Yield": 3.0, "Owner_Yield": 8.0, "ROIC_Norm": 15.0,
                      "EPV_Per_Share_Yield": 8.0, "Dividend_WHT": 0.0,
                      "NetDebt_to_EBIT": 1.0, "Net_Debt": 10.0, "Common Stock Equity_Y0": 100.0} | over  # fmt: skip

    df = pd.DataFrame([r("USD", 1.0), r("INR", 2.0, Excess_Inflation_10Y=0.03)])  # 5% nominal, 2% in US terms
    growth = income_list(df).set_index("Ticker")["Income_Growth"]
    assert growth["USD"] == pytest.approx(5.0) and growth["INR"] == pytest.approx(1.05 / 1.03 * 100 - 100)


def test_the_ten_year_excess_inflation_spans_the_dividend_records_years(cfg):
    raw = make_raw(cfg, n=2)
    raw["financialCurrency"] = [None, "EGP"]
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Excess_Inflation_10Y"] == pytest.approx(1.20 / 1.03 - 1)
    assert np.isnan(df.loc[0, "Excess_Inflation_10Y"])  # no currency: unknown, and the lists count it as 0
