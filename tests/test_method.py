"""Ranking-method safeguards: leverage, ROIC inputs, stability, growth, coverage and suspect data."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.filters import clean_and_filter
from fq.metrics import calculate_metrics
from fq.metrics.quality import _return_quality, _roic_fit
from fq.ranking import apply_dual_ranking, criterion_percentile, top_n


def _set(raw: pd.DataFrame, i: int, concept: str, values, periods) -> None:
    for p, v in zip(periods, values, strict=True):
        raw.loc[i, f"{concept}_{p}"] = v


def test_negative_equity_is_not_the_best_balance_sheet(cfg):
    raw = make_raw(cfg, n=3)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Stockholders Equity_{p}"] = -2e9  # buyback-heavy: negative book equity, heavy debt
        raw.loc[0, f"Total Debt_{p}"] = 3e9
        raw.loc[1, f"Total Debt_{p}"] = 0.0  # net cash
        raw.loc[1, f"Cash And Cash Equivalents_{p}"] = 1e9
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Calculated_Debt_to_Equity"] < 0  # the old criterion would have ranked this best
    nd = df["NetDebt_to_EBIT"]
    assert nd[1] == 0 < nd[2] < nd[0]  # net cash: none, whatever the size of the pile
    assert criterion_percentile(nd, "bottom", 0).idxmin() == 0  # worst on leverage


def test_net_debt_without_operating_profit_is_worst(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        for concept in ("Operating Income", "EBIT"):
            raw.loc[0, f"{concept}_{p}"] = -1e8
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "NetDebt_to_EBIT"] == 99.0
    assert df.loc[0, "Interest_Coverage"] < 0


def test_roic_trend_is_in_percentage_points(cfg):
    raw = make_raw(cfg, n=2)
    # ROIC ~ operating margin x 0.79 / 1.1 with the synthetic balance sheet (tax 21%, capital 1.1 x revenue,
    # cash included), a little higher on the average capital of a growing company.
    for i, (old, new) in enumerate([(0.01, 0.03), (0.20, 0.30)]):
        for concept in ("Operating Income", "EBIT"):
            for p in cfg.fiscal_years:
                m = new if p in ("TTM", "Y0", "Y1") else old
                raw.loc[i, f"{concept}_{p}"] = raw.loc[i, f"Total Revenue_{p}"] * m * 1.1 / 0.79
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "ROIC_Trend"] == pytest.approx(2, rel=0.1)
    assert df.loc[1, "ROIC_Trend"] == pytest.approx(10, rel=0.1)  # was +50% vs +200%: now ranks higher


def test_roic_uses_notional_tax_and_average_capital_with_cash(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Tax Provision_TTM"] = np.nan
    raw.loc[0, "Short Term Investments_TTM"] = 1e8  # cash stays in the capital (as ROIC.ai)
    raw.loc[0, "Minority Interest_TTM"] = 5e7
    df = calculate_metrics(raw, cfg)
    r = raw.iloc[0]
    closing = r["Stockholders Equity_TTM"] + r["Total Debt_TTM"] + 5e7
    # Opening = a year before the TTM end, on the line from Y1 to Y0 (no minority interest then).
    y0, y1 = (r[f"Stockholders Equity_{p}"] + r[f"Total Debt_{p}"] for p in ("Y0", "Y1"))
    span = (pd.Timestamp(r["PeriodEnd_TTM"]) - pd.Timestamp(r["PeriodEnd_Y0"])).days / 365.25
    opening = y1 + span * (y0 - y1)
    expected = r["Operating Income_TTM"] * (1 - cfg.notional_tax_rate) * 100 / ((closing + opening) / 2)
    assert df.loc[0, "ROIC_TTM"] == pytest.approx(expected)

    raw.loc[0, "Tax Provision_TTM"] = -0.5 * r["Pretax Income_TTM"]  # tax credit: outside the bounds
    assert calculate_metrics(raw, cfg).loc[0, "ROIC_TTM"] == pytest.approx(expected)


def test_fcf_consistency_and_conversion(cfg):
    raw = make_raw(cfg, n=1)
    periods = cfg.fiscal_years  # TTM, Y0..Y4
    _set(raw, 0, "Free Cash Flow", [100, -50, 100, 100, 100, -50], periods)
    _set(raw, 0, "Stock Based Compensation", [0] * 6, periods)
    _set(raw, 0, "Net Income", [100, 100, 50, 100, 200, 100], periods)
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "FCF_Positive_Share"] == pytest.approx(4 / 6 * 100)  # TTM six months after Y0: both kept
    # Conversions 1, 0 (clipped), 2, 1, 0.5, 0 -> median 0.75
    assert df.loc[0, "FCF_Conversion"] == pytest.approx(0.75)
    assert pd.notna(df.loc[0, "ROIC_StdDev"]) and df.loc[0, "ROIC_StdDev"] >= 0


def test_growth_is_per_share_and_in_usd_of_the_time(cfg):
    raw = make_raw(cfg, n=3)
    for k, p in enumerate(cfg.annual_years):
        raw.loc[1, f"Ordinary Shares Number_{p}"] = 1e8 * 2 ** (-k / 4)  # company 1 doubles its share count
        raw.loc[2, f"FX_Hist_{p}"] = 1.5**k  # company 2: currency lost a third of its value each year
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Revenue_Per_Share_CAGR"] < df.loc[1, "Revenue_Growth_CAGR"]
    local = calculate_metrics(raw.drop(columns=[c for c in raw.columns if c.startswith("FX_Hist")]), cfg)
    g = local.loc[2, "Revenue_Growth_CAGR"] / 100
    assert df.loc[2, "Revenue_Growth_CAGR"] == pytest.approx(((1 + g) / 1.5 - 1) * 100)
    assert df.loc[0, "Revenue_Growth_CAGR"] == pytest.approx(local.loc[0, "Revenue_Growth_CAGR"])  # no history: today


def test_financial_mode_for_lenders(cfg):
    raw = make_raw(cfg, n=3)
    raw.loc[0, "Ticker"] = "KSPI"  # override: a bank, Yahoo calls it software
    raw.loc[1, ["industry", "sector"]] = ["Insurance - Diversified", "Financial Services"]
    raw.loc[2, ["industry", "sector"]] = ["Utilities - Regulated Electric", "Utilities"]
    for p in cfg.fiscal_years:
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = 5e9  # backs customer deposits
        raw.loc[[0, 1], f"Total Assets_{p}"] = raw.loc[[0, 1], f"Stockholders Equity_{p}"] * 8
    df = calculate_metrics(raw, cfg)
    assert list(df["Financial_Mode"]) == [True, True, False]  # utility debt is real debt
    assert list(df["Business_Model"]) == ["bank", "insurer", "asset_heavy"]
    assert df.loc[0, "enterpriseValue"] == df.loc[0, "marketCap"]
    average_equity = (raw.loc[0, "Stockholders Equity_Y0"] + raw.loc[0, "Stockholders Equity_Y1"]) / 2
    assert df.loc[0, "ROIC_Y0"] == pytest.approx(raw.loc[0, "Net Income_Y0"] * 100 / average_equity)
    assert df.loc[0, "FCF_Adj_Y0"] == raw.loc[0, "Net Income_Y0"]
    assert df.loc[[0, 1], ["NetDebt_to_EBIT", "FCF_Conversion", "Interest_Coverage"]].isna().all().all()
    assert df.loc[0, "Equity_to_Assets"] == pytest.approx(12.5)
    assert df.loc[[0, 1], ["ROA_Avg", "Pretax_Yield_Avg", "Book_Compounding"]].notna().all().all()
    assert df.loc[2, ["ROA_Avg", "Book_Compounding"]].isna().all()  # never in their distribution
    assert df["Pretax_Yield_Avg"].notna().all()  # per-share Value criterion of every model
    assert df.loc[2, "enterpriseValue"] > df.loc[2, "marketCap"]
    assert apply_dual_ranking(df, cfg)["Insufficient_Data"].sum() == 0  # own criteria, full coverage

    kept, removed = clean_and_filter(df, list(df["Ticker"]))
    assert removed == []  # every model is measured now; only shells and mortgage REITs are excluded


def test_business_models():
    from fq.universe import business_model, is_excluded, is_financial

    assert business_model("V", "Credit Services") == "operating" and not is_financial("V", "Credit Services")
    assert business_model("AXP", "Credit Services") == "bank" and is_financial("AXP", "Credit Services")
    assert business_model("BLK", "Asset Management") == "operating"  # fee business
    assert business_model("NTRS", "Asset Management") == "bank"  # custody bank, by override
    assert business_model("MMC", "Insurance Brokers") == "operating"
    assert business_model("PGR", "Insurance—Property & Casualty") == "insurer"  # dash variants match
    assert business_model("BRK-B", "Insurance - Diversified") == "holding"
    assert business_model("ADP", "Software - Application") == "client_cash"
    assert business_model("UNH", "Healthcare Plans") == "float"
    assert (
        business_model("O", "REIT - Retail") == business_model("NEE", "Utilities - Regulated Electric") == "asset_heavy"
    )
    assert business_model("HON", "Conglomerates") == "operating"
    assert not is_excluded("XYZ", "REIT - Mortgage") and is_excluded("XYZ", "Shell Companies")
    assert business_model("XYZ", "REIT - Mortgage") == "bank"
    assert not is_excluded("O", "REIT - Retail") and not is_excluded("JPM", "Banks - Diversified")


def test_asset_heavy_capex_above_depreciation_buys_growth(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, ["industry", "sector"]] = ["Utilities - Regulated Electric", "Utilities"]
    for p in cfg.fiscal_years:
        rev = raw.loc[0, f"Total Revenue_{p}"]
        raw.loc[[0, 1], f"Depreciation And Amortization_{p}"] = rev * 0.05
        raw.loc[[0, 1], f"Capital Expenditure_{p}"] = -rev * 0.15  # a growing rate base
        raw.loc[[0, 1], f"Free Cash Flow_{p}"] = raw.loc[[0, 1], f"Operating Cash Flow_{p}"] - rev * 0.15
    df = calculate_metrics(raw, cfg)
    sbc = raw.loc[0, "Stock Based Compensation_Y0"]
    rev0 = raw.loc[0, "Total Revenue_Y0"]
    assert df.loc[0, "FCF_Adj_Y0"] == pytest.approx(raw.loc[0, "Operating Cash Flow_Y0"] - rev0 * 0.05 - sbc)
    assert df.loc[0, "Growth_Capex_Y0"] == pytest.approx(rev0 * 0.10)
    # Operating without net PPE: no capital intensity to split capex by, so all of it stays charged.
    assert df.loc[1, "Growth_Capex_Y0"] == 0
    assert df.loc[1, "FCF_Adj_Y0"] == pytest.approx(
        raw.loc[1, "Free Cash Flow_Y0"] - raw.loc[1, "Stock Based Compensation_Y0"]
    )


def _builder(cfg, rows=(0,)):
    """Synthetic operating companies building plant: net PPE 50% of revenue, capex 10%, depreciation 3%."""
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        rev = raw.loc[list(rows), f"Total Revenue_{p}"]
        raw.loc[list(rows), f"Net PPE_{p}"] = rev * 0.5
        raw.loc[list(rows), f"Depreciation_{p}"] = rev * 0.03
        raw.loc[list(rows), f"Depreciation And Amortization_{p}"] = rev * 0.06  # acquired intangibles: no capex
        raw.loc[list(rows), f"Capital Expenditure_{p}"] = -rev * 0.10
        raw.loc[list(rows), f"Free Cash Flow_{p}"] = raw.loc[list(rows), f"Operating Cash Flow_{p}"] - rev * 0.10
    return raw


def test_organic_growth_capex_is_owner_earnings(cfg):
    raw = _builder(cfg)
    df = calculate_metrics(raw, cfg)
    rev0, rev1 = raw.loc[0, "Total Revenue_Y0"], raw.loc[0, "Total Revenue_Y1"]
    # Greenwald: intensity 0.5 x revenue added beyond that year's US inflation, at most the capex above depreciation.
    us = df.loc[0, "Inflation_US_Y0"]
    growth = min(0.5 * (rev0 / rev1 / (1 + us) - 1) * rev0, 0.07 * rev0)
    assert df.loc[0, "Growth_Capex_Y0"] == pytest.approx(growth)
    assert df.loc[0, "FCF_Adj_Y0"] == pytest.approx(df.loc[0, "FCF_Reported_Y0"] + growth)
    oldest = cfg.annual_years[-1]
    # No older year to measure growth against: the share of capex the other years spent on growth.
    assert df.loc[0, f"Growth_Capex_{oldest}"] == pytest.approx(growth / rev0 * raw.loc[0, f"Total Revenue_{oldest}"])
    assert df.loc[0, "Growth_Capex_TTM"] == pytest.approx(growth / rev0 * raw.loc[0, "Total Revenue_TTM"])

    plain = calculate_metrics(raw, replace(cfg, organic_capex_models=()))
    assert df.loc[0, "FCF_Positive_Share"] == plain.loc[0, "FCF_Positive_Share"]  # Quality keeps all capex
    assert df.loc[0, "FCF_Conversion"] == plain.loc[0, "FCF_Conversion"]
    assert df.loc[1, "FCF_Adj_Y0"] == pytest.approx(plain.loc[1, "FCF_Adj_Y0"])  # no plant data: unchanged


def test_shrinking_or_cyclical_companies_get_no_growth_capex(cfg):
    raw = _builder(cfg, rows=(0, 1))
    years = list(reversed(cfg.fiscal_years))  # oldest first
    for k, p in enumerate(years):  # row 1 shrinks 5% a year
        scale = 0.95**k / raw.loc[1, f"Total Revenue_{p}"] * 1e9
        for col in [c for c in raw.columns if c.endswith(f"_{p}") and not c.startswith(("PeriodEnd", "Ordinary"))]:
            raw.loc[1, col] = raw.loc[1, col] * scale
    for p, sign in zip(cfg.fiscal_years, [1, 1, -1, 1, -1, 1], strict=True):  # row 0: profit changes sign
        for concept in ("Operating Income", "EBIT"):
            raw.loc[0, f"{concept}_{p}"] = sign * abs(raw.loc[0, f"{concept}_{p}"])
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Growth_Used"] < 0 and df.loc[0, "Cyclical"]
    assert not df["Organic_Capex"].any()
    assert (df["FCF_Adj_Y0"] == df["FCF_Reported_Y0"]).all()


def test_stock_pay_dilution_is_not_counted_twice(cfg):
    raw = make_raw(cfg, n=1)
    years = list(reversed(cfg.fiscal_years))
    for k, p in enumerate(years):  # shares grow 1% a year, all of it stock pay (SBC already deducted from FCF)
        raw.loc[0, f"Ordinary Shares Number_{p}"] = 1e8 * 1.01**k
        raw.loc[0, f"Stock Based Compensation_{p}"] = 6e7
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "SBC_Dilution"] == pytest.approx(6e7 * 100 / df.loc[0, "marketCap"])  # today's share count
    assert df.loc[0, "SBC_Dilution"] > 1
    assert df.loc[0, "Revenue_Growth_USD"] - 1.1 < df.loc[0, "Revenue_Per_Share_CAGR"] < df.loc[0, "Revenue_Growth_USD"]
    assert df.loc[0, "Growth_Used"] == pytest.approx(min(df.loc[0, "Revenue_Growth_USD"], cfg.growth_cap))


def test_cyclical_flag(cfg):
    raw = make_raw(cfg, n=2)
    for p, sign in zip(cfg.fiscal_years, [1, 1, -1, 1, -1, 1], strict=True):
        for concept in ("Operating Income", "EBIT"):
            raw.loc[0, f"{concept}_{p}"] = sign * abs(raw.loc[0, f"{concept}_{p}"])
    df = calculate_metrics(raw, cfg)
    assert list(df["Cyclical"]) == [True, False]  # steady grower is not flagged


def _margins(raw, i, margins, cfg):
    """Operating margins per period (TTM first), on the company's own revenue."""
    for p, m in zip(cfg.fiscal_years, margins, strict=True):
        for concept in ("Operating Income", "EBIT"):
            raw.loc[i, f"{concept}_{p}"] = raw.loc[i, f"Total Revenue_{p}"] * m


def test_cyclicals_at_peak_margins_leave_top_lists(cfg):
    raw = make_raw(cfg, n=4)
    raw.loc[[0, 1], "industry"] = "Gold"  # cyclical by list
    _margins(raw, 0, [0.44, 0.44, 0.36, 0.05, 0.04, 0.04], cfg)  # a boom filling most of the window
    _margins(raw, 2, [0.16, 0.16, 0.14, 0.12, 0.10, 0.10], cfg)  # same climb, not cyclical: a better business
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Peak_Margin_Ratio"] == pytest.approx(44 / 4.2)  # 30th percentile of 44, 36, 5, 4, 4: 4.2%
    assert df.loc[2, "Peak_Margin_Ratio"] > cfg.peak_margin_ratio and not df.loc[2, "Cyclical"]
    assert df["Peak_Margin"].tolist() == [True, False, False, False]  # company 1: cyclical with steady margins
    ranked = apply_dual_ranking(df, cfg)
    assert raw.loc[0, "Ticker"] not in set(top_n(ranked, "Value_Rank", 4)["Ticker"])

    for p in ("Y2", "Y3", "Y4"):  # under three annual margins: no judgement
        raw.loc[0, f"Total Revenue_{p}"] = np.nan
    assert not calculate_metrics(raw, cfg).loc[0, "Peak_Margin"]


def _balance(raw, i, cfg, ca, cl, re, liabilities, equity, assets=10e9):
    for p in cfg.fiscal_years:
        _set(raw, i, "Current Assets", [ca], [p])
        _set(raw, i, "Current Liabilities", [cl], [p])
        _set(raw, i, "Retained Earnings", [re], [p])
        _set(raw, i, "Total Liabilities Net Minority Interest", [liabilities], [p])
        _set(raw, i, "Stockholders Equity", [equity], [p])
        _set(raw, i, "Total Assets", [assets], [p])


def test_distress_needs_a_weak_balance_sheet_and_strained_cash(cfg):
    raw = make_raw(cfg, n=4)
    for i in range(4):  # all with a weak balance sheet: working capital short, accumulated losses
        _balance(raw, i, cfg, ca=1e9, cl=3e9, re=-4e9, liabilities=12e9, equity=-2e9)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Interest Expense_{p}"] = raw.loc[0, f"EBIT_{p}"]  # EBIT covers interest once
        raw.loc[1, f"Operating Cash Flow_{p}"] = -raw.loc[1, f"Total Revenue_{p}"] * 0.05  # burns cash
        raw.loc[1, f"Free Cash Flow_{p}"] = -raw.loc[1, f"Total Revenue_{p}"] * 0.09
    _balance(raw, 3, cfg, ca=6e9, cl=2e9, re=5e9, liabilities=4e9, equity=6e9)  # sound balance sheet
    df = calculate_metrics(raw, cfg)
    ta = 10e9
    ebit = raw.loc[3, "EBIT_TTM"]
    assert df.loc[3, "Altman_Z"] == pytest.approx(6.56 * 0.4 + 3.26 * 0.5 + 6.72 * ebit / ta + 1.05 * 6 / 4)
    assert df.loc[2, "Altman_Z"] < cfg.distress_z  # weak balance sheet, but cash and cover are fine: not flagged
    assert df["Distress"].tolist() == [True, True, False, False]
    ranked = apply_dual_ranking(df, cfg)
    assert not set(raw.loc[[0, 1], "Ticker"]) & set(top_n(ranked, "Value_Rank", 4)["Ticker"])
    assert ranked["Value_Score"].notna().all()  # still scored

    raw.loc[0, [f"Retained Earnings_{p}" for p in cfg.fiscal_years]] = np.nan  # a gap is no judgement
    assert not calculate_metrics(raw, cfg).loc[0, "Distress"]


def test_beneish_m_score_by_hand(cfg):
    raw = make_raw(cfg, n=3)
    t, s = cfg.latest, cfg.prior
    for p, rev, rec, gp, sga, dep, ppe, ca in ((t, 1.2e9, 0.3e9, 0.48e9, 0.12e9, 50e6, 450e6, 0.6e9),
                                               (s, 1.0e9, 0.2e9, 0.40e9, 0.10e9, 50e6, 450e6, 0.5e9)):  # fmt: skip
        _set(raw, 0, "Total Revenue", [rev], [p])
        _set(raw, 0, "Receivables", [rec], [p])
        _set(raw, 0, "Gross Profit", [gp], [p])
        _set(raw, 0, "Selling General And Administration", [sga], [p])
        for name in ("Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion",
                     "Reconciled Depreciation"):  # fmt: skip
            _set(raw, 0, name, [dep], [p])
        _set(raw, 0, "Net PPE", [ppe], [p])
        _set(raw, 0, "Current Assets", [ca], [p])
        _set(raw, 0, "Total Assets", [2e9], [p])
        _set(raw, 0, "Total Liabilities Net Minority Interest", [1e9], [p])
    _set(raw, 0, "Net Income", [0.2e9], [t])
    _set(raw, 0, "Operating Cash Flow", [0.1e9], [t])
    df = calculate_metrics(raw, cfg)
    dsri = (0.3 / 1.2) / (0.2 / 1.0)
    aqi = (1 - 1.05 / 2) / (1 - 0.95 / 2)
    expected = -4.84 + 0.92 * dsri + 0.528 * 1.0 + 0.404 * aqi + 0.892 * 1.2 + 0.115 * 1.0 - 0.172 * 1.0
    expected += 4.679 * (0.1 / 2) - 0.327 * 1.0
    assert df.loc[0, "Beneish_M"] == pytest.approx(expected)
    assert df["Beneish_M"].iloc[1:].isna().all()  # no receivables / SG&A rows: left blank, not guessed
    assert "Beneish_M" not in {c[0] for c in cfg.quality_criteria}  # observed, never scored


def test_growth_per_share_is_quality_not_value(cfg):
    quality, value = {c[0] for c in cfg.quality_criteria}, {c[0] for c in cfg.value_criteria}
    assert "Per_Share_Compounding" in {c[0] for c in cfg.shown_percentiles}
    assert "Owner_Yield" in value and "Return_Quality" in quality
    # What a long-term holder's share grew by and the share count are Quality (user's choice, Oct 28, 2026), never
    # Value, and the same question for every model, so market-wide.
    assert {"Per_Share_Compounding", "Share_Reduction_Rate"} <= quality
    assert not {"Per_Share_Compounding", "Share_Reduction_Rate"} & value
    assert {"Per_Share_Compounding", "Share_Reduction_Rate"} <= set(cfg.quality_market_criteria)
    weights = {c[0]: c[2] for c in cfg.quality_criteria}
    assert weights["Return_Quality"] / sum(weights.values()) >= cfg.min_coverage  # alone it still scores


def _toy(cfg, a, b):
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(len(a))], "a": a, "b": b})
    crit = (("a", "top", 3.0), ("b", "top", 1.0))
    return df, replace(cfg, value_criteria=crit, quality_criteria=crit, winsor_pct=0)


def test_missing_criterion_is_not_last_but_low_coverage_is_insufficient(cfg):
    df, c = _toy(cfg, a=[10, 5, 1, np.nan], b=[np.nan, 5, 1, 10])
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    assert ranked.loc["T0", "Rank"] == 1  # best on the heavy criterion, b missing
    assert ranked.loc["T0", "Value_Coverage"] == pytest.approx(0.75)
    assert ranked.loc["T3", "Insufficient_Data"] and pd.isna(ranked.loc["T3", "Value_Rank"])
    assert pd.isna(ranked.loc["T3", "Rank"]) and ranked.index[-1] == "T3"


def test_combined_order_is_the_worse_position_so_one_dimension_does_not_buy_the_other(cfg):
    # T0: best Value, worst Quality; T1: second on both. A sum of positions ranked T0 first (1 + 5 = 6 vs 4).
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(5)], "v": [5.0, 4, 3, 2, 1], "q": [1.0, 4, 3, 2, 5]})
    c = replace(cfg, value_criteria=(("v", "top", 1.0),), quality_criteria=(("q", "top", 1.0),), winsor_pct=0,
                quality_market_criteria=())  # fmt: skip
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    assert ranked.loc["T1", "Rank"] < ranked.loc["T0", "Rank"]
    assert ranked.loc["T0", "Rank"] >= 4  # worse position 5, like T4 (5, 1): after T1, T2, T3


def test_combined_ties_on_the_worse_position_go_to_the_lower_sum(cfg):
    # T0 positions (2, 3), T1 (3, 1): both worse 3; T1's sum 4 beats T0's 5.
    df = pd.DataFrame({"Ticker": ["T0", "T1", "T2"], "v": [2.0, 1, 3], "q": [1.0, 3, 2]})
    c = replace(cfg, value_criteria=(("v", "top", 1.0),), quality_criteria=(("q", "top", 1.0),), winsor_pct=0,
                quality_market_criteria=())  # fmt: skip
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    assert ranked.loc["T1", "Rank"] < ranked.loc["T0", "Rank"]


def test_utility_quality_is_taken_against_the_market_and_bank_quality_among_banks(cfg):
    n = 20
    models = ["operating"] * 10 + ["asset_heavy"] * 5 + ["bank"] * 5
    rq = list(np.linspace(20, 40, 10)) + [5.0, 6, 7, 8, 9] + [5.0, 6, 7, 8, 9]
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "Business_Model": models, "v": np.arange(n, 0, -1.0),
                       "Return_Quality": rq})  # fmt: skip
    c = replace(cfg, value_criteria=(("v", "top", 1.0),), quality_criteria=(("Return_Quality", "top", 1.0),),
                financial_quality_criteria=(("Return_Quality", "top", 1.0),), winsor_pct=0,
                quality_market_criteria=())  # fmt: skip
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    scores = ranked.groupby("Business_Model")["Quality_Score"]
    utilities, banks = scores.get_group("asset_heavy"), scores.get_group("bank")
    assert utilities.max() < 40  # the best utility's 9% is near the bottom of the market, not top of its peers
    assert banks.max() > 90  # a bank's ROE is still compared with banks


def test_insufficient_data_sorts_after_every_scored_company(cfg):
    # T0 is best on Value but has no Quality; it used to sort (and rank) ahead of mid-table companies.
    n = 40
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "v": np.arange(n, 0, -1.0), "q": np.arange(n, 0, -1.0)})
    df.loc[0, "q"] = np.nan
    c = replace(cfg, value_criteria=(("v", "top", 1.0),), quality_criteria=(("q", "top", 1.0),), winsor_pct=0)
    ranked = apply_dual_ranking(df, c)
    assert ranked["Ticker"].iloc[-1] == "T0" and pd.isna(ranked["Rank"].iloc[-1])
    assert ranked["Rank"].iloc[:-1].tolist() == list(range(1, n))


def test_fcf_base_uses_todays_exchange_rate(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        for concept in ("Total Revenue", "Free Cash Flow"):  # flat revenue: the 3-year average is the base
            raw.loc[1, f"{concept}_{p}"] = raw.loc[1, f"{concept}_Y0"]
    plain = calculate_metrics(raw, cfg)
    for k, p in enumerate(cfg.annual_years):
        raw.loc[1, f"FX_Hist_{p}"] = 1.5**k  # the currency fell: old revenue was worth more USD then
    fx = calculate_metrics(raw, cfg)
    assert fx.loc[1, "FCFy_Norm"] == pytest.approx(plain.loc[1, "FCFy_Norm"])
    assert fx.loc[1, "Revenue_Growth_CAGR"] < plain.loc[1, "Revenue_Growth_CAGR"]  # growth stays in USD then


def test_recent_change_scales_the_ttm_overlap(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "PeriodEnd_TTM"] = (pd.Timestamp(raw.loc[0, "PeriodEnd_Y0"]) + pd.DateOffset(months=3)).strftime(
        "%Y-%m-%d"
    )
    for concept in ("Operating Income", "EBIT"):
        raw.loc[0, f"{concept}_TTM"] = raw.loc[0, f"{concept}_Y0"] * 0.91  # one quarter down 36% on the year
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "EBIT_Change_Recent"] == pytest.approx(-36, abs=0.5)


def test_ttm_buyback_is_a_twelve_month_change(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "PeriodEnd_TTM"] = (pd.Timestamp(raw.loc[0, "PeriodEnd_Y0"]) + pd.DateOffset(months=3)).strftime(
        "%Y-%m-%d"
    )
    for c in ("Cash Dividends Paid", "Repurchase Of Capital Stock", "Issuance Of Capital Stock"):
        for p in cfg.fiscal_years:
            raw.loc[0, f"{c}_{p}"] = 0.0
    _set(raw, 0, "Ordinary Shares Number", [100e6, 90e6, 89e6], ["Y1", "Y0", "TTM"])
    df = calculate_metrics(raw, cfg)
    span = (pd.Timestamp(df.loc[0, "PeriodEnd_TTM"]) - pd.Timestamp(df.loc[0, "PeriodEnd_Y0"])).days / 365.25
    year_ago = 100e6 - span * 10e6  # on the Y1 -> Y0 line, a year before the TTM end
    expected = (year_ago - 89e6) / year_ago * 100  # per share: the part of the company each holder gained
    assert df.loc[0, "DirectCashtoInvestors_Yield_TTM"] == pytest.approx(expected, rel=0.01)


def test_winsorized_outlier_ties_with_its_neighbour():
    v = pd.Series([*range(1, 30), 1000.0])
    pct = criterion_percentile(v, "top", 0.01)
    assert pct.iloc[-1] == pct.iloc[-2] > pct.iloc[-3]


def test_suspect_data_leaves_top_lists(cfg):
    raw = make_raw(cfg, n=4)
    for p in cfg.fiscal_years:  # FCF far above revenue x EV plausibility
        raw.loc[0, f"Free Cash Flow_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 5
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Data_Suspect"] and "FCF yield" in df.loc[0, "Suspect_Reasons"]
    assert not df.loc[1:, "Data_Suspect"].any()
    ranked = apply_dual_ranking(df, cfg)
    assert raw.loc[0, "Ticker"] in set(ranked["Ticker"])
    assert raw.loc[0, "Ticker"] not in set(top_n(ranked, "Value_Rank", 4)["Ticker"])


def test_mortgage_reits_are_unexcluded_and_routed_to_bank(cfg):
    from fq.universe import business_model

    df = calculate_metrics(make_raw(cfg, n=3), cfg)
    df.loc[0, "industry"] = "REIT - Mortgage"
    df.loc[1, "industry"] = "REIT - Retail"  # equity REITs are measured as asset-heavy
    scope = list(df["Ticker"])
    assert clean_and_filter(df, scope)[1] == []
    assert business_model(df.loc[0, "Ticker"], "REIT - Mortgage") == "bank"
    assert business_model(df.loc[1, "Ticker"], "REIT - Retail") == "asset_heavy"


def test_falling_margins_are_valued_at_the_latest_level(cfg):
    raw = make_raw(cfg, n=2)
    for k, p in enumerate(reversed(cfg.fiscal_years)):  # oldest first: FCF margin 40% falling to 15%
        raw.loc[0, f"Free Cash Flow_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * (0.40 - 0.05 * k)
        raw.loc[0, f"Stock Based Compensation_{p}"] = 0.0
        raw.loc[0, f"Interest Expense_{p}"] = 0.0  # no financing: FCF before and after it are the same
    df = calculate_metrics(raw, cfg)
    expected = 0.15 * raw.loc[0, "Total Revenue_TTM"] * 100 / df.loc[0, "marketCap"]
    assert df.loc[0, "FCFy_Norm"] == pytest.approx(expected)


def test_financials_are_scored_on_their_own_criteria(cfg):
    df, c = _toy(cfg, a=[10, 5, 1], b=[np.nan, 5, 1])
    c = replace(c, financial_value_criteria=(("a", "top", 1.0),), financial_quality_criteria=(("a", "top", 1.0),))
    df["Financial_Mode"] = [True, False, False]
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    assert ranked.loc["T0", "Value_Coverage"] == 1.0  # b is not one of its criteria
    assert ranked.loc["T1", "Value_Coverage"] == 1.0 and ranked.loc["T0", "Quality_Score"] == pytest.approx(100.0)


def test_unranked_models_are_scored_but_not_ranked(cfg):
    df, c = _toy(cfg, a=[10, 5, 1, 7], b=[1, 5, 1, 2])
    c = replace(c, financial_value_criteria=c.value_criteria, financial_quality_criteria=c.quality_criteria,
                ranked_models=("operating", "client_cash", "float"),
                value_market_families=("operating", "asset_heavy"))  # the mechanism, with models left out  # fmt: skip
    df["Business_Model"] = ["bank", "operating", "operating", "operating"]
    ranked = apply_dual_ranking(df, c)
    alone = apply_dual_ranking(df[df["Business_Model"] == "operating"].reset_index(drop=True), c).set_index("Ticker")
    by = ranked.set_index("Ticker")
    for t in alone.index:  # operating companies score and rank exactly as without the bank
        assert by.at[t, "Value_Score"] == pytest.approx(alone.at[t, "Value_Score"])
        assert by.at[t, "Quality_Score"] == pytest.approx(alone.at[t, "Quality_Score"])
        assert by.at[t, "Value_Rank"] == alone.at[t, "Value_Rank"] and by.at[t, "Rank"] == alone.at[t, "Rank"]
    assert ranked["Ticker"].iloc[-1] == "T0" and pd.isna(ranked["Rank"].iloc[-1]) and not by.at["T0", "Ranked_Model"]
    # A bank is judged against its peers only (alone: 1.0), never against the market's yields on EV.
    assert by.at["T0", "Value_Score"] == pytest.approx(100.0)
    assert by.at["T0", "Quality_Score"] == pytest.approx(100.0)  # Quality only against its peer group
    # Asset-heavy Value against the market (a=10 tops it, b=1 ties the bottom: 1/3); with a sector blend of 0.5,
    # half of it against its peers (1.0).
    as_heavy = df.assign(Business_Model=["asset_heavy", *["operating"] * 3])
    heavy = apply_dual_ranking(as_heavy, c).set_index("Ticker")
    assert heavy.at["T0", "Value_Score"] == pytest.approx((3 * 1.0 + 1 / 3) / 4 * 100)
    heavy = apply_dual_ranking(as_heavy, replace(c, sector_blend=0.5)).set_index("Ticker")
    assert heavy.at["T0", "Value_Score"] == pytest.approx((3 * 1.0 + 1 * (0.5 + 0.5 / 3)) / 4 * 100)

    group = apply_dual_ranking(df.iloc[[1]].reset_index(drop=True), c, reference=ranked, benchmark="b")
    assert group["Benchmark_N"].iloc[0] == 3  # the bank is not counted among the ranked
    ranked_on = apply_dual_ranking(df, replace(c, ranked_models=("operating", "bank"), rank_apart_models=()))
    assert ranked_on["Rank"].notna().all()  # switching a model on ranks it


def test_banks_insurers_holdings_reits_and_loss_makers_are_scored_but_ranked_apart(cfg):
    # User's choice, Nov 5, 2026: the rankings hold profitable operating companies only; scores do not move.
    df, c = _toy(cfg, a=[10, 5, 1, 7, 6, 4, 3], b=[1, 5, 1, 2, 3, 4, 2])
    c = replace(c, financial_value_criteria=c.value_criteria, financial_quality_criteria=c.quality_criteria)
    df["Business_Model"] = ["bank", "operating", "holding", "operating", "insurer", "asset_heavy", "operating"]
    df["industry"] = [None, None, None, None, None, "REIT - Retail", None]
    df["Owner_Yield"] = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -1.0]  # T6 loses money in a normal year
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    assert ranked["Rank_Group"].to_dict() == {"T0": "financial", "T1": "ranked", "T2": "holding", "T3": "ranked",
                                              "T4": "financial", "T5": "real_estate", "T6": "unprofitable"}  # fmt: skip
    apart = ranked["Rank_Group"] != "ranked"
    assert ranked.loc[apart, ["Rank", "Value_Rank", "Quality_Rank"]].isna().all().all()
    assert ranked.loc[~apart, "Rank"].tolist() and ranked["Ranked_Model"].all()  # still in the market's reference
    # The scores are those of one ranking: taking the kinds apart moves positions only.
    together = apply_dual_ranking(df, replace(c, rank_apart_models=())).set_index("Ticker")
    for col in ("Value_Score", "Quality_Score"):
        assert ranked[col].to_dict() == pytest.approx(together[col].to_dict())
    # Cheap among its own kind: a lender's levered yield is not set against EV yields.
    market = apply_dual_ranking(df, replace(c, value_market_families=("operating", "asset_heavy", "financial")))
    assert ranked.at["T0", "Value_Score"] > market.set_index("Ticker").at["T0", "Value_Score"]
    # The ranked rows first, then the others.
    order = apply_dual_ranking(df, c)["Ticker"].tolist()
    assert set(order[:2]) == {"T1", "T3"}


def test_peer_groups_fall_back_to_the_family():
    from fq.ranking import _peer_percentile

    df = pd.DataFrame({"Business_Model": ["bank"] * 3 + ["insurer"] * 2, "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    rows = pd.Series(True, index=df.index)
    by_peer = _peer_percentile(df, "x", "top", 0.0, rows, df, min_members=2)
    assert by_peer[3] == pytest.approx(0.5)  # among the two insurers
    by_family = _peer_percentile(df, "x", "top", 0.0, rows, df, min_members=3)
    assert by_family[3] == pytest.approx(0.8)  # too few insurers: among all five financials


def test_global_benchmark_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no portfolio.txt: holdings would join "others"
    from fq.universe import BENCHMARK_SETS, benchmark_part, resolve_selection

    sp500 = benchmark_part("sp500")
    assert {"AAPL", "V", "BRK-B", "DECK"} <= set(sp500) and len(sp500) > 490  # index lists taken whole
    assert len(benchmark_part("ibex35")) == 34 and len(benchmark_part("dax")) == 40
    assert "0027.HK" in benchmark_part("others") and "5425.T" not in benchmark_part("others")  # excluded
    combined = resolve_selection("benchmark")
    assert combined.label == "benchmark" and {"AAPL", "ITX.MC", "SAP.DE", "0001.HK"} <= set(combined.tickers)
    assert len(benchmark_part("eurostoxx50")) == 50 and {"MC.PA", "NOKIA.HE"} <= set(combined.tickers)
    assert len(benchmark_part("ftse100")) == 99 and len(benchmark_part("sti")) == 30
    assert {"BT-A.L", "D05.SI"} <= set(combined.tickers)
    assert len(benchmark_part("klci")) == 30 and "1155.KL" in combined.tickers
    assert "1299.HK" in benchmark_part("hsi_hk") and "0700.HK" not in benchmark_part("hsi_hk")  # HK companies only
    assert (
        len(benchmark_part("obx")) == 25 and len(benchmark_part("wig20")) == 20 and len(benchmark_part("omxs30")) == 27
    )
    assert {"EQNR.OL", "VOLV-B.ST", "PKO.WA"} <= set(combined.tickers)
    # One listing per company: AstraZeneca and Nordea are in the benchmark through London and Helsinki.
    assert not {"AZN.ST", "NDA-SE.ST"} & set(combined.tickers) and {"AZN.L", "NDA-FI.HE"} <= set(combined.tickers)
    # MSCI World, taken whole; the primary listing (MSCI's) replaces a second one held elsewhere.
    msci = set(benchmark_part("msci_world"))
    assert len(msci) > 1200 and {"7203.T", "RY.TO", "NESN.SW", "BHP.AX"} <= msci
    assert {"ABBN.SW", "MT.AS", "CCEP"} <= msci and not {"ABB.ST", "MTS.MC", "CCEP.L"} & set(combined.tickers)
    assert not {"RIO.AX", "ATCO-B.ST", "0019.HK", "HEIO.AS"} & set(combined.tickers)  # same company, held once
    # MSCI ACWI emerging markets: net of every other list, A shares over H shares, one class per company.
    em = set(benchmark_part("msci_em"))
    assert len(em) > 800 and {"2330.TW", "000660.KS", "HDFCBANK.NS", "PETR4.SA", "601398.SS", "2222.SR"} <= em
    assert not em & msci and not {"1398.HK", "005935.KS", "PETR3.SA"} & em and "1155.KL" not in em
    # ASX 50: the primary listing of each company; the ASX lines of US-primary, NZ and London-listed companies are dups.
    asx = set(benchmark_part("asx50"))
    assert len(asx) == 43 and {"CAR.AX", "MIN.AX", "JHX"} <= asx and "JHX.AX" not in combined.tickers
    assert not {"NEM.AX", "XYZ.AX", "RMD.AX", "AMC.AX", "NWS.AX", "FPH.AX"} & set(combined.tickers)
    # Mid and small-cap additions: UK FTSE 250, ASX Mid/Small, SGX Small, Spain Continuo, German MDAX/SDAX, Russell 2000
    assert len(benchmark_part("ftse250")) == 92 and {"GRG.L", "JET2.L"} <= set(combined.tickers)
    assert len(benchmark_part("asx_mid_small")) == 101 and {"BRG.AX", "TNE.AX"} <= set(combined.tickers)
    assert len(benchmark_part("st_smallcap")) == 40 and {"C52.SI", "AIY.SI"} <= set(combined.tickers)
    assert len(benchmark_part("spain_continuo")) == 44 and {"CAF.MC", "CIE.MC"} <= set(combined.tickers)
    assert len(benchmark_part("mdax_sdax")) == 56 and {"BOSS.DE", "AIXA.DE"} <= set(combined.tickers)
    assert len(benchmark_part("russell_small")) == 500 and {"BOOT", "FROG"} <= set(combined.tickers)
    assert len(benchmark_part("nasdaq_mid")) == 96 and {"AAON", "CART"} <= set(combined.tickers)
    assert len(benchmark_part("nyse_mid")) == 197 and {"ACM", "ALLY"} <= set(combined.tickers)
    assert len(benchmark_part("msci_eafe_small")) == 500 and {"6532.T", "CLIS.TA"} <= set(combined.tickers)
    assert len(benchmark_part("msci_europe_small")) == 500 and {"PSPN.SW", "STB.OL"} <= set(combined.tickers)
    assert len(benchmark_part("omx_smallcap")) == 54 and {"FAG.ST", "MTRS.ST"} <= set(combined.tickers)
    assert len(benchmark_part("ibovespa")) == 57 and {"ASAI3.SA", "TOTS3.SA"} <= set(combined.tickers)
    assert len(benchmark_part("russell_2000_rest")) == 1439 and {"CRD-A", "BH"} <= set(combined.tickers)
    assert len(benchmark_part("nasdaq_nextgen")) == 13 and {"DKNG", "Z"} <= set(combined.tickers)
    assert len(benchmark_part("offshore_leaders")) == 8 and {"AXS", "JTC.L", "LXMPR.LU"} <= set(combined.tickers)
    assert len(benchmark_part("tsx_composite")) == 134 and {"VNP.TO", "ARE.TO"} <= set(combined.tickers)
    assert len(benchmark_part("ftse_smallcap")) == 90 and {"AEWU.L", "YOU.L"} <= set(combined.tickers)
    assert len(benchmark_part("singapore_smallcap")) == 40 and {"S59.SI", "VC2.SI"} <= set(combined.tickers)
    assert len(benchmark_part("hk_smallcap")) == 180 and {"0013.HK", "0142.HK"} <= set(combined.tickers)
    assert len(benchmark_part("spain_smallcap")) == 45 and {"AI.MC", "ALC.MC"} <= set(combined.tickers)
    assert len(benchmark_part("germany_smallcap")) == 49 and {"1U1.DE", "8TRA.DE"} <= set(combined.tickers)
    japan = set(benchmark_part("japan_100m"))  # Japan >= 100 M USD beyond the other lists, one line per company
    assert len(japan) == 1740 and {"2593.T", "8303.T"} <= japan and not {"25935.T", "9284.T", "7203.T"} & japan
    parts = set().union(*(benchmark_part(n) for n in BENCHMARK_SETS))
    assert set(combined.tickers) == parts
    assert resolve_selection("sp500").label == "custom"  # no longer a run of its own


def test_growth_ignores_buybacks_and_halves_cyclicals(cfg):
    raw = make_raw(cfg, n=3)
    for k, p in enumerate(cfg.fiscal_years):
        raw.loc[1, f"Ordinary Shares Number_{p}"] = 1e8 * 1.05**k  # 5% buybacks a year: per share grows faster
    for p, sign in zip(cfg.fiscal_years, [1, 1, -1, 1, -1, 1], strict=True):
        for concept in ("Operating Income", "EBIT"):
            raw.loc[2, f"{concept}_{p}"] = sign * abs(raw.loc[2, f"{concept}_{p}"])
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Growth_Used"] == pytest.approx(df.loc[1, "Revenue_Growth_USD"])
    assert df.loc[2, "Cyclical"]
    assert df.loc[2, "Growth_Used"] == pytest.approx(min(df.loc[2, "Revenue_Growth_USD"], cfg.growth_cap) * 0.5)
    assert "Revenue_Per_Share_CAGR" not in {c[0] for c in cfg.quality_criteria}


def test_missing_growth_keeps_value_ranked(cfg):
    raw = make_raw(cfg, n=3)
    for p in cfg.annual_years[1:]:
        raw.loc[0, f"Total Revenue_{p}"] = np.nan  # one year of revenue: no growth measurable
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Growth_Used"] == 0 and pd.notna(df.loc[0, "EPV_Yield"])


def test_float_mode_keeps_debt_and_cash_out(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, ["industry", "sector"]] = ["Healthcare Plans", "Healthcare"]
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Float_Mode"] and not df.loc[0, "Financial_Mode"]
    debt = raw.loc[0, "Total Debt_TTM"]
    assert df.loc[0, "enterpriseValue"] == pytest.approx(df.loc[0, "marketCap"] + debt)
    assert pd.notna(df.loc[0, "NetDebt_to_EBIT"]) and df.loc[0, "FCF_Adj_Y0"] == raw.loc[0, "Net Income_Y0"]
    kept, removed = clean_and_filter(df, list(df["Ticker"]))
    assert removed == []  # ranked, not excluded


def test_profit_on_no_capital_gets_the_roic_cap(cfg):
    raw = make_raw(cfg, n=3)
    for p in cfg.fiscal_years:  # buybacks beyond book: equity below minus the debt (BKNG-like)
        raw.loc[0, f"Stockholders Equity_{p}"] = -raw.loc[0, f"Total Debt_{p}"] * 2
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "ROIC_Norm"] == cfg.roic_cap
    assert not apply_dual_ranking(df, cfg).set_index("Ticker").loc[raw.loc[0, "Ticker"], "Insufficient_Data"]


def test_roic_ex_goodwill_takes_acquired_capital_out(cfg):
    raw = make_raw(cfg, n=3)
    for p in cfg.fiscal_years:  # half the equity is goodwill from acquisitions
        raw.loc[0, f"Goodwill And Other Intangible Assets_{p}"] = raw.loc[0, f"Stockholders Equity_{p}"] / 2
        raw.loc[1, f"Goodwill And Other Intangible Assets_{p}"] = raw.loc[1, f"Stockholders Equity_{p}"] * 10
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "ROIC_Ex_Goodwill"] > df.loc[0, "ROIC_Norm"]
    assert df.loc[1, "ROIC_Ex_Goodwill"] == cfg.roic_cap  # goodwill beyond all capital: profit on none left


def test_zero_equity_placeholder_is_read_from_common_equity(cfg):
    """PXED: Yahoo's `Stockholders Equity` was an exact 0 beside a positive common equity, so ROIC was the cap."""
    raw = make_raw(cfg, n=2)
    expected = calculate_metrics(raw, cfg).loc[0, "ROIC_Norm"]
    for p in cfg.fiscal_years:
        raw.loc[0, f"Common Stock Equity_{p}"] = raw.loc[0, f"Stockholders Equity_{p}"]
        raw.loc[0, f"Stockholders Equity_{p}"] = 0.0
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "ROIC_Norm"] == pytest.approx(expected) and expected < cfg.roic_cap
    assert "read as 0" in df.loc[0, "Anomaly_Notes"]


def test_capital_wiped_out_by_losses_is_measured_on_the_plant(cfg):
    """Equity lost to past write-offs is no moat: a business with plant and stock earns on those, not on nothing."""
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Stockholders Equity_{p}"] = -raw.loc[0, f"Total Debt_{p}"] * 2
        raw.loc[0, f"Net PPE_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 2
    df = calculate_metrics(raw, cfg)
    assert 0 < df.loc[0, "ROIC_Norm"] < 20  # EBIT 20% of revenue on plant twice the revenue, after tax


def test_share_classes_with_statements_converted_on_different_days_are_merged(cfg):
    from fq.filters import merge_share_classes

    df = calculate_metrics(make_raw(cfg, n=3), cfg)
    df.loc[1] = df.loc[0]
    df["Fin_Rate"] = [1.12448, 1.123596, 1.0]
    for col in ("Total Revenue_Y0", "Net Income_Y0"):  # the same euros at each fetch day's rate
        df.loc[1, col] = df.loc[0, col] / 1.12448 * 1.123596
    df.loc[[0, 1], "Ticker"] = ["MFEA.MI", "MFEB.MI"]
    df.loc[[0, 1], "shortName"] = ["MFE A", "MFE B"]
    df["sharesOutstanding"] = [4.7e8, 2.3e8, 1e8]
    merged, removed = merge_share_classes(df)
    assert removed == [("MFEB.MI", "second listing or share class of MFEA.MI")]
    df.loc[1, "Net Income_Y0"] *= 1.05  # different profit: two companies
    assert merge_share_classes(df)[1] == []


def test_suspect_checks_the_yield_value_scores(cfg):
    from fq.metrics.flags import _data_suspect

    df = calculate_metrics(make_raw(cfg, n=3), cfg)
    df.loc[0, "Owner_Yield"] = 33.0  # the trend lifts a 21% level beyond the suspect line
    df.loc[1, ["EPV_Yield", "FCFy_EV_Norm"]] = [25.0, 11.7]  # mostly add-backs
    df.loc[2, ["EPV_Yield", "FCFy_EV_Norm", "Owner_Yield"]] = [8.0, 3.0, 9.0]  # a grower at a modest yield
    _data_suspect(df, cfg)
    assert "owner earnings yield" in df.loc[0, "Suspect_Reasons"]
    assert "twice the reported FCF" in df.loc[1, "Suspect_Reasons"]
    assert not df.loc[2, "Data_Suspect"]


def test_share_classes_are_merged_once(cfg):
    from fq.filters import merge_share_classes

    df = calculate_metrics(make_raw(cfg, n=3), cfg)
    df.loc[1, [c for c in df.columns if c not in ("Ticker", "sharesOutstanding")]] = df.loc[0].drop(
        ["Ticker", "sharesOutstanding"], errors="ignore"
    )
    df["sharesOutstanding"] = [2e8, 3e8, 1e8]
    merged, removed = merge_share_classes(df)
    assert list(merged["Ticker"]) == [df.loc[1, "Ticker"], df.loc[2, "Ticker"]]  # more shares outstanding wins
    assert removed == [(df.loc[0, "Ticker"], f"second listing or share class of {df.loc[1, 'Ticker']}")]
    held, _ = merge_share_classes(df, protected={df.loc[0, "Ticker"]})
    assert df.loc[0, "Ticker"] in set(held["Ticker"]) and df.loc[1, "Ticker"] not in set(held["Ticker"])
    swire = df.assign(Ticker=["0087.HK", "0019.HK", df.loc[2, "Ticker"]])
    assert len(merge_share_classes(swire)[0]) == 3  # unequal-rights classes are never merged


def test_dual_listings_keep_the_home_listing(cfg):
    from fq.filters import merge_share_classes

    df = calculate_metrics(make_raw(cfg, n=3), cfg)
    df.loc[1] = df.loc[0]
    df.loc[[0, 1], "Ticker"] = ["IAG.L", "IAG.MC"]
    # Yahoo truncates the names differently; the same share count confirms one company.
    df.loc[[0, 1], "shortName"] = ["INTERNATIONAL CONSOLIDATED AIRL", "INTL CONSOLIDATED AIRLINES GROU"]
    df.loc[[0, 1], "currency"] = ["GBp", "EUR"]
    df["financialCurrency"] = "EUR"
    df["sharesOutstanding"] = [4.36e9, 4.36e9, 1e8]
    merged, removed = merge_share_classes(df)
    assert removed == [("IAG.L", "second listing or share class of IAG.MC")]  # quoted in its reporting currency
    df["financialCurrency"] = "GBP"  # pence count as pounds
    assert merge_share_classes(df)[1] == [("IAG.MC", "second listing or share class of IAG.L")]
    df.loc[1, "sharesOutstanding"] = 2e9  # different names and share counts: not confirmed
    assert merge_share_classes(df)[1] == []


def test_kept_excluded_holdings_leave_top_lists(cfg):
    df, c = _toy(cfg, a=[10, 5, 1], b=[10, 5, 1])
    df["Industry_Excluded_Kept"] = [True, False, False]
    ranked = apply_dual_ranking(df, c)
    assert "T0" in set(ranked["Ticker"]) and "T0" not in set(top_n(ranked, "Value_Rank", 3)["Ticker"])


def test_value_quality_correlation_is_reported(cfg):
    from fq.ranking import value_quality_correlation

    df, c = _toy(cfg, a=[10, 5, 1, 7], b=[10, 5, 1, 7])
    assert value_quality_correlation(apply_dual_ranking(df, c)) == pytest.approx(1.0)
    assert np.isnan(value_quality_correlation(df))


def test_value_quality_correlation_excludes_unprofitable(cfg):
    from fq.ranking import value_quality_correlation

    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D"],
            "Business_Model": ["operating"] * 4,
            "Value_Score": [90.0, 80.0, 70.0, 10.0],
            "Quality_Score": [30.0, 40.0, 50.0, 10.0],  # A, B, C are negative correlation (-1.0); D pulls it up
            "FCFy_Norm": [5.0, 4.0, 3.0, -10.0],
        }
    )
    assert value_quality_correlation(df, profitable_only=True) == pytest.approx(-1.0)
    assert value_quality_correlation(df, profitable_only=False) > -1.0


def test_value_quality_correlation_leaves_out_financials():
    from fq.ranking import value_quality_correlation

    df = pd.DataFrame(
        {
            "Ticker": list("ABCDEF"),
            "Business_Model": ["operating"] * 3 + ["bank"] * 3,
            "Financial_Mode": [False] * 3 + [True] * 3,
            "Value_Score": [90.0, 80.0, 70.0, 95.0, 50.0, 5.0],
            "Quality_Score": [30.0, 40.0, 50.0, 96.0, 51.0, 6.0],  # banks: both rest on ROE, move together
            "FCFy_Norm": [5.0] * 6,
        }
    )
    assert value_quality_correlation(df) == pytest.approx(-1.0)
    assert value_quality_correlation(df, operating_only=False) > 0


def test_cyclical_industry_gets_mid_cycle_value(cfg):
    from conftest import make_raw
    from fq.metrics import calculate_metrics

    raw = make_raw(cfg, n=2, seed=4)
    raw.loc[1] = raw.loc[0]
    raw.loc[1, ["Ticker", "industry"]] = ["HOME", "Residential Construction"]  # same figures, cyclical industry
    out = calculate_metrics(raw, cfg).set_index("Ticker")
    steady, home = out.loc["T0"], out.loc["HOME"]
    assert not steady["Cyclical"] and home["Cyclical"] and home["Cyclical_Industry"]
    assert home["Growth_Used"] == pytest.approx(steady["Growth_Used"] * cfg.cyclical_growth_factor)
    assert home["FCFy_Norm"] < steady["FCFy_Norm"]  # 3-year average revenue, not the latest (peak?) year


def test_statements_that_suggest_another_model_are_flagged_not_applied(cfg):
    raw = make_raw(cfg, n=3)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Assets_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 3
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 2  # 67% of assets
        raw.loc[1, f"Total Assets_{p}"] = raw.loc[1, f"Total Revenue_{p}"] * 3
        raw.loc[1, f"Pretax Income_{p}"] = raw.loc[1, f"Total Revenue_{p}"] * 0.6  # 3x operating income
    raw.loc[2, "Ticker"] = "V"  # same as row 0, but its model is set by hand: never flagged
    for p in cfg.fiscal_years:
        raw.loc[2, f"Total Assets_{p}"] = raw.loc[0, f"Total Assets_{p}"]
        raw.loc[2, f"Cash And Cash Equivalents_{p}"] = raw.loc[0, f"Cash And Cash Equivalents_{p}"]
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Business_Model"] == "client_cash" and df.loc[0, "Archetype_Source"] == "signature:client_cash"
    assert df.loc[1, "Model_Review"].startswith("holding")
    assert pd.isna(df.loc[2, "Model_Review"])


def test_investment_trusts_are_flagged_as_holdings(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Assets_{p}"] = raw.loc[0, f"Stockholders Equity_{p}"] / 0.9  # 90% equity-funded
        raw.loc[0, f"Net PPE_{p}"] = 0.0
    raw.loc[0, "currentPrice"] = raw.loc[0, "Stockholders Equity_TTM"] / raw.loc[0, "Ordinary Shares Number_TTM"]
    df = calculate_metrics(raw, cfg)  # priced at book value
    assert "holding" in df.loc[0, "Model_Review"] and pd.isna(df.loc[1, "Model_Review"])


def test_financing_is_charged_at_todays_debt(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Debt_{p}"] = 1e9
    raw.loc[0, "Total Debt_TTM"] = 3e9  # debt-funded buyback after the last annual report
    raw.loc[0, "Interest Expense_TTM"] = 1e8
    df = calculate_metrics(raw, cfg)
    tax = df.loc[0, "Tax_Rate_TTM"]
    # TTM ends half a year after Y0: a year earlier the debt was 1bn, so 1e8 was paid on 2bn on average.
    today = 1e8 * 3e9 / 2e9 * (1 - tax)
    assert df.loc[0, "Net_Financing_Yield"] == pytest.approx(today * 100 / df.loc[0, "FCF_Base"])
    assert df.loc[0, "FCFy_Norm"] == pytest.approx(df.loc[0, "FCFy_PreFin_Norm"] - df.loc[0, "Net_Financing_Yield"])
    y2 = raw.loc[0, "Interest Expense_Y2"] * (1 - df.loc[0, "Tax_Rate_Y2"])
    assert df.loc[0, "FCF_PreFin_Y2"] == pytest.approx(df.loc[0, "FCF_Adj_Y2"] + y2)  # past interest taken out


def test_leverage_adjustment_is_shown_not_ranked(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Debt_{p}"] = 0.0
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = 2e9
        raw.loc[0, f"Interest Income_{p}"] = 8e7  # 4% on the cash
    df = calculate_metrics(raw, cfg)
    tax = df.loc[0, "Tax_Rate_TTM"]
    gearing = -2e9 / df.loc[0, "Equity_Value"]
    assert df.loc[0, "Leverage_Adjustment"] == pytest.approx((cfg.asset_required_return - 4 * (1 - tax)) * gearing)
    assert df.loc[1, "Net_Debt"] > 0 and df.loc[1, "Leverage_Adjustment"] > 0
    assert not {c[0] for c in cfg.value_criteria} & {"Leverage_Adjustment", "NetDebt_to_EBIT"}
    # Interest on cash held steady: taken out of past FCF, given back at today's rate and balance.
    assert df.loc[0, "Net_Financing_Yield"] < 0


def test_interest_received_under_investing_is_not_taken_out_of_fcf(cfg):
    raw = make_raw(cfg, n=1)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Interest Income_{p}"] = 5e7
    plain = calculate_metrics(raw, cfg)
    raw.loc[0, "Interest Received Cfi_Y1"] = 5e7  # IFRS: never in operating cash flow
    df = calculate_metrics(raw, cfg)
    expense = raw.loc[0, "Interest Expense_Y1"] * (1 - df.loc[0, "Tax_Rate_Y1"])
    assert df.loc[0, "FCF_PreFin_Y1"] == pytest.approx(df.loc[0, "FCF_Adj_Y1"] + expense)
    assert plain.loc[0, "FCF_PreFin_Y1"] < df.loc[0, "FCF_PreFin_Y1"]


def test_one_offs_are_not_earning_power(cfg):
    raw = make_raw(cfg, n=2)
    plain = calculate_metrics(raw, cfg)
    gain = 2e8  # a disposal gain in the latest year
    raw.loc[0, "Pretax Income_Y0"] += gain
    raw.loc[0, "Net Income_Y0"] += gain * 0.8
    raw.loc[0, "Total Unusual Items_Y0"] = gain
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Pretax_Yield_Avg"] == pytest.approx(plain.loc[0, "Pretax_Yield_Avg"])
    tax = df.loc[0, "Tax_Rate_Y0"]
    assert df.loc[0, "Net_Income_Clean_Y0"] == pytest.approx(raw.loc[0, "Net Income_Y0"] - gain * (1 - tax))


def test_payout_is_not_value(cfg):
    raw = make_raw(cfg, n=6)
    plain = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Cash Dividends Paid_{p}"] = -5e8  # pays out more: no cheaper for it
    ranked = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    assert "DirectCashtoInvestors_Yield_Avg" not in {c[0] for c in cfg.value_criteria}
    by_ticker = lambda d: d.set_index("Ticker")["Value_Score"]  # noqa: E731
    pd.testing.assert_series_equal(by_ticker(ranked).sort_index(), by_ticker(plain).sort_index())


def test_earnings_power_charges_maintenance_capex_only(cfg):
    raw = _builder(cfg)
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Organic_Capex"]
    assert df.loc[0, "EPV_Yield"] == pytest.approx(df.loc[0, "FCFy_EV_Norm"])  # growth capex already added back
    plain = calculate_metrics(raw, replace(cfg, organic_capex_models=()))
    assert df.loc[0, "EPV_Yield"] > plain.loc[0, "EPV_Yield"]


def _working_capital(cfg, growth: float, payables: float, moved: float):
    """Two operating companies growing at `growth`; company 0 holds non-cash current assets of 50% of revenue,
    current liabilities of `payables` x revenue and reports `moved` x revenue as its yearly working capital
    change (cash flow sign: negative = tied up)."""
    raw = make_raw(cfg, n=2)
    for k, p in enumerate(reversed(cfg.fiscal_years)):
        rev = 1e9 * (1 + growth) ** k
        for concept, share in (("Net Income", 0.15), ("Operating Income", 0.2), ("EBIT", 0.2),
                               ("Pretax Income", 0.19), ("Tax Provision", 0.04), ("Interest Expense", 0.01),
                               ("Free Cash Flow", 0.12), ("Operating Cash Flow", 0.16), ("Capital Expenditure", -0.04),
                               ("Stock Based Compensation", 0.01), ("Stockholders Equity", 0.8), ("Total Debt", 0.3),
                               ("Cash And Cash Equivalents", 0.2)):  # fmt: skip
            raw.loc[0, f"{concept}_{p}"] = rev * share
        raw.loc[0, f"Total Revenue_{p}"] = rev
        raw.loc[0, f"Current Assets_{p}"] = rev * 0.7  # 0.2 of it is cash
        raw.loc[0, f"Current Liabilities_{p}"] = rev * payables
    plain = calculate_metrics(raw, cfg)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Change In Working Capital_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * moved
    return raw, plain, calculate_metrics(raw, cfg)


def test_growth_working_capital_is_owner_earnings(cfg):
    raw, plain, df = _working_capital(cfg, growth=0.10, payables=0.2, moved=-0.05)
    rev0, rev1 = raw.loc[0, "Total Revenue_Y0"], raw.loc[0, "Total Revenue_Y1"]
    # Intensity (0.5 - 0.2) x the revenue added beyond inflation, within what the cash flow moved.
    us = df.loc[0, "Inflation_US_Y0"]
    assert df.loc[0, "Growth_WC_Owner_Y0"] == pytest.approx(0.3 * (rev0 / rev1 / (1 + us) - 1) * rev0)
    assert df.loc[0, "EPV_Yield"] > plain.loc[0, "EPV_Yield"]  # no row before: no adjustment
    assert df.loc[0, "FCFy_Norm"] == pytest.approx(plain.loc[0, "FCFy_Norm"])  # cash as it moved
    assert (df.loc[1, [f"Growth_WC_Owner_{p}" for p in cfg.fiscal_years]] == 0).all()  # no current rows

    _, _, small = _working_capital(cfg, growth=0.10, payables=0.2, moved=-0.001)
    assert small.loc[0, "Growth_WC_Owner_Y0"] == pytest.approx(0.001 * rev0)  # never beyond the cash flow
    _, _, flat = _working_capital(cfg, growth=0.02, payables=0.2, moved=-0.05)
    assert flat.loc[0, "Growth_WC_Owner_Y0"] == 0  # below inflation: price, not volume


def test_supplier_funded_growth_is_not_earning_power(cfg):
    # Costco-like: current liabilities above non-cash current assets, so growth frees cash.
    _, plain, df = _working_capital(cfg, growth=0.10, payables=0.9, moved=0.05)
    assert df.loc[0, "Growth_WC_Owner_Y0"] < 0
    assert df.loc[0, "EPV_Yield"] < plain.loc[0, "EPV_Yield"]


def test_financials_keep_book_against_inflation(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "industry"] = "Banks - Regional"
    df = calculate_metrics(raw, cfg)
    none = calculate_metrics(raw, replace(cfg, inflation=0.0))
    book = raw.loc[0, "Stockholders Equity_Y0"]
    assert df.loc[0, "Retention_Owner_Y0"] == pytest.approx(cfg.inflation * book)
    assert df.loc[0, "Owner_Earnings_Y0"] == pytest.approx(none.loc[0, "Owner_Earnings_Y0"] - cfg.inflation * book)
    assert df.loc[0, "EPV_Yield"] < none.loc[0, "EPV_Yield"]
    assert (df.loc[1, [f"Retention_Owner_{p}" for p in cfg.fiscal_years]] == 0).all()  # operating


def test_interest_no_longer_reported_is_not_restated(cfg):
    raw = make_raw(cfg, n=1)
    for p in ("TTM", cfg.latest, cfg.prior):  # AAPL: interest no longer broken out
        raw.loc[0, f"Interest Expense_{p}"] = np.nan
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Net_Financing_Yield"] == 0
    assert df.loc[0, "FCF_PreFin_Y3"] == pytest.approx(df.loc[0, "FCF_Adj_Y3"])  # old interest not given back


def test_ifrs_lease_interest_leaves_nopat(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[1, "Ticker"] = "T1.MC"  # IFRS
    for p in cfg.fiscal_years:
        for i in (0, 1):
            raw.loc[i, f"Capital Lease Obligations_{p}"] = raw.loc[i, f"Total Debt_{p}"] * 0.5
            raw.loc[i, f"Current Capital Lease Obligation_{p}"] = raw.loc[i, f"Total Debt_{p}"] * 0.1
    df = calculate_metrics(raw, cfg)
    debt, rate = raw.loc[1, "Total Debt_Y0"], raw.loc[1, "Interest Expense_Y0"] / raw.loc[1, "Total Debt_Y0"]
    lease_interest = debt * 0.5 * rate
    tax = df.loc[1, "Tax_Rate_Y0"]
    assert df.loc[1, "NOPAT_Y0"] == pytest.approx((df.loc[1, "EBIT_Y0"] - lease_interest) * (1 - tax))
    assert df.loc[0, "NOPAT_Y0"] == pytest.approx(df.loc[0, "EBIT_Y0"] * (1 - df.loc[0, "Tax_Rate_Y0"]))  # US GAAP


def test_book_growth_leaves_out_unrealized_bond_losses(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, ["industry", "sector"]] = ["Banks - Regional", "Financial Services"]
    plain = calculate_metrics(raw, cfg)
    raw.loc[0, f"Stockholders Equity_{cfg.annual_years[2]}"] *= 0.8  # rates rose: the bond book marked down
    raw.loc[0, f"Gains Losses Not Affecting Retained Earnings_{cfg.annual_years[2]}"] = (
        -0.2 * raw.loc[0, f"Stockholders Equity_{cfg.annual_years[2]}"] / 0.8
    )
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Book_Growth_USD"] == pytest.approx(plain.loc[0, "Book_Growth_USD"])


def test_growth_metric_rewards_steady_and_accelerating_growth(cfg):
    raw = make_raw(cfg, n=3)
    years = list(reversed(cfg.annual_years))  # oldest first
    for k, p in enumerate(years):
        raw.loc[0, f"Total Revenue_{p}"] = 1e9 * 1.10**k  # 10% every year
        raw.loc[1, f"Total Revenue_{p}"] = 1e9 * 1.10**k * (1.25 if k % 2 else 0.85)  # same trend, bumpy
    for k, p in enumerate(years):  # row 2: growth speeds up every year, 5% to 8%
        raw.loc[2, f"Total Revenue_{p}"] = 1e9 * np.prod([1 + (4 + i) / 100 for i in range(1, k + 1)])
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Revenue_Growth_Steady"] == pytest.approx(10.0)  # rising every year: the latest year
    assert "rising every year" in df.loc[0, "Revenue_Growth_Basis"]
    assert 0 < df.loc[1, "Revenue_Growth_Steady"] < df.loc[0, "Revenue_Growth_Steady"]  # bumps: trend x R²
    assert "R²" in df.loc[1, "Revenue_Growth_Basis"]
    assert df.loc[2, "Revenue_Growth_Steady"] == pytest.approx(8.0 + 1.0)  # latest 8% + its 1 pp step-up


def test_value_is_neutral_to_debt_and_cash_at_the_same_price_for_the_operations(cfg):
    """One business priced at the same value for its operations: unlevered (0), with debt (1), with spare cash (2)."""
    cfg = replace(cfg, operating_cash_pct=0.0)
    raw = make_raw(cfg, n=3)
    claim, rate = 2e9, 0.04
    shares = raw.loc[0, "Ordinary Shares Number_TTM"]
    for p in cfg.fiscal_years:
        for col in [c for c in raw.columns if c.endswith(f"_{p}")]:
            raw.loc[[1, 2], col] = raw.loc[0, col]
        tax = raw.loc[0, f"Tax Provision_{p}"] / raw.loc[0, f"Pretax Income_{p}"]
        raw.loc[:, f"Interest Expense_{p}"] = 0.0
        raw.loc[:, f"Total Debt_{p}"] = 0.0
        raw.loc[:, f"Cash And Cash Equivalents_{p}"] = 0.0
        interest = claim * rate
        raw.loc[1, f"Total Debt_{p}"] = claim
        raw.loc[1, f"Interest Expense_{p}"] = interest
        raw.loc[2, f"Cash And Cash Equivalents_{p}"] = claim
        raw.loc[2, f"Interest Income_{p}"] = interest
        for i, sign in ((1, -1), (2, 1)):  # interest in operating cash flow and pre-tax income, same tax rate
            raw.loc[i, f"Pretax Income_{p}"] += sign * interest
            raw.loc[i, f"Tax Provision_{p}"] = tax * raw.loc[i, f"Pretax Income_{p}"]
            for col in ("Net Income", "Free Cash Flow", "Operating Cash Flow"):
                raw.loc[i, f"{col}_{p}"] += sign * interest * (1 - tax)
    raw.loc[1, "currentPrice"] = raw.loc[0, "currentPrice"] - claim / shares
    raw.loc[2, "currentPrice"] = raw.loc[0, "currentPrice"] + claim / shares
    df = calculate_metrics(raw, cfg)
    assert df["Value_Base"].tolist() == pytest.approx([df.loc[0, "marketCap"]] * 3)
    for col in ("EPV_Yield", "Pretax_Yield_Avg"):
        assert df[col].tolist() == pytest.approx([df.loc[0, col]] * 3, rel=1e-6), col
    assert df.loc[1, "FCFy_Norm"] > df.loc[0, "FCFy_Norm"] > df.loc[2, "FCFy_Norm"]  # per share, display only
    assert df.loc[1, "NetDebt_to_EBIT"] > df.loc[0, "NetDebt_to_EBIT"] == df.loc[2, "NetDebt_to_EBIT"] == 0


def test_debt_and_cash_count_at_face_only_where_their_cost_left_the_flows(cfg):
    raw = make_raw(cfg, n=4)
    for p in cfg.fiscal_years:
        raw.loc[:, f"Total Debt_{p}"] = 2e9
        raw.loc[:, f"Interest Expense_{p}"] = 8e7  # 4%: an ordinary borrower
        raw.loc[1, f"Interest Expense_{p}"] = 1.5e7  # 0.75%: most of the cost sits elsewhere (captive lender)
        raw.loc[2, f"Interest Expense_{p}"] = np.nan  # no interest row: all of it is still in the flows (lessor)
        raw.loc[:, f"Cash And Cash Equivalents_{p}"] = 1e9
        raw.loc[3, f"Interest Income_{p}"] = 4e7  # only row 3's cash interest is taken out of the flows
    df = calculate_metrics(raw, replace(cfg, operating_cash_pct=0.0))
    assert df.loc[0, "Debt_At_Face"] == pytest.approx(2e9)
    assert df.loc[1, "Debt_At_Face"] == pytest.approx(2e9 * 0.75 / cfg.debt_cost_floor)
    assert df.loc[2, "Debt_At_Face"] == 0
    assert df.loc[0, "Cash_At_Face"] == 0 and df.loc[3, "Cash_At_Face"] == pytest.approx(1e9)
    assert df.loc[3, "Net_Claims"] == pytest.approx(2e9 - 1e9)


def test_lenders_and_insurers_filed_elsewhere_are_flagged(cfg):
    raw = make_raw(cfg, n=3)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Debt_{p}"] = 2.5 * raw.loc[0, "marketCap"]  # a captive lender: the debt is the business
        raw.loc[0, f"Interest Expense_{p}"] = np.nan  # its cost sits in cost of sales
        raw.loc[1, f"Net Policyholder Benefits And Claims_{p}"] = 5e8
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Debt_Cost_Hidden"] and "captive lender" in df.loc[0, "Suspect_Reasons"]
    assert df.loc[0, "Model_Review"].startswith("bank or asset_heavy")
    assert df.loc[1, "Business_Model"] == "insurer" and df.loc[1, "Archetype_Source"] == "signature:claims"
    assert pd.isna(df.loc[2, "Model_Review"])


def test_spare_cash_does_not_dilute_gross_profitability(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[:, f"Total Assets_{p}"] = 4e9
        raw.loc[:, f"Gross Profit_{p}"] = 8e8
    raw.loc[:, [f"Cash And Cash Equivalents_{p}" for p in cfg.fiscal_years]] = 0.0
    raw.loc[1, [f"Cash And Cash Equivalents_{p}" for p in cfg.fiscal_years]] = 1e9
    raw.loc[1, [f"Total Assets_{p}" for p in cfg.fiscal_years]] = 5e9  # the same business plus 1bn of cash
    df = calculate_metrics(raw, replace(cfg, operating_cash_pct=0.0))
    assert df.loc[1, "Gross_Profitability_Avg"] == pytest.approx(df.loc[0, "Gross_Profitability_Avg"])


def test_roic_spread_is_measured_around_its_trend_against_its_level(cfg):
    raw = make_raw(cfg, n=2)
    df = calculate_metrics(raw, cfg)
    roic = pd.DataFrame({p: df[f"ROIC_{p}"] for p in cfg.fiscal_years})
    level = roic.mean(axis=1).abs().clip(lower=cfg.roic_variability_floor)
    _, spread, _ = _roic_fit(df, cfg)
    assert df["ROIC_Variability"].tolist() == pytest.approx((spread / level).tolist())
    assert (spread <= df["ROIC_StdDev"] + 1e-9).all()  # the trend line explains part of the spread
    assert "ROIC_Variability" in {c[0] for c in cfg.shown_percentiles}
    assert "operatingMargins_Avg" not in {c[0] for c in cfg.quality_criteria}  # shown, not scored


def test_a_steady_climb_in_roic_is_a_trend_not_unsteadiness(cfg):
    years = cfg.annual_years[:4]
    climbing = {f"ROIC_{p}": 25.0 - 5.0 * i for i, p in enumerate(years)}  # 10 -> 25, a straight line
    wobbling = {f"ROIC_{p}": 15.0 + (5.0 if i % 2 else -5.0) for i, p in enumerate(years)}
    slope, spread, r2 = _roic_fit(pd.DataFrame([climbing, wobbling]), cfg)
    assert slope[0] == pytest.approx(5.0) and spread[0] == pytest.approx(0.0, abs=1e-9)
    assert spread[1] > 3.0
    assert r2[0] == pytest.approx(1.0) and r2[1] < 0.5  # the wobble is no trend to project


def test_roic_trend_needs_four_points_and_counts_by_its_r2(cfg):
    years = cfg.annual_years[:4]
    three = {f"ROIC_{p}": 20.0 - 5.0 * i for i, p in enumerate(years[:3])}  # 10 -> 20 on three points
    four = {f"ROIC_{p}": 25.0 - 5.0 * i for i, p in enumerate(years)}
    slope, _, r2 = _roic_fit(pd.DataFrame([three, four]), cfg)
    assert slope[0] == pytest.approx(5.0)  # shown as measured
    assert r2[0] == 0.0 and r2[1] == pytest.approx(1.0)  # but three points earn no trend


def _quality_frame(**overrides):
    base = {
        "ROIC_Norm": [20.0],
        "ROIC_Slope": [-3.0],
        "ROIC_Fit_R2": [1.0],
        "ROIC_Variability": [0.0],
        "ROIC_Norm_Age": [0.0],
        "ROIC_Norm_Capped": [False],
        "ROIC_Norm_Declining": [False],
        "Cyclical": [False],
        "Financial_Mode": [False],
        "Business_Model": ["operating"],
        "NetDebt_to_EBIT": [0.0],
        "Equity_to_Assets": [np.nan],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_cyclical_roic_trend_is_not_charged_unless_it_fell_every_year(cfg):
    # Mid-cycle level already: a fall since the peak is the cycle, not news (Oct 28, 2026).
    cyc = _quality_frame(Cyclical=True)
    _return_quality(cyc, cfg)
    assert cyc.loc[0, "Return_Quality"] == pytest.approx(20.0)
    falling = _quality_frame(Cyclical=True, ROIC_Norm_Declining=True)
    _return_quality(falling, cfg)
    assert falling.loc[0, "Return_Quality"] == pytest.approx(20.0 - 3 * 3.0)
    rising = _quality_frame(Cyclical=True, ROIC_Slope=3.0)
    _return_quality(rising, cfg)
    assert rising.loc[0, "Return_Quality"] == pytest.approx(20.0)


def test_roic_trend_runs_from_the_levels_date_and_by_its_r2(cfg):
    old = _quality_frame(ROIC_Slope=1.0, ROIC_Norm_Age=1.5)  # a median ~1.5 years back
    _return_quality(old, cfg)
    assert old.loc[0, "Return_Quality"] == pytest.approx(20.0 + (3 + 1.5) * 1.0)
    noisy = _quality_frame(ROIC_Slope=1.0, ROIC_Fit_R2=0.5)
    _return_quality(noisy, cfg)
    assert noisy.loc[0, "Return_Quality"] == pytest.approx(20.0 + 3 * 1.0 * 0.5)


def test_an_over_earner_capped_at_its_median_gets_no_rise_back(cfg):
    capped = _quality_frame(ROIC_Slope=4.7, ROIC_Norm_Capped=True)
    _return_quality(capped, cfg)
    assert capped.loc[0, "Return_Quality"] == pytest.approx(20.0)
    falling = _quality_frame(ROIC_Slope=-1.0, ROIC_Norm_Capped=True)
    _return_quality(falling, cfg)
    assert falling.loc[0, "Return_Quality"] == pytest.approx(17.0)


def test_unknown_roic_steadiness_is_not_perfect_steadiness(cfg):
    df = pd.DataFrame(
        {
            "ROIC_Norm": [20.0, 20.0],
            "ROIC_Slope": [0.0, 0.0],
            "ROIC_Variability": [0.0, np.nan],  # perfectly steady record vs under 3 years of ROIC
            "Cyclical": [False, False],
            "Financial_Mode": [False, False],
            "NetDebt_to_EBIT": [0.0, 0.0],
            "Equity_to_Assets": [np.nan, np.nan],
        }
    )
    out = df.assign(Business_Model="operating")
    _return_quality(out, cfg)
    assert out.loc[1, "Return_Quality"] == pytest.approx(20.0 / (1 + cfg.missing_roic_variability))
    assert out.loc[1, "Return_Quality"] < out.loc[0, "Return_Quality"]


def test_bank_roe_is_judged_at_a_standard_leverage(cfg):
    # 15% ROE on 5% equity (ROA 0.75%) against 12% ROE on 10% equity (ROA 1.2%): the second is the better bank.
    df = pd.DataFrame(
        {
            "ROIC_Norm": [15.0, 12.0],
            "ROIC_Slope": [0.0, 0.0],
            "ROIC_Variability": [0.0, 0.0],
            "Cyclical": [False, False],
            "Financial_Mode": [True, True],
            "Business_Model": ["bank", "bank"],
            "NetDebt_to_EBIT": [np.nan, np.nan],
            "Equity_to_Assets": [5.0, 10.0],
        }
    )
    _return_quality(df, cfg)
    assert df["Return_Quality"].tolist() == pytest.approx([15.0 * 5 / 8, 12.0 * 10 / 8])
    assert df.loc[1, "Return_Quality"] > df.loc[0, "Return_Quality"]


def test_a_loss_with_negative_fcf_is_the_worst_conversion(cfg):
    raw = make_raw(cfg, n=1)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Net Income_{p}"] = -1e8
        raw.loc[0, f"Free Cash Flow_{p}"] = -5e7
    assert calculate_metrics(raw, cfg).loc[0, "FCF_Conversion"] == 0


def test_pretax_yield_falling_every_year_is_taken_at_its_latest_level(cfg):
    raw = make_raw(cfg, n=2)
    for k, p in enumerate(reversed(cfg.fiscal_years)):  # oldest first: pre-tax income falls every period
        raw.loc[0, f"Pretax Income_{p}"] = 4e8 * 0.8**k
        raw.loc[0, f"Tax Provision_{p}"] = 0.2 * raw.loc[0, f"Pretax Income_{p}"]
        raw.loc[0, f"Interest Expense_{p}"] = 0.0
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Pretax_Yield_Avg"] == pytest.approx(
        df.loc[0, "Pretax_PreFin_Yield_TTM"] * df.loc[0, "Equity_Value"] / df.loc[0, "Value_Base"]
    )


def test_aftertax_yield_is_pretax_after_the_period_tax(cfg):
    df = calculate_metrics(make_raw(cfg, n=4), cfg)
    for p in cfg.fiscal_years:
        expected = df[f"Pretax_PreFin_Yield_{p}"] * (1 - df[f"Tax_Rate_{p}"])
        assert df[f"Aftertax_PreFin_Yield_{p}"].to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)
    assert (df["Aftertax_Yield_Avg"] < df["Pretax_Yield_Avg"]).all()


def test_per_share_compounding_is_the_trend_shrunk_for_bumps(cfg):
    from fq.metrics.valuation import _steady_growth

    slots = [4, 3, 2, 1, 0]  # oldest first in time: slot 4 is the oldest year
    steady = [100 * 1.1**k for k in range(5)]
    bumpy = [100, 125, 105, 140, 146]
    frame = pd.DataFrame([dict(zip(slots, steady, strict=True)), dict(zip(slots, bumpy, strict=True))])
    growth, _ = _steady_growth(frame, trend_only=True)
    assert growth[0] == pytest.approx(10.0)  # a straight line: R² = 1
    x, y = -np.array(slots, dtype=float), np.log(bumpy)
    slope = np.polyfit(x, y, 1)[0]
    assert 0 < growth[1] < (np.exp(slope) - 1) * 100  # bumps shrink it
    df = calculate_metrics(make_raw(cfg, n=4), cfg)
    assert df["Per_Share_Compounding"].notna().any()
    assert df.loc[df["Financial_Mode"], "Per_Share_Compounding"].isna().all()


def test_adjusted_compounding_credits_no_chance_line(cfg):
    """`Owner_Earnings_Compounding` weighs its slope by adjusted R²: noise around a flat line through four points
    is not credited a third of its slope; a smooth trend keeps nearly all of it."""
    from fq.metrics.valuation import _steady_growth

    slots = [3, 2, 1, 0]
    noisy = [100, 140, 95, 120]
    smooth = [100, 109, 121, 133]
    frame = pd.DataFrame([dict(zip(slots, noisy, strict=True)), dict(zip(slots, smooth, strict=True))])
    plain, _ = _steady_growth(frame, trend_only=True)
    adjusted, _ = _steady_growth(frame, trend_only=True, adjusted=True)
    assert plain[0] > 0 and adjusted[0] == pytest.approx(0.0)
    assert adjusted[1] == pytest.approx(plain[1], rel=0.02)


def test_market_criteria_compare_every_model_on_one_scale(cfg):
    df, c = _toy(cfg, a=[5, 5, 1, 9, 3], b=[5, 5, 1, 9, 3])
    c = replace(c, financial_quality_criteria=c.quality_criteria, quality_market_criteria=("b",))
    df["Business_Model"] = ["bank", "operating", "operating", "operating", "bank"]
    by = apply_dual_ranking(df, c).set_index("Ticker")
    assert by.at["T0", "Pct_b"] == pytest.approx(by.at["T1", "Pct_b"])  # same value, same percentile
    assert by.at["T0", "Pct_a"] == pytest.approx(100.0)  # a stays within peers: best of the two banks


def test_earnings_power_alone_is_enough_coverage_and_roic_leads_quality(cfg):
    value = dict((c[0], c[2]) for c in cfg.value_criteria)
    quality = dict((c[0], c[2]) for c in cfg.quality_criteria)
    assert value["Owner_Yield"] / sum(value.values()) >= cfg.min_coverage
    assert max(quality, key=quality.get) == "Return_Quality"
    # Resilience is asked of every company, banks included, on one scale.
    assert "Revenue_Worst_Change" in {c[0] for c in cfg.shown_percentiles}


def test_a_profit_collapse_caps_earnings_power_at_the_ttm(cfg):
    raw = make_raw(cfg, n=3)
    for concept in ("Pretax Income", "Net Income", "Operating Income", "EBIT"):
        raw.loc[0, f"{concept}_TTM"] = raw.loc[0, f"{concept}_{cfg.latest}"] * 0.5  # profit halved over the TTM
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Earnings_Collapse"] and not df.loc[1:, "Earnings_Collapse"].any()
    ttm = df.loc[0, "Aftertax_PreFin_Yield_TTM"] * df.loc[0, "Equity_Value"] / df.loc[0, "Value_Base"]
    assert df.loc[0, "EPV_Yield"] <= ttm + 1e-9
    assert df.loc[0, "Aftertax_Yield_Avg"] == pytest.approx(ttm)  # the latest, not the average of better years


def _quarters(raw, cfg, i, first, second):
    """Two stored quarters for row i, one quarter apart, ending at the TTM end."""
    end = pd.Timestamp(raw.loc[i, "PeriodEnd_TTM"])
    raw.loc[i, "PeriodEnd_Q0"] = end.strftime("%Y-%m-%d")
    raw.loc[i, "PeriodEnd_Q1"] = (end - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    for q, value in (("Q0", first), ("Q1", second)):
        raw.loc[i, f"Pretax Income_{q}"] = value
        raw.loc[i, f"Total Unusual Items_{q}"] = 0.0
        raw.loc[i, f"Interest Expense_{q}"] = 0.0


def test_a_lasting_fall_is_capped_at_the_half_year_run_rate_but_one_bad_quarter_is_not(cfg):
    raw = make_raw(cfg, n=3)
    for i in (0, 1):
        for concept in ("Pretax Income", "Net Income", "Operating Income", "EBIT"):
            raw.loc[i, f"{concept}_TTM"] = raw.loc[i, f"{concept}_{cfg.latest}"] * 0.6
        raw.loc[i, "Interest Expense_TTM"] = raw.loc[i, f"Interest Expense_{cfg.latest}"] = 0.0
    quarter = raw.loc[0, f"Pretax Income_{cfg.latest}"] / 4
    _quarters(raw, cfg, 0, quarter * 0.3, quarter * 0.5)  # every quarter lower (QFIN)
    _quarters(raw, cfg, 1, quarter * -1.0, quarter * 1.0)  # one quarter of charges, the one before normal (TEF)
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Earnings_Collapse"] and not df.loc[1, "Earnings_Collapse"]
    expected = (quarter * 0.3 + quarter * 0.5) * 2 * 100 / df.loc[0, "Equity_Value"] * (1 - df.loc[0, "Tax_Rate_TTM"])
    assert df.loc[0, "Earnings_Run_Rate"] == pytest.approx(expected)
    on_base = expected * df.loc[0, "Equity_Value"] / df.loc[0, "Value_Base"]
    assert df.loc[0, "EPV_Yield"] <= on_base + 1e-9 and df.loc[0, "Aftertax_Yield_Avg"] == pytest.approx(on_base)


def test_financial_compounding_follows_book_value_per_share(cfg):
    raw = make_raw(cfg, n=3)
    raw.loc[0, ["industry", "sector"]] = ["Banks - Regional", "Financial Services"]
    for k, p in enumerate(cfg.fiscal_years):  # book per share +10% a year, steadily
        raw.loc[0, f"Common Stock Equity_{p}"] = raw.loc[0, f"Ordinary Shares Number_{p}"] * 10 / 1.1**k
        raw.loc[0, f"FX_Hist_{p}"] = 1.0
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Financial_Mode"]
    assert df.loc[0, "Per_Share_Compounding"] == pytest.approx(10.0, abs=1.5)
    assert {"Return_Quality", "Per_Share_Compounding"} <= {c[0] for c in cfg.financial_quality_criteria}


def test_insurers_take_the_lower_of_median_and_mean(cfg):
    raw = make_raw(cfg, n=3)
    raw.loc[0, ["industry", "sector"]] = ["Insurance - Property & Casualty", "Financial Services"]
    df = calculate_metrics(raw, cfg)
    name = "Aftertax_PreFin_Yield"
    from fq.metrics.common import _distinct, _distinct_mean

    mean, median = (
        _distinct_mean(df, name, cfg.avg_years, cfg)[0],
        _distinct(df, name, cfg.avg_years, cfg).median(axis=1)[0],
    )
    assert df.loc[0, "Business_Model"] == "insurer"
    assert df.loc[0, "Aftertax_Yield_Avg"] <= max(mean, median) + 1e-9


def test_a_bank_by_industry_without_a_lenders_balance_sheet_is_flagged(cfg):
    raw = make_raw(cfg, n=2)
    raw["industry"] = "Capital Markets"
    for p in cfg.fiscal_years:
        for i in (0, 1):
            raw.loc[i, f"Stockholders Equity_{p}"] = raw.loc[i, f"Total Revenue_{p}"]
        raw.loc[0, f"Total Assets_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 2  # equity-funded, revenue 50% of assets
        raw.loc[1, f"Total Assets_{p}"] = raw.loc[1, f"Total Revenue_{p}"] * 12  # a lender's balance sheet
        raw.loc[0, f"Net Interest Income_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 0.5  # not a fee network
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Business_Model"] == "bank"
    assert "not a lender" not in str(df.loc[1, "Model_Review"])
    assert "operating or holding" in str(df.loc[0, "Model_Review"])


def test_a_bank_by_industry_holding_mostly_equity_is_flagged_whatever_its_revenue(cfg):
    # A crypto treasury filed as "Capital Markets" (PURR): assets ~1.1x equity, revenue under 1% of them. No deposit
    # lender is funded like that, so the review must not wait for revenue to be large.
    raw = make_raw(cfg, n=1)
    raw["industry"] = "Capital Markets"
    for p in cfg.fiscal_years:
        raw.loc[0, f"Stockholders Equity_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 100
        raw.loc[0, f"Total Assets_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 110
        raw.loc[0, f"Net Interest Income_{p}"] = raw.loc[0, f"Total Revenue_{p}"] * 0.5
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Business_Model"] == "bank"
    assert "operating or holding" in str(df.loc[0, "Model_Review"])


def test_the_over_earner_cap_spares_a_steady_climb_but_not_a_last_year_spike():
    from fq.metrics.normalise import _pick_level

    # FICO's ROIC rose every year (41 -> 69): a steady improver, not a peak, keeps its latest level.
    level, _, capped = _pick_level([41, 45, 48, 60, 69], "STRICT_GROWTH")
    assert (level, capped) == (69, False)
    # Rising every year but the last step alone is a jump (12 -> 30): a windfall, capped at the median.
    level, _, capped = _pick_level([10, 11, 12, 30], "STRICT_GROWTH")
    assert capped and level == pytest.approx(11.5)
    # Every year up, but with 1.5x steps on the way: a recovery from a trough (FLXS), capped.
    assert _pick_level([3.6, 7.0, 14.5, 21.4], "STRICT_GROWTH")[2]
    # A climb out of losses ending far above the anchor is a spike too.
    assert _pick_level([-5, -3, -1, 8], "STRICT_GROWTH")[2]
    # The decision-log case (not a strict climb) stays capped.
    assert _pick_level([10, 10, 12, 25], "RECENT_GROWTH")[2]


def test_the_owner_trend_is_a_rate_of_the_larger_of_the_line_and_the_level(cfg):
    from fq.metrics.valuation import _trend_denominator

    # Reported FCF at 0.6 per share, earning power 1.0: a slope of 0.06 a year is 6% of what gets compounded, not 10%.
    assert _trend_denominator(0.6, 1.0, 0.0, cfg) == 1.0
    assert _trend_denominator(1.2, 1.0, 0.0, cfg) == 1.2  # a level below the line never raises the rate
    assert _trend_denominator(0.6, 0.0, 0.0, cfg) == 0.6  # no positive level: the line alone, as before


def test_growth_is_measured_beyond_each_years_us_inflation(cfg, monkeypatch):
    # 2022-like: US prices up 8% a year; 8% more USD revenue bought no volume, so no growth capex or working capital.
    from fq import inflation

    rate = inflation.rate
    monkeypatch.setattr(inflation, "rate", lambda s, c, y: 8.0 if c == "USD" else rate(s, c, y))
    raw, _, df = _working_capital(cfg, growth=0.08, payables=0.2, moved=-0.05)
    assert df.loc[0, "Inflation_US_Y0"] == pytest.approx(0.08)
    assert df.loc[0, "Growth_WC_Owner_Y0"] == pytest.approx(0, abs=1e-6 * raw.loc[0, "Total Revenue_Y0"])
    assert df.loc[0, "Growth_Capex_Owner_Y0"] == pytest.approx(0, abs=1e-6 * raw.loc[0, "Total Revenue_Y0"])


def test_shrinking_revenue_is_flagged_and_leaves_highlights(cfg):
    raw = make_raw(cfg, n=4)
    for k, p in enumerate(reversed(cfg.fiscal_years)):  # oldest first
        for col in [c for c in raw.columns if c.endswith(f"_{p}") and not c.startswith(("PeriodEnd", "Ordinary"))]:
            raw.loc[1, col] = raw.loc[0, col] * 0.88**k  # company 0 shrunk 12% a year
    raw.loc[2, "industry"] = "Banks - Regional"
    df = calculate_metrics(raw, cfg)
    assert df.loc[1, "Revenue_Trend_Real"] < cfg.revenue_decline_rate and df.loc[1, "Revenue_Trend_R2"] > 0.9
    assert df.loc[1, "Revenue_Decline"]
    assert not df.loc[0, "Revenue_Decline"] and not df.loc[2, "Revenue_Decline"]  # grower; bank left out
    ranked = apply_dual_ranking(df, cfg)
    assert raw.loc[1, "Ticker"] not in set(top_n(ranked, "Value_Rank", 4)["Ticker"])
