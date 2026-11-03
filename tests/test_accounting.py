"""Accounting consistency: claims in EV, worst cases that must not read as missing, declining businesses."""

import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import EV_FLOOR_SHARE, calculate_metrics


def test_negative_growth_is_not_softened_for_cyclicals(cfg):
    raw = make_raw(cfg, n=2)
    for k, p in enumerate(cfg.annual_years):  # revenue falling ~4% a year for both
        raw.loc[[0, 1], f"Total Revenue_{p}"] = 1e9 * 1.04**k
    raw.loc[1, "industry"] = "Steel"  # cyclical industry
    raw.loc[0, "industry"] = "Software - Application"
    df = calculate_metrics(raw, cfg)
    assert bool(df.loc[1, "Cyclical"]) and not bool(df.loc[0, "Cyclical"])
    assert df.loc[0, "Growth_Used"] < 0
    assert df.loc[1, "Growth_Used"] == pytest.approx(df.loc[0, "Growth_Used"], rel=0.05)


def test_declining_revenue_base_is_the_latest_level(cfg):
    raw = make_raw(cfg, n=2)
    for k, p in enumerate(reversed(cfg.fiscal_years)):  # oldest first, falling every period
        raw.loc[0, f"Total Revenue_{p}"] = 2e9 * 0.9**k
        raw.loc[0, f"Free Cash Flow_{p}"] = 2e9 * 0.9**k * 0.1  # steady 10% margin before SBC
        raw.loc[0, f"Stock Based Compensation_{p}"] = 0.0
        raw.loc[0, f"Interest Expense_{p}"] = 0.0  # no financing: FCF before and after it are the same
    df = calculate_metrics(raw, cfg)
    latest = raw.loc[0, "Total Revenue_TTM"]
    assert df.loc[0, "FCFy_Norm"] == pytest.approx(0.1 * latest * 100 / df.loc[0, "marketCap"])  # per share


def test_net_cash_above_market_cap_floors_ev_and_is_flagged(cfg):
    raw = make_raw(cfg, n=2)
    plain = calculate_metrics(raw, cfg)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Debt_{p}"] = 0.0
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = 20e9  # far above the ~5bn market cap
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "enterpriseValue"] == pytest.approx(df.loc[0, "marketCap"] * EV_FLOOR_SHARE)
    assert df.loc[0, "FCFy_TTM"] == pytest.approx(plain.loc[0, "FCFy_TTM"])  # per share: the price, not EV
    assert bool(df.loc[0, "Data_Suspect"]) and "net cash" in df.loc[0, "Suspect_Reasons"]
    assert not bool(df.loc[1, "EV_Floored"])


def test_loss_on_no_capital_is_worst_roic(cfg):
    raw = make_raw(cfg, n=1)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Stockholders Equity_{p}"] = -5e9
        raw.loc[0, f"Total Debt_{p}"] = 1e9
        for concept in ("Operating Income", "EBIT"):
            raw.loc[0, f"{concept}_{p}"] = -1e8
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "ROIC_TTM"] == -cfg.roic_cap

    raw.loc[0, "Ticker"] = "KSPI"  # a bank: ROE on negative equity is also the worst case
    assert calculate_metrics(raw, cfg).loc[0, "ROIC_TTM"] == -cfg.roic_cap


def test_leases_leave_debt_and_ifrs_pays_them_from_fcf(cfg):
    raw = make_raw(cfg, n=3)
    raw.loc[0, "Ticker"] = "USCO"  # US GAAP: rent already in OCF
    raw.loc[1, "Ticker"] = "EUCO.MC"  # IFRS: lease principal in financing
    raw.loc[2, "Ticker"] = "EUNO.MC"  # IFRS without the current part: leases stay debt
    plain = calculate_metrics(raw, cfg)
    for p in cfg.fiscal_years:
        raw.loc[:, f"Capital Lease Obligations_{p}"] = 2e8  # inside Total Debt (0.3 x revenue >= 0.3bn)
        raw.loc[[0, 1], f"Current Capital Lease Obligation_{p}"] = 4e7
    df = calculate_metrics(raw, cfg)
    assert list(df["Accounting_Standard"]) == ["us_gaap", "ifrs", "ifrs"]
    assert df.loc[0, "Net_Debt"] == pytest.approx(plain.loc[0, "Net_Debt"] - 2e8)
    assert df.loc[0, "FCF_Adj_Y0"] == pytest.approx(plain.loc[0, "FCF_Adj_Y0"])
    assert df.loc[1, "Net_Debt"] == pytest.approx(plain.loc[1, "Net_Debt"] - 2e8)
    assert df.loc[1, "FCF_Adj_Y0"] == pytest.approx(plain.loc[1, "FCF_Adj_Y0"] - 4e7)
    assert df.loc[2, "Net_Debt"] == pytest.approx(plain.loc[2, "Net_Debt"])
    assert df.loc[2, "FCF_Adj_Y0"] == pytest.approx(plain.loc[2, "FCF_Adj_Y0"])
    assert list(df["Lease_Adjusted"]) == [True, True, False]


def test_ifrs_lease_interest_is_not_added_back_to_value(cfg):
    # IFRS: interest expense holds the leases' interest, but the lease liability left debt (`capital._leases`) and
    # only its principal is charged (`Lease_Cost`). Adding all interest back left the leases' interest unpaid and
    # unclaimed (BOWL.L: no financial debt, 316M of leases, 19.8M of interest back).
    raw = make_raw(cfg, n=2)
    raw.loc[0, "Ticker"] = "EUCO.MC"  # only leases
    raw.loc[1, "Ticker"] = "EUMX.MC"  # half leases, half borrowings
    for p in cfg.fiscal_years:
        raw.loc[:, f"Interest Expense_{p}"] = -2e7
        raw.loc[0, f"Total Debt_{p}"] = raw.loc[0, f"Capital Lease Obligations_{p}"] = 4e8
        raw.loc[1, f"Total Debt_{p}"], raw.loc[1, f"Capital Lease Obligations_{p}"] = 8e8, 4e8
        raw.loc[:, f"Current Capital Lease Obligation_{p}"] = 4e7
    df = calculate_metrics(raw, cfg)
    for p in ("TTM", cfg.latest):
        back = df[f"FCF_PreFin_{p}"] - df[f"FCF_Adj_{p}"]
        after_tax = 1 - df[f"Tax_Rate_{p}"].fillna(cfg.notional_tax_rate)
        assert back[0] == pytest.approx(0.0, abs=1.0)
        assert back[1] == pytest.approx(1e7 * after_tax[1])  # the borrowings' half of the interest
    assert df.loc[1, "Debt_Rate"] == pytest.approx(2.5, rel=0.05)  # 10M on 400M of borrowings, not 20M on them


def test_quarters_shared_by_ttm_and_latest_year_count_once(cfg):
    raw = make_raw(cfg, n=1)  # TTM ends six months after Y0
    periods = cfg.fiscal_years  # TTM, Y0..Y4
    for i, v in enumerate([100, -50, 100, 100, 100, -50]):
        raw.loc[0, f"Free Cash Flow_{periods[i]}"] = v
        raw.loc[0, f"Stock Based Compensation_{periods[i]}"] = 0.0
    df = calculate_metrics(raw, cfg)
    # Half a year apart: Y0 holds six months TTM does not (CNC's loss year), so it stays in spreads and
    # shares, and weighs only that half in averages.
    assert df.loc[0, "FCF_Positive_Share"] == pytest.approx(4 / 6 * 100)
    span = (pd.Timestamp(raw.loc[0, "PeriodEnd_TTM"]) - pd.Timestamp(raw.loc[0, "PeriodEnd_Y0"])).days / 365.25
    y = {p: df.loc[0, f"EBIT_Yield_{p}"] for p in ("TTM", "Y0", "Y1", "Y2")}
    expected = (y["TTM"] + span * y["Y0"] + y["Y1"] + y["Y2"]) / (3 + span)
    assert df.loc[0, "EBIT_Yield_Avg"] == pytest.approx(expected)

    # One quarter apart: TTM holds three of Y0's quarters, so Y0 leaves the spreads.
    raw.loc[0, "PeriodEnd_TTM"] = (pd.Timestamp(raw.loc[0, "PeriodEnd_Y0"]) + pd.DateOffset(months=3)).strftime(
        "%Y-%m-%d"
    )
    assert calculate_metrics(raw, cfg).loc[0, "FCF_Positive_Share"] == pytest.approx(4 / 5 * 100)

    raw.loc[0, "PeriodEnd_TTM"] = None  # unknown TTM date: nothing is dropped
    assert calculate_metrics(raw, cfg).loc[0, "FCF_Positive_Share"] == pytest.approx(4 / 6 * 100)


def test_preferred_stock_is_a_claim_in_ev(cfg):
    raw = make_raw(cfg, n=2)
    plain = calculate_metrics(raw, cfg)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Preferred Stock Equity_{p}"] = 1e9
    raw.loc[1, "Ticker"] = "KSPI"  # financial mode: market cap + minorities + preferred
    for p in cfg.fiscal_years:
        raw.loc[1, f"Preferred Stock Equity_{p}"] = 5e8
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "enterpriseValue"] == pytest.approx(plain.loc[0, "enterpriseValue"] + 1e9)
    assert df.loc[0, "Net_Debt"] == pytest.approx(plain.loc[0, "Net_Debt"] + 1e9)
    assert df.loc[1, "enterpriseValue"] == pytest.approx(df.loc[1, "marketCap"] + 5e8)


def test_value_yields_are_per_share(cfg):
    raw = make_raw(cfg, n=3)
    plain = calculate_metrics(raw, cfg)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Interest Paid Cff_{p}"] = -5e7  # IFRS: interest paid under financing, still in FCF
        raw.loc[1, f"Net Income Including Noncontrolling Interests_{p}"] = (
            raw.loc[1, f"Net Income_{p}"] / 0.75
        )  # minorities: a quarter
        raw.loc[2, f"Total Debt_{p}"] = raw.loc[2, f"Total Debt_{p}"] * 5  # more debt at the same share price
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "FCF_Adj_Y0"] == pytest.approx(plain.loc[0, "FCF_Adj_Y0"] - 5e7)
    assert df.loc[1, "Parent_Share"] == pytest.approx(0.75)
    assert df.loc[1, "FCFy_Norm"] == pytest.approx(plain.loc[1, "FCFy_Norm"] * 0.75)
    # The same consolidated flow; the shareholders own three quarters of it, so the group costs 4/3 of the price.
    assert df.loc[1, "Value_Base"] == pytest.approx(df.loc[1, "marketCap"] / 0.75 + df.loc[1, "Net_Claims"])
    flow = lambda d: d.loc[1, "Pretax_Yield_Avg"] * d.loc[1, "Value_Base"]  # noqa: E731
    assert flow(df) == pytest.approx(flow(plain))
    # Debt counts at face: the operations cost more at the same share price.
    assert df.loc[2, "Value_Base"] > plain.loc[2, "Value_Base"]
    assert df.loc[2, "EPV_Yield"] < plain.loc[2, "EPV_Yield"]


def test_a_quarter_reporting_only_leases_keeps_the_annual_borrowings(cfg):
    from conftest import make_raw
    from fq.metrics import calculate_metrics

    raw = make_raw(cfg, n=2)
    raw.loc[0, f"Total Debt_{cfg.latest}"] = 40e9
    raw.loc[0, f"Capital Lease Obligations_{cfg.latest}"] = 8e9
    raw.loc[0, "Total Debt_TTM"] = raw.loc[0, "Capital Lease Obligations_TTM"] = 9e9  # the quarter shows leases only
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Financial_Debt_TTM"] == pytest.approx(32e9)


def test_missing_sbc_with_ocf_is_zero(cfg):
    import numpy as np

    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Stock Based Compensation_{p}"] = np.nan
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "SBC_to_Revenue"] == pytest.approx(0.0)


def test_gross_unusual_items_cannot_push_clean_pretax_below_operating_income(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Operating Income_{p}"] = 4.2e9
        raw.loc[0, f"EBIT_{p}"] = 4.2e9
        raw.loc[0, f"Interest Expense_{p}"] = 0.2e9
        raw.loc[0, f"Pretax Income_{p}"] = 5.3e9
        raw.loc[0, f"Total Unusual Items_{p}"] = 5.7e9  # CME-like gross collateral income > pretax
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Aftertax_Yield_Avg"] > 0


def test_working_capital_addback_is_bounded_by_nopat(cfg):
    raw = make_raw(cfg, n=2)
    for k, p in enumerate(reversed(cfg.fiscal_years)):
        rev = 1e9 * (1.5**k)
        raw.loc[0, f"Total Revenue_{p}"] = rev
        raw.loc[0, f"Operating Income_{p}"] = rev * 0.10
        raw.loc[0, f"EBIT_{p}"] = rev * 0.10
        raw.loc[0, f"Pretax Income_{p}"] = rev * 0.10
        raw.loc[0, f"Tax Provision_{p}"] = rev * 0.02
        raw.loc[0, f"Net Income_{p}"] = rev * 0.08
        raw.loc[0, f"Operating Cash Flow_{p}"] = -rev * 0.10
        raw.loc[0, f"Free Cash Flow_{p}"] = -rev * 0.12
        raw.loc[0, f"Capital Expenditure_{p}"] = -rev * 0.02
        raw.loc[0, f"Stock Based Compensation_{p}"] = 0.0
        raw.loc[0, f"Interest Expense_{p}"] = 0.0
        raw.loc[0, f"Current Assets_{p}"] = rev * 0.9
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = rev * 0.1
        raw.loc[0, f"Current Liabilities_{p}"] = rev * 0.1
        raw.loc[0, f"Change In Working Capital_{p}"] = -rev * 0.30
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "FCF_Owner_Y0"] <= df.loc[0, "NOPAT_Y0"] + 1e-6


def test_financials_do_not_contaminate_operating_value_market_pool(cfg):
    from dataclasses import replace

    from fq.ranking import apply_dual_ranking

    c = replace(cfg, value_criteria=(("a", "top", 1.0),), quality_criteria=(("a", "top", 1.0),),
                financial_value_criteria=(("a", "top", 1.0),), financial_quality_criteria=(("a", "top", 1.0),),
                winsor_pct=0.0)  # fmt: skip
    df = pd.DataFrame(
        {
            "Ticker": ["OP1", "OP2", "OP3", "B1", "B2", "B3"],
            "Business_Model": ["operating", "operating", "operating", "bank", "bank", "bank"],
            "Financial_Mode": [False, False, False, True, True, True],
            "a": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    ranked = apply_dual_ranking(df, c).set_index("Ticker")
    alone = apply_dual_ranking(df[df["Business_Model"] == "operating"].reset_index(drop=True), c).set_index("Ticker")
    for t in ("OP1", "OP2", "OP3"):
        assert ranked.at[t, "Value_Score"] == pytest.approx(alone.at[t, "Value_Score"])


def test_client_cash_ocf_cap_bounds_customer_deposit_inflows(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "Ticker"] = "WISE.L"  # client_cash override
    for p in cfg.fiscal_years:
        raw.loc[0, f"Net Income_{p}"] = 4e8
        raw.loc[0, f"Depreciation And Amortization_{p}"] = 5e7
        raw.loc[0, f"Operating Cash Flow_{p}"] = 3e9  # customer wallet deposits mixed into OCF
        raw.loc[0, f"Free Cash Flow_{p}"] = 2.95e9
        raw.loc[0, f"Capital Expenditure_{p}"] = -5e7
        raw.loc[0, f"Stock Based Compensation_{p}"] = 0.0
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "FCF_Reported_Y0"] <= 4e8 + 5e7 - 5e7 + 1e-6
