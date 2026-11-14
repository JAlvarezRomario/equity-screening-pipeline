"""The lists by kind (user's choice, Nov 5, 2026): banks, insurers, holdings and REITs ranked apart and judged against
their own kind; the quadrant draws ranked companies only (the Energy tab retired Nov 9, 2026)."""

import numpy as np
import pandas as pd
import pytest

from fq.report import screens as sc
from fq.report.index_charts import quadrant_figure


def _frame(n: int, **cols) -> pd.DataFrame:
    base = {
        "Ticker": [f"T{i}" for i in range(n)],
        "shortName": [f"Co {i}" for i in range(n)],
        "country": "Japan",
        "Market Cap": 1.0,
        "FCFy_Norm": 5.0,
        "Value_Score": np.linspace(90, 40, n),
        "Quality_Score": np.linspace(85, 45, n),
        "Rank_Group": "ranked",
        "Business_Model": "operating",
    }
    return pd.DataFrame(base | cols)


def test_each_kind_is_ordered_and_cut_among_its_own():
    n = 10
    df = _frame(n, Rank_Group=["financial"] * 5 + ["ranked"] * 5, Business_Model=["bank"] * 5 + ["operating"] * 5)
    rank, best = sc.kind_order(df)
    assert rank[:5].tolist() == [1, 2, 3, 4, 5] and rank[5:].isna().all()  # ranked companies have no kind
    assert best.tolist() == [True] + [False] * 9  # the best fifth of five banks: one


def test_bank_and_insurer_lists_hold_their_own_kind_with_enough_capital():
    n = 10
    models = ["bank"] * 5 + ["insurer"] * 5
    df = _frame(n, Rank_Group="financial", Business_Model=models, Financial_Mode=True, Equity_to_Assets=10.0)
    df.loc[0, "Equity_to_Assets"] = 3.0  # the best bank, thinly capitalised
    banks, insurers = sc.bank_list(df), sc.insurer_list(df)
    assert banks.empty  # the best fifth of banks is T0 alone, and it fails the capital test
    assert insurers["Ticker"].tolist() == ["T5"]


def test_real_estate_list_bounds_debt_and_never_tests_book():
    df = _frame(5, Rank_Group="real_estate", Business_Model="asset_heavy", Owner_Yield=6.0, Interest_Coverage=3.0,
                NetDebt_to_EBIT=6.0, BV_to_Price=40.0)  # fmt: skip
    assert sc.real_estate_list(df)["Ticker"].tolist() == ["T0"]  # far above book, still its kind's best
    df.loc[0, "Interest_Coverage"] = 1.1  # a debt trap
    assert sc.real_estate_list(df).empty


def test_compounding_below_nav_takes_holdings_by_their_kind_and_orders_by_price_to_nav():
    # Nov 14, 2026: the price against the estimated net assets orders it; no rank is asked, so a holding sits with the
    # ranked companies; no cyclical (Nov 10, 2026).
    df = _frame(6, Rank_Group=["ranked"] * 5 + ["holding"], Business_Model=["operating"] * 5 + ["holding"],
                Rank=[1.0, 2, 3, 4, 5, np.nan], BV_to_Price=[110.0, 110, 110, 110, 110, 140.0], NAV_Compounding=9.0,
                Check_Book_Book=100.0, Check_Book_NAV_Base=[100.0, np.nan, np.nan, np.nan, np.nan, 100.0],
                Check_Book_Topics="[]")  # fmt: skip
    assert sc.asset_play_list(df)["Ticker"].tolist() == ["T5", "T0"]  # 0.71x before 0.91x, rank or not
    assert sc.asset_play_list(df.assign(NAV_Compounding=5.0)).empty
    assert sc.asset_play_list(df.assign(Cyclical=True)).empty


def test_quadrant_draws_ranked_companies_only():
    df = _frame(6, Rank_Group=["ranked"] * 4 + ["financial", "unprofitable"])
    fig, _ = quadrant_figure(df)
    assert "n=4" in fig.layout.title.text  # the bank and the loss-maker are not drawn


def _record(df: pd.DataFrame, **cols) -> pd.DataFrame:
    """The dividend record's columns (`dividend_record.COLUMNS`) set directly: a clean record unless `cols` says."""
    base = {"Dividend_Unbroken_Years": 15.0, "Dividend_Record_Whole": True, "Regular_DPS_Growth_10Y": 4.0,
            "Dividend_Cut_Recent": False, "Recession_Dividend_Fall": 0.0, "Recession_Dividend_Note": None,
            "Recessions_Covered": 2.0, "Recessions_Due": 2.0}  # fmt: skip
    return df.assign(**(base | cols))


def test_lists_by_kind_leave_out_a_dividend_cut_paid_this_year():
    banks = _record(_frame(5, Rank_Group="financial", Business_Model="bank", Financial_Mode=True,
                           Equity_to_Assets=10.0), Dividend_Cut_Recent=[True, False, False, False, False])  # fmt: skip
    assert sc.bank_list(banks).empty  # its best fifth is T0 alone, which just cut
    assert sc.bank_list(banks.assign(Dividend_Cut_Recent=False))["Ticker"].tolist() == ["T0"]


def test_reits_leave_out_a_downturn_fall_over_a_quarter_but_keep_a_blank():
    reits = _record(_frame(5, Rank_Group="real_estate", Business_Model="asset_heavy", Owner_Yield=6.0))
    assert sc.real_estate_list(reits.assign(Recession_Dividend_Fall=-40.0)).empty
    assert sc.real_estate_list(reits.assign(Recession_Dividend_Fall=np.nan))["Ticker"].tolist() == ["T0"]


def test_a_benchmark_ranked_before_the_rank_groups_never_lends_a_bank_its_old_position():
    from fq.ranking import rankable

    old = _frame(3, Business_Model=["operating", "bank", "operating"], Owner_Yield=5.0, ROIC_Norm=12.0,
                 Rank=[1.0, 2, 3]).drop(columns="Rank_Group")  # fmt: skip
    assert rankable(old).tolist() == [True, False, True]
    snapshot = old[["Ticker", "Value_Score", "Quality_Score", "Rank", "Business_Model"]]  # scores without inputs
    assert rankable(snapshot).tolist() == [True, True, True]  # an old snapshot keeps its ranks as they were


def test_the_borrowing_cost_is_obtained_from_the_ecb_with_a_stored_and_a_built_in_fallback(monkeypatch, tmp_path):
    from fq import fx

    path = tmp_path / "borrow_rate.json"
    monkeypatch.setattr(fx, "_estr", lambda: 2.44)
    assert fx.get_borrow_rate(path) == pytest.approx(2.44 + fx.BROKER_SPREAD + fx.BORROW_BUFFER)
    monkeypatch.setattr(fx, "_estr", lambda: None)
    monkeypatch.setattr(fx, "_borrow_cache", {})
    assert fx.get_borrow_rate(path) == pytest.approx(4.44)  # the ECB out of reach: the stored rate
    assert fx.get_borrow_rate(tmp_path / "none.json") == fx.FALLBACK_BORROW_RATE


def test_a_young_company_in_a_hard_hit_industry_needs_to_lead_it_with_net_cash():
    from fq.report.screens import _lasting_business

    df = _record(_frame(3, industry="Airports & Air Services", Industry_Leader=[False, True, True],
                        Net_Debt=[-10.0, -10.0, 50.0]), Recession_Dividend_Fall=np.nan)  # fmt: skip
    # No record reaches a recession: T0 is no leader, T2 leads but owes; T1 leads with net cash.
    assert _lasting_business(df).tolist() == [False, True, False]
    assert _lasting_business(df.assign(Recession_Dividend_Fall=0.0)).all()  # its own record decides
    assert _lasting_business(df.assign(industry="Packaged Foods")).all()  # not a hard-hit industry: a blank passes


def test_utilities_list_holds_its_own_kind_with_debt_bounded_and_the_dividend_held():
    df = _frame(5, Rank_Group="utility", Business_Model="asset_heavy", Owner_Yield=6.0, Interest_Coverage=3.0,
                NetDebt_to_EBIT=6.0)  # fmt: skip
    assert sc.utility_list(df)["Ticker"].tolist() == ["T0"]  # the best fifth of five utilities
    df.loc[0, "NetDebt_to_EBIT"] = 9.0  # financed beyond the bound
    assert sc.utility_list(df).empty
    assert sc.kind(df).eq("utility").all()


def _payers(n: int, **cols) -> pd.DataFrame:
    """Dividend payers passing every Income gate unless `cols` says: a clean 15-year record growing 4% a year (Income's
    own record, `Income_Unbroken_Years`, as well as the 5% one)."""
    base = dict(Dividend_WHT=0.0, Dividend_Yield=5.0, EPV_Per_Share_Yield=9.0, ROIC_Norm=20.0,
                Rank=np.arange(1.0, n + 1), In_Best_Fraction=True, Income_Unbroken_Years=15.0,
                Income_Cut_Recent=False)  # fmt: skip
    return _record(_frame(n, **(base | cols)))


def test_income_keeps_a_withholding_of_ten_percent_at_most():
    out = sc.income_list(_payers(3, Dividend_WHT=[0.0, 10.0, 15.0]))
    assert sorted(out["Ticker"]) == ["T0", "T1"]
    assert out.set_index("Ticker").loc["T1", "Net_Dividend_Yield"] == pytest.approx(4.5)  # what a foreigner keeps


def test_income_needs_a_decade_without_a_cut_and_a_covered_dividend():
    df = _payers(4, EPV_Per_Share_Yield=[9.0, 9.0, 9.0, np.nan])
    df = df.assign(Income_Unbroken_Years=[15.0, 5.0, 15.0, 15.0], Income_Cut_Recent=[False, False, True, False])
    # T1 cut five years ago, T2 cut this year, T3 has no earnings power to cover the dividend with.
    assert sc.income_list(df)["Ticker"].tolist() == ["T0"]


def test_income_orders_by_yield_after_withholding_plus_growth():
    df = _payers(2, Dividend_Yield=4.0).assign(Regular_DPS_Growth_10Y=[2.0, 6.0])
    out = sc.income_list(df)
    assert out["Ticker"].tolist() == ["T1", "T0"]  # same yield, faster growth first, whatever the combined rank
    assert out["Income_Return"].tolist() == pytest.approx([10.0, 6.0])


def test_income_counts_growth_no_faster_than_the_business_funds_it_and_at_most_ten():
    df = _payers(2, Dividend_Yield=[3.0, 6.0], ROIC_Norm=[20.0, 10.0], EPV_Per_Share_Yield=[9.0, 8.0])
    df = df.assign(Regular_DPS_Growth_10Y=[15.0, 8.0])
    grown = sc.income_list(df).set_index("Ticker")["Income_Growth"]
    assert grown["T0"] == pytest.approx(10.0)  # 15% a year counts as 10: past it, a growth stock
    assert grown["T1"] == pytest.approx(2.5)  # pays out 75%: 10% ROIC on the quarter kept funds 2.5% a year


def test_income_shows_the_spread_over_the_loan_without_testing_it():
    out = sc.income_list(_payers(1, Dividend_Yield=3.0, Borrow_Rate=4.8))
    assert out["Ticker"].tolist() == ["T0"] and out["Loan_Spread"].iloc[0] == pytest.approx(-1.8)


def _annual(amounts: list[float], first: int = 2010) -> str:
    """A `DPS_Payments` JSON: one payment each June from `first`."""
    import json

    return json.dumps({f"{first + i}-06-15": a for i, a in enumerate(amounts) if a})


def test_income_lets_a_payout_ratio_dividend_dip_up_to_a_fifth_in_a_year():
    """Nov 8, 2026 (user's choice): a year may pay 20% less than the one before (Income only; the other lists keep
    5%); a deeper dip, or a year without a dividend, breaks the record."""
    years = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7]  # 2010-2024
    dip15, dip25, gap = years.copy(), years.copy(), years.copy()
    dip15[10], dip25[10], gap[10] = 1.45 * 0.85, 1.45 * 0.75, 0.0  # 2020
    df = _frame(4, DPS_Payments=[_annual(years), _annual(dip15), _annual(dip25), _annual(gap)],
                Fetched_At=pd.Timestamp("2025-03-01"), Dividend_WHT=0.0)  # fmt: skip
    out = sc.with_income_record(df)
    assert out["Income_Unbroken_Years"].tolist() == [15.0, 15.0, 4.0, 4.0]
    strict = sc.with_dividend_record(df)["Dividend_Unbroken_Years"]
    assert strict.tolist()[:2] == [15.0, 4.0]  # the 5% record the other lists read is unchanged
    taxed = sc.with_income_record(df.assign(Dividend_WHT=15.0))
    assert taxed["Income_Unbroken_Years"].isna().all()  # never judged: Income would not take it anyway
