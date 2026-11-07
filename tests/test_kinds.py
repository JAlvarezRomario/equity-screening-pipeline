"""The lists by kind (user's choice, Nov 5, 2026): banks, insurers, holdings and REITs ranked apart and judged against
their own kind; energy ranked and seen apart; the quadrant draws ranked companies only."""

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


def test_energy_list_wants_mid_cycle_returns_and_a_sound_balance_sheet():
    df = _frame(4, sector="Energy", Rank=[1.0, 2, 3, 4], ROIC_Norm=[15.0, 15, 5, 15], Net_Cash_to_Cap=[10.0, -50, 10, 5],
                NetDebt_to_EBIT=[0.0, 6.0, 0.0, 0.0])  # fmt: skip
    df.loc[3, "sector"] = "Industrials"
    # T1 is in net debt above the Shortlist's bound, T2 earns too little on its capital, T3 is not energy.
    assert sc.energy_list(df)["Ticker"].tolist() == ["T0"]


def test_below_book_takes_holdings_by_their_kind_and_orders_by_its_score():
    df = _frame(6, Rank_Group=["ranked"] * 5 + ["holding"], Business_Model=["operating"] * 5 + ["holding"],
                Rank=[1.0, 2, 3, 4, 5, np.nan], Asset_Play=[True, False, False, False, False, True],
                Asset_Play_Score=[40.0, np.nan, np.nan, np.nan, np.nan, 80.0])  # fmt: skip
    assert sc.asset_play_list(df)["Ticker"].tolist() == ["T5", "T0"]  # no rank for the holding: its score orders


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


def test_reits_and_energy_leave_out_a_downturn_fall_over_a_quarter_but_keep_a_blank():
    reits = _record(_frame(5, Rank_Group="real_estate", Business_Model="asset_heavy", Owner_Yield=6.0))
    assert sc.real_estate_list(reits.assign(Recession_Dividend_Fall=-40.0)).empty
    assert sc.real_estate_list(reits.assign(Recession_Dividend_Fall=np.nan))["Ticker"].tolist() == ["T0"]
    energy = _record(_frame(2, sector="Energy", Rank=[1.0, 2], ROIC_Norm=15.0, Net_Cash_to_Cap=10.0,
                            Value_Score=[90.0, 85.0], Quality_Score=80.0),
                     Recession_Dividend_Fall=[-60.0, np.nan])  # fmt: skip
    assert sc.energy_list(energy)["Ticker"].tolist() == ["T1"]  # T0 halved its dividend in a downturn; T1 pays none


def test_untaxed_dividends_needs_ten_years_without_a_cut_and_a_record():
    n = 5
    df = _frame(n, Dividend_WHT=0.0, Dividend_Yield=6.0, Owner_Yield=9.0, EPV_Per_Share_Yield=9.0,
                Rank=np.arange(1.0, n + 1), In_Best_Fraction=True, **{f"Ordinary Shares Number_{p}": 100.0 for p in ("Y0", "Y1", "Y2", "Y3")},
                **{f"Owner_Earnings_{p}": v for p, v in zip(("Y0", "Y1", "Y2", "Y3"), (13.0, 12, 11, 10), strict=True)})  # fmt: skip
    df = _record(df, Dividend_Unbroken_Years=[15.0, 6.0, 0.0, np.nan, 12.0],
                 Dividend_Cut_Recent=[False, False, True, False, False])  # fmt: skip
    # T1 cut 6 years ago, T2 cut this year, T3 has no record; T0 and T4 have held for 10 years or more.
    out = sc.dividend_carry_list(df)
    assert out["Ticker"].tolist() == ["T0", "T4", "T1", "T3"]  # the shorter records fill; a cut now never does
    assert out["Carry_Record"].tolist() == [True, True, False, False]


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


def test_untaxed_dividends_yield_must_beat_the_obtained_borrowing_cost_by_a_point():
    n = 2
    df = _frame(n, Dividend_WHT=0.0, Dividend_Yield=[6.0, 5.5], Owner_Yield=9.0, EPV_Per_Share_Yield=9.0,
                Rank=[1.0, 2.0], In_Best_Fraction=True, Borrow_Rate=4.8,
                **{f"Ordinary Shares Number_{p}": 100.0 for p in ("Y0", "Y1", "Y2", "Y3")},
                **{f"Owner_Earnings_{p}": v for p, v in zip(("Y0", "Y1", "Y2", "Y3"), (13.0, 12, 11, 10), strict=True)})  # fmt: skip
    assert sc.dividend_carry_list(_record(df))["Ticker"].tolist() == ["T0"]  # 5.5% is under a 4.8% loan plus a point
    out = sc.dividend_carry_list(
        _record(df.assign(Borrow_Rate=3.5, Dividend_Yield=[6.0, 4.8]))
    )  # 4.8%: a point above the loan, under the 5% floor
    assert out["Ticker"].tolist() == ["T0", "T1"] and out["Carry_Record"].tolist() == [True, False]


def test_untaxed_dividends_are_covered_by_what_a_share_earns_after_interest_and_never_filled():
    n = 3
    df = _frame(n, Dividend_WHT=0.0, Dividend_Yield=6.0, Owner_Yield=12.0, EPV_Per_Share_Yield=[9.0, 7.0, np.nan],
                Rank=[1.0, 2.0, 3.0], In_Best_Fraction=True,
                **{f"Ordinary Shares Number_{p}": 100.0 for p in ("Y0", "Y1", "Y2", "Y3")},
                **{f"Owner_Earnings_{p}": v for p, v in zip(("Y0", "Y1", "Y2", "Y3"), (13.0, 12, 11, 10), strict=True)})  # fmt: skip
    # T1 pays 86% of its earnings power after interest though its owner yield (before interest, 3 years ahead) is
    # twice the dividend; T2 has no figure to cover it with: on borrowed money, a blank is no evidence.
    out = sc.dividend_carry_list(_record(df), 20)
    assert out["Ticker"].tolist() == ["T0"]  # one passes, one is shown: the list is not filled


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
