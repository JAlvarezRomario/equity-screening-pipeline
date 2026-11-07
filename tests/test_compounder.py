"""The Compounder flag ("Mastercard-like", every trait at once) and owner earnings per share compounding."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.metrics.common import TTM_SHARED_SPAN, _ttm_span, adjusted_r2
from fq.ranking import apply_dual_ranking, compounder

MASTERCARD = {
    "Business_Model": "operating",
    "Pct_ROIC_Norm": 98.0,
    "Pct_ROIC_Variability": 85.0,
    "ROIC_Trend": 6.5,
    "Pct_Per_Share_Compounding": 88.0,
    "Owner_Earnings_Compounding": 19.7,
    "Share_Reduction_Rate": 2.4,
    "Value_Score": 30.0,
    "Quality_Score": 82.0,
}


def _rows(**changes: dict) -> pd.DataFrame:
    """One Mastercard-like row per name, each with its changes."""
    rows = [{"Ticker": name, **MASTERCARD, **change} for name, change in changes.items()]
    return pd.DataFrame(rows)


def test_a_mastercard_passes_and_each_missing_trait_fails(cfg):
    df = _rows(
        MA={},
        RISING={"Pct_ROIC_Variability": 40.0, "ROIC_Trend": 10.0},
        STEADY_WIGGLE={"Pct_ROIC_Variability": 80.0, "ROIC_Trend": -1.5},
        TINY_ISSUANCE={"Share_Reduction_Rate": -0.2},
        LOW_ROIC={"Pct_ROIC_Norm": 70.0},
        BUMPY={"Pct_ROIC_Variability": 40.0, "ROIC_Trend": 1.0},
        FADING={"ROIC_Trend": -6.0},
        SLOW={"Pct_Per_Share_Compounding": 60.0},
        STALLED={"Owner_Earnings_Compounding": 1.0},
        DILUTING={"Share_Reduction_Rate": -1.0},
        RECENT_DROP={"EBIT_Change_Recent": -15.0},
        UNKNOWN={"Owner_Earnings_Compounding": np.nan},
        LOW_QUALITY={"Quality_Score": 65.0},
        BANK={"Business_Model": "bank", "Financial_Mode": True},
        CASH_BURN={"FCF_Positive_Share": 60.0},
        POOR_CONV={"FCF_Conversion": 0.4},
        LEVERED={"NetDebt_to_EBIT": 4.0},
    )
    assert compounder(df, cfg).tolist() == [True] * 4 + [False] * 13


def test_untrusted_or_cyclical_companies_are_not_compounders(cfg):
    df = _rows(
        MA={},
        CYCLE={"Cyclical": True},
        SUSPECT={"Data_Suspect": True},
        DISTRESS={"Distress": True},
        PEAK={"Peak_Margin": True},
        THIN={"Insufficient_Data": True},
    )
    assert compounder(df, cfg).tolist() == [True] + [False] * 5


def test_owner_earnings_compounding_is_the_usd_per_share_trend(cfg):
    df = calculate_metrics(make_raw(cfg, n=4), cfg)
    assert df["Owner_Earnings_Compounding"].notna().any()
    row = df.iloc[0]
    # Hand-computed: owner earnings per share at each period's real date, TTM in place of a mostly shared Y0.
    span = _ttm_span(df.iloc[[0]], cfg).iloc[0]
    periods = [p for p in cfg.fiscal_years if not (p == cfg.latest and span < TTM_SHARED_SPAN)]
    y0 = pd.Timestamp(row[f"PeriodEnd_{cfg.latest}"])
    points = []
    for p in periods:
        value = row[f"Owner_Earnings_{p}"] / row[f"Ordinary Shares Number_{p}"]
        if pd.notna(value) and value > 0:
            points.append(((y0 - pd.Timestamp(row[f"PeriodEnd_{p}"])).days / 365.25, value))
    x, y = -np.array([t for t, _ in points]), np.log([v for _, v in points])
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (intercept + slope * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    expected = (np.exp(slope) - 1) * 100 * adjusted_r2(r2, len(points))  # adjusted R², as the scored trends
    assert row["Owner_Earnings_Compounding"] == pytest.approx(expected)


def test_the_flag_changes_no_score_or_rank(cfg):
    df = calculate_metrics(make_raw(cfg, n=6), cfg)
    ranked = apply_dual_ranking(df, cfg)
    assert "Compounder" in ranked.columns
    again = apply_dual_ranking(df.assign(Owner_Earnings_Compounding=99.0), cfg)
    for col in ("Value_Score", "Quality_Score", "Rank"):
        assert ranked[col].to_numpy() == pytest.approx(again[col].to_numpy(), nan_ok=True)


def test_off_the_radar_keeps_small_little_followed_steady_growers_in_rank_order():
    from fq.report.screens import off_radar_list

    base = {"country": "Spain", "sector": "Industrials", "Cyclical": False, "Peak_Margin_Ratio": 1.1,
            "numberOfAnalystOpinions": 1.0, "Market Cap": 1.5, "Per_Share_Compounding": 9.0,
            "Owner_Earnings_Compounding": 8.0, "Share_Reduction_Rate": 0.5, "ROIC_Norm": 18.0,
            "Value_Score": 70.0, "Quality_Score": 65.0}  # fmt: skip
    changes = {
        "B": {"Rank": 2.0},
        "A": {"Rank": 1.0, "Share_Reduction_Rate": -0.2},
        "NO_ANALYSTS": {"Rank": 3.0, "numberOfAnalystOpinions": np.nan},
        "US": {"country": "United States"},
        "BANK": {"sector": "Financial Services"},
        "REIT": {"sector": "Real Estate"},
        "CYCLE": {"Cyclical": True},
        "OVEREARNER": {"Peak_Margin_Ratio": 1.8},
        "FOLLOWED": {"numberOfAnalystOpinions": 3.0},
        "OUTSIDE_QUINTILE": {"Rank": 20.0},
        "BIG": {"Market Cap": 12.0},
        "DEAR": {"Value_Score": 55.0},
        "WEAK": {"Quality_Score": 55.0},
        "LOW_ROIC": {"ROIC_Norm": 8.0},
        "SLOW": {"Per_Share_Compounding": 2.0},
        "OE_FALLING": {"Owner_Earnings_Compounding": -3.0},
        "DILUTING": {"Share_Reduction_Rate": -1.0},
        "RECENT_DROP": {"EBIT_Change_Recent": -25.0},
        "SUSPECT": {"Data_Suspect": True},
    }
    df = pd.DataFrame([{"Ticker": t, "Rank": 1.0, **base, **c} for t, c in changes.items()])
    # 19 rows in all: the best quintile is the first ceil(0.2 x 19) = 4 ranks; "OUTSIDE_QUINTILE" ranks 20th.
    assert len(df) == 19
    assert off_radar_list(df)["Ticker"].tolist() == ["A", "B", "NO_ANALYSTS"]


def test_dividend_carry_needs_untaxed_covered_unbroken_dividends_in_the_best_quintile():
    from fq.report.screens import dividend_carry_list

    def company(ticker, rank, div=(6, 5, 4), oe=(9, 8, 7), **changes):
        row = {"Ticker": ticker, "Rank": rank, "Dividend_WHT": 0.0, "Dividend_Yield": 6.0, "Owner_Yield": 9.0,
               "EPV_Per_Share_Yield": 9.0, "Cyclical": False, "Peak_Margin_Ratio": 1.0,
               # The dividend record (Nov 5, 2026: it replaced the 4-year dividend trend), clean unless changed.
               "Dividend_Unbroken_Years": 12.0, "Dividend_Record_Whole": True, "Regular_DPS_Growth_10Y": 3.0,
               "Dividend_Cut_Recent": False, "Recession_Dividend_Fall": 0.0, "Recession_Dividend_Note": None,
               "Recessions_Covered": 2.0, "Recessions_Due": 2.0}  # fmt: skip
        for i in range(3):
            row |= {f"Cash Dividends Paid_Y{i}": -div[i], f"Owner_Earnings_Y{i}": oe[i],
                    f"Ordinary Shares Number_Y{i}": 1.0}  # fmt: skip
        return row | changes

    df = pd.DataFrame(
        [
            company("B", 2.0),
            company("A", 1.0, div=(6, 7, 4)),  # yearly payouts no longer judge it: the record does
            company("TAXED", 3.0, Dividend_WHT=15.0),
            company("LOW_YIELD", 3.0, Dividend_Yield=4.8),  # a point above the 3.75% fallback loan, under 5%
            company("UNCOVERED", 3.0, EPV_Per_Share_Yield=7.0),  # 86% of what a share earns after interest
            company("CUT_SIX_YEARS_AGO", 3.0, Dividend_Unbroken_Years=6.0),
            company("CUT_NOW", 3.0, Dividend_Unbroken_Years=0.0, Dividend_Cut_Recent=True),
            company("NO_RECORD", 3.0, Dividend_Unbroken_Years=float("nan")),
            company("LOWER_THAN_TEN_YEARS_AGO", 3.0, Regular_DPS_Growth_10Y=-1.0),
            company("OE_FALLING", 3.0, oe=(7, 8, 9)),
            company("CYCLE", 3.0, Cyclical=True),
            company("OUTSIDE", 30.0),
        ]
    )
    out = dividend_carry_list(df)
    assert out.loc[out["Carry_Record"], "Ticker"].tolist() == ["A", "B"]  # the full record first
    # Then the shorter records (Nov 7, 2026): no cut now, untaxed, covered, not cyclical, in the best.
    assert set(out.loc[~out["Carry_Record"], "Ticker"]) == {
        "LOW_YIELD",
        "CUT_SIX_YEARS_AGO",
        "NO_RECORD",
        "LOWER_THAN_TEN_YEARS_AGO",
        "OE_FALLING",
    }
    assert out["Ticker"].tolist()[:2] == ["A", "B"] and out["Carry_Order"].is_monotonic_increasing
    assert dividend_carry_list(df, 3)["Ticker"].tolist()[:2] == ["A", "B"] and len(dividend_carry_list(df, 3)) == 3


def _twenty_year_row(ticker, rank, owner=(130, 120, 110, 100), revenue=(400, 300, 200, 100), shares=(10, 10, 10, 11),
                     roic=(20, 19, 18, 17), margin=(0.22, 0.21, 0.20, 0.19), **over):  # fmt: skip
    """A company both 20-year lists admit, before `over`."""
    out = {"Ticker": ticker, "Rank": rank, "Owner_Yield": 8.0, "ROIC_Norm": 20.0, "Owner_Earnings_Compounding": 18.0,
           "Per_Share_Compounding": 10.0, "Cyclical": False, "Financial_Mode": False, "Peak_Margin_Ratio": 1.0,
           "industry": "Packaged Foods", "In_Best_Fraction": True}  # fmt: skip
    for i, p in enumerate(("Y0", "Y1", "Y2", "Y3")):
        out |= {
            f"Owner_Earnings_{p}": owner[i],
            f"Total Revenue_{p}": revenue[i],
            f"Ordinary Shares Number_{p}": shares[i],
            f"ROIC_{p}": roic[i],
            f"operatingMargins_{p}": margin[i],
        }
    return out | over


def test_operating_companies_need_high_returns_a_yield_and_smooth_growth_on_every_front():
    from fq.report.screens import operating_list

    r = _twenty_year_row
    df = pd.DataFrame([
        r("B", 2.0),
        r("A", 1.0),
        r("FLAT_HIGH_ROIC", 3.0, roic=(30, 30, 30, 30)),  # a high return may stay flat
        r("FLAT_MARGIN", 4.0, margin=(0.2, 0.2, 0.2, 0.2)),
        r("ROIC_SLIPS", 4.5, roic=(20, 18.5, 19, 17)),  # a year down 3%, still above where it started
        r("ONE_FLAT_OE", 5.0, owner=(130, 135, 110, 100)),  # a minor working-capital wiggle within the smooth trend
        r("PACKAGING", 5.5, industry="Packaging & Containers", Recession_Dividend_Fall=0.0),  # boring packaging/grocery companies are allowed
        # A recession hits packaging hard: with no record through one, only a leader with net cash (Nov 5, 2026).
        r("PACKAGING_YOUNG", 5.6, industry="Packaging & Containers"),
        r("BANK", 6.0, Financial_Mode=True),
        r("LOW_ROIC", 6.0, ROIC_Norm=11.0),
        r("DEAR", 6.0, Owner_Yield=4.9),
        r("CYCLICAL", 6.0, Cyclical=True),
        r("PEAK", 6.0, Peak_Margin_Ratio=2.0),
        r("AVOIDED", 6.0, industry="Biotechnology"),
        r("AI_RISK", 6.0, industry="Consulting Services"),
        r("ROIC_SLIDES", 6.0, roic=(17, 18, 19, 20)),
        r("MARGIN_SLIDES", 6.0, margin=(0.18, 0.20, 0.21, 0.22)),
        r("MARGIN_DROPS", 6.0, margin=(0.22, 0.18, 0.21, 0.19)),  # down 14% in a year
        r("REVENUE_DIP", 6.0, revenue=(300, 320, 200, 100)),
        r("OE_FALLS", 6.0, owner=(100, 110, 120, 130)),
        r("ERRATIC_OE", 6.0, owner=(130, 20, 110, 100)),
        r("DILUTES", 6.0, shares=(12, 10, 10, 11)),
        r("ISSUED_SINCE", 6.0, **{"Ordinary Shares Number_TTM": 11}),
        r("RECESSION_CUT", 6.0, Recession_Dividend_Fall=-30.0),  # its dividend fell 30% in a past recession
        r("RECESSION_HELD", 6.5, Recession_Dividend_Fall=-10.0),  # a 10% dip passes
        r("LEVERED", 6.0, NetDebt_to_EBIT=6.0, Interest_Coverage=3.0),
        r("LOW_RANK", 6.0, In_Best_Fraction=False),
    ])  # fmt: skip
    assert operating_list(df)["Ticker"].tolist() == [
        "A",
        "B",
        "FLAT_HIGH_ROIC",
        "FLAT_MARGIN",
        "ROIC_SLIPS",
        "ONE_FLAT_OE",
        "PACKAGING",
        "RECESSION_HELD",
    ]


def test_dividend_growers_need_a_decade_without_a_cut_growth_cover_and_a_lasting_business():
    from fq.report.screens import dividend_grower_list

    def r(ticker, rank, **over):
        row = _twenty_year_row(ticker, rank, roic=(15, 15, 14, 14))
        return row | {"Dividend_Unbroken_Years": 12.0, "Dividend_Cut_Recent": False, "Regular_DPS_Growth_10Y": 5.0,
                      "Recession_Dividend_Fall": -3.0, "Dividend_Yield": 3.0, "Owner_Yield": 8.0,
                      "EPV_Per_Share_Yield": 8.0, "ROIC_Norm": 15.0, "NetDebt_to_EBIT": 1.0, "Net_Debt": 10.0,
                      "Common Stock Equity_Y0": 100.0} | over  # fmt: skip

    df = pd.DataFrame([
        r("B", 2.0),
        r("A", 1.0),
        r("BANK", 3.0, Financial_Mode=True, Equity_to_Assets=9.0),  # ROE stands for ROIC
        r("YOUNG", 4.0, Dividend_Unbroken_Years=9.0),
        r("CUT_NOW", 4.0, Dividend_Cut_Recent=True),
        r("RECESSION_CUT", 4.0, Recession_Dividend_Fall=-30.0),
        r("LOW_YIELD", 4.0, Dividend_Yield=2.4),
        r("PAYS_IT_ALL", 4.0, Dividend_Yield=7.0),  # above 80% of what a share earns after interest
        r("PAID_FROM_GROWTH", 4.0, EPV_Per_Share_Yield=3.5),  # 86% at zero growth, 38% of the owner yield
        r("FLAT_DIVIDEND", 4.0, Regular_DPS_Growth_10Y=2.9),
        r("LOW_ROIC", 4.0, ROIC_Norm=9.0),
        r("CYCLICAL", 4.0, Cyclical=True),
        r("AI_RISK", 4.0, industry="Consulting Services"),
        r("INDEBTED", 4.0, NetDebt_to_EBIT=4.0, Interest_Coverage=3.0),
        r("THIN_BANK", 4.0, Financial_Mode=True, Equity_to_Assets=4.0),
        r("LOW_RANK", 4.0, In_Best_Fraction=False),
        r("NO_RECORD", 4.0, Dividend_Unbroken_Years=np.nan),
    ])  # fmt: skip
    assert dividend_grower_list(df)["Ticker"].tolist() == ["A", "B", "BANK"]


def test_dividend_growers_take_a_bank_ranked_apart_on_its_own_kinds_best():
    from fq.report.screens import dividend_grower_list

    def r(ticker, rank, **over):
        row = _twenty_year_row(ticker, rank, roic=(15, 15, 14, 14))
        return row | {"Dividend_Unbroken_Years": 12.0, "Regular_DPS_Growth_10Y": 5.0, "Dividend_Yield": 3.0,
                      "Owner_Yield": 8.0, "ROIC_Norm": 15.0, "NetDebt_to_EBIT": 1.0, "Net_Debt": 10.0,
                      "Common Stock Equity_Y0": 100.0, "Rank_Group": "ranked", "In_Best_Fraction": True,
                      "Kind_Rank": np.nan, "In_Kind_Best": False} | over  # fmt: skip

    apart = {"Rank": np.nan, "In_Best_Fraction": False, "Rank_Group": "financial", "Business_Model": "bank",
             "Financial_Mode": True, "Equity_to_Assets": 9.0}  # fmt: skip
    df = pd.DataFrame([
        r("A", 1.0),
        r("B", 9.0),  # ninth of ten ranked: after a bank first among its kind
        r("BANK", 4.0, Kind_Rank=1.0, In_Kind_Best=True, **apart),
        r("BANK_LOW", 4.0, Kind_Rank=2.0, In_Kind_Best=False, **apart),  # not in its kind's best fraction
    ])  # fmt: skip
    df = pd.concat([df, pd.DataFrame({"Ticker": [f"X{i}" for i in range(8)], "Rank": np.arange(2.0, 10.0)})])
    assert dividend_grower_list(df.reset_index(drop=True))["Ticker"].tolist() == ["A", "BANK", "B"]


def test_compounders_take_either_route_once_in_rank_order():
    from fq.report import screens as sc

    df = pd.DataFrame({"Ticker": ["S", "F", "BOTH", "NONE"], "Rank": [3.0, 2.0, 1.0, 4.0]})
    steady, fast = sc.operating_list, sc.fast_grower_list
    try:
        sc.operating_list = lambda d, n=20: d[d["Ticker"].isin(["S", "BOTH"])].head(n)
        sc.fast_grower_list = lambda d, n=20: d[d["Ticker"].isin(["F", "BOTH"])].head(n)
        assert sc.compounder_list(df)["Ticker"].tolist() == ["BOTH", "F", "S"]
    finally:
        sc.operating_list, sc.fast_grower_list = steady, fast


def test_fast_growers_need_fast_smooth_owner_earnings_rising_margins_and_a_yield():
    from fq.report.screens import fast_grower_list

    r = lambda *a, **k: _twenty_year_row(*a, **({"Per_Share_Compounding": 16.0} | k))  # noqa: E731
    df = pd.DataFrame([
        r("B", 2.0),
        r("A", 1.0),
        r("LOW_ROIC", 3.0, ROIC_Norm=9.0),  # no return floor: fast growth already shows the returns
        r("BANK", 4.0, Financial_Mode=True),  # operating companies only
        r("BIOTECH", 4.5, industry="Biotechnology"),  # excluded
        r("SLOW", 5.0, Owner_Earnings_Compounding=14.0),
        r("SALES_SLOW", 5.0, Per_Share_Compounding=14.0),  # sales must compound at 15%/yr too
        r("DEAR", 5.0, Owner_Yield=3.9),
        r("MARGIN_FLAT", 5.0, margin=(0.2, 0.2, 0.2, 0.2)),
        r("MARGIN_FALLS", 5.0, margin=(0.18, 0.21, 0.20, 0.19)),
        r("MARGIN_LOST", 5.0, Gross_Margin_Held=-5.0),
        r("REVENUE_DIP", 5.0, revenue=(300, 320, 200, 100)),
        r("ERRATIC_OE", 5.0, owner=(130, 20, 110, 100)),
        r("LOSS", 5.0, owner=(130, 120, 110, -5)),
        r("DILUTES", 5.0, shares=(12, 10, 10, 11)),
        r("ISSUED_SINCE", 5.0, **{"Ordinary Shares Number_TTM": 11}),
        r("CYCLICAL", 5.0, Cyclical=True),
        r("AI_RISK", 5.0, industry="Consulting Services"),
        r("RECESSION_CUT", 5.0, Recession_Dividend_Fall=-30.0),  # its dividend fell 30% in a past recession
        r("LEVERED", 5.0, NetDebt_to_EBIT=6.0, Interest_Coverage=3.0),
        r("LOW_RANK", 5.0, In_Best_Fraction=False),
    ])  # fmt: skip
    assert fast_grower_list(df)["Ticker"].tolist() == ["A", "B", "LOW_ROIC"]  # A, B: no dividend record, passes
