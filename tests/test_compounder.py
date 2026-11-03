"""The Compounder flag ("Mastercard-like", every trait at once) and owner earnings per share compounding."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.metrics.common import TTM_SHARED_SPAN, _ttm_span, adjusted_r2
from fq.ranking import apply_dual_ranking, compounder
from fq.report.screens import compounder_list

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


def test_the_list_is_cheapest_first_with_no_value_floor(cfg):
    df = _rows(DEAR={"Value_Score": 10.0}, CHEAP={"Value_Score": 85.0}, MID={"Value_Score": 40.0}, NOT={})
    df["Compounder"] = [True, True, True, False]
    assert compounder_list(df)["Ticker"].tolist() == ["CHEAP", "MID", "DEAR"]


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


def test_better_every_year_needs_every_measure_rising_and_shares_not_rising():
    from fq.report.screens import improver_list

    def company(
        ticker,
        rank,
        oe=(130, 120, 110),
        rev=(300, 280, 260),
        roic=(15, 14, 13),
        margin=(12, 11, 10),
        shares=(10, 10, 11),
        financial=False,
    ):
        row = {"Ticker": ticker, "Rank": rank, "Financial_Mode": financial}
        for i in range(3):
            row |= {
                f"Owner_Earnings_Y{i}": oe[i],
                f"Total Revenue_Y{i}": rev[i],
                f"ROIC_Y{i}": roic[i],
                f"operatingMargins_Y{i}": margin[i],
                f"Ordinary Shares Number_Y{i}": shares[i],
            }
        return row

    df = pd.DataFrame(
        [
            company("B", 2.0),
            company("A", 1.0),
            company("BANK", 3.0, margin=(5, 9, 12), financial=True),  # margin not applicable
            company("OE_FLAT", 4.0, oe=(120, 120, 110)),
            company("REV_DIP", 5.0, rev=(300, 250, 260)),
            company("ROIC_DIP", 6.0, roic=(15, 16, 13)),
            company("MARGIN_DIP", 7.0, margin=(12, 13, 10)),
            company("DILUTING", 8.0, shares=(11, 10, 10)),
            company("OE_PER_SHARE_DIP", 9.0, oe=(130, 120, 110), shares=(13, 10, 10)),
            company("TWO_YEARS", 10.0, oe=(130, 120, np.nan)),
            company("LOSS", 11.0, oe=(-1, -2, -3)),
        ]
    )
    assert improver_list(df)["Ticker"].tolist() == ["A", "B", "BANK"]


def test_dividend_carry_needs_untaxed_covered_rising_dividends_in_the_best_quintile():
    from fq.report.screens import dividend_carry_list

    def company(ticker, rank, div=(6, 5, 4), oe=(9, 8, 7), **changes):
        row = {"Ticker": ticker, "Rank": rank, "Dividend_WHT": 0.0, "Dividend_Yield": 6.0, "Owner_Yield": 9.0,
               "Cyclical": False, "Peak_Margin_Ratio": 1.0}  # fmt: skip
        for i in range(3):
            row |= {f"Cash Dividends Paid_Y{i}": -div[i], f"Owner_Earnings_Y{i}": oe[i],
                    f"Ordinary Shares Number_Y{i}": 1.0}  # fmt: skip
        return row | changes

    df = pd.DataFrame(
        [
            company("B", 2.0),
            company("A", 1.0, div=(6, 7, 4)),  # one dip allowed
            company("TAXED", 3.0, Dividend_WHT=15.0),
            company("LOW_YIELD", 3.0, Dividend_Yield=4.0),
            company("UNCOVERED", 3.0, Owner_Yield=5.0),
            company("DIV_FALLING", 3.0, div=(4, 5, 6)),
            company("OE_FALLING", 3.0, oe=(7, 8, 9)),
            company("CYCLE", 3.0, Cyclical=True),
            company("OUTSIDE", 30.0),
        ]
    )
    assert dividend_carry_list(df)["Ticker"].tolist() == ["A", "B"]


def test_nectar_needs_every_trait_and_leaves_out_the_avoided_industries():
    from fq.report.screens import nectar_list

    def row(ticker, **over):
        out = {"Ticker": ticker, "Rank": len(ticker), "Owner_Yield": 12.0, "Cyclical": False,
               "industry": "Packaged Foods", "ROIC_Variability": 0.05, "In_Best_Fraction": True, "Net_Cash_to_Cap": 5.0,
               "Owner_Earnings_Compounding": 5.0, "ROIC_Norm": 13.0}  # fmt: skip
        for p, owner, revenue, shares in zip(("Y0", "Y1", "Y2", "Y3"), (130, 120, 110, 100), (400, 300, 200, 100),
                                             (10, 10, 10, 11), strict=True):  # fmt: skip
            out |= {
                f"Owner_Earnings_{p}": owner,
                f"Total Revenue_{p}": revenue,
                f"Ordinary Shares Number_{p}": shares,
                f"ROIC_{p}": owner / 10,
            }
        return out | over

    df = pd.DataFrame([
        row("KEEP"),
        row("DEARER", Owner_Yield=5.9),
        row("CYCLICAL", Cyclical=True),
        row("BIOTECH", industry="Biotechnology"),
        row("AIRISK", industry="Consulting Services"),
        row("BUMPY", ROIC_Variability=0.2),
        row("LOWRANK", In_Best_Fraction=False),
        row("ROICDOWN", **{"ROIC_Y0": 11.5}),
        row("OEDIP", **{"Owner_Earnings_Y1": 135}),
        row("FLATYEAR", **{"Owner_Earnings_Y1": 130}),
        row("NODEBTNOT", Net_Cash_to_Cap=-1.0),
        row("BANKNOCASH", Net_Cash_to_Cap=None),
        row("LOWERYIELD", Owner_Yield=6.5),
        row("ISSUEDSINCE", **{"Ordinary Shares Number_TTM": 11}),
        row("STABLEHIGHROIC", ROIC_Norm=25.0, **{"ROIC_Y0": 10.0}),
        row("TWOFLATS", **{"Owner_Earnings_Y1": 130, "Owner_Earnings_Y2": 130}),
        row("REVDOWN", **{"Total Revenue_Y0": 250}),
        row("DILUTES", **{"Ordinary Shares Number_Y0": 12}),
        row("ERRATIC", **{"Owner_Earnings_Y1": 20}),
        row("SHRINKS", **{"Owner_Earnings_Y0": 90, "Owner_Earnings_Y1": 100, "Owner_Earnings_Y2": 110,
                          "Owner_Earnings_Y3": 120}),
    ])  # fmt: skip
    assert nectar_list(df)["Ticker"].tolist() == ["KEEP", "FLATYEAR", "LOWERYIELD", "STABLEHIGHROIC"]
