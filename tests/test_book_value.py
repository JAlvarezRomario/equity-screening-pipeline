"""Book value, minority-adjusted tangible book and split adjustment (generic for every company)."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.fetch import _apply_splits, split_factors
from fq.metrics import calculate_metrics
from fq.metrics.capital import _common_book_and_tangible


def frame(**cols):
    df = pd.DataFrame({k: [v] for k, v in cols.items()})
    return lambda name: pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series([np.nan])


def test_holding_company_tangible_book_is_minority_adjusted():
    # Brookfield Corp, Dec-25 (USD bn), as served by Yahoo.
    c = frame(**{
        "Stockholders Equity": 47.9, "Common Stock Equity": 43.8, "Preferred Stock Equity": 4.1,
        "Total Equity Gross Minority Interest": 166.2, "Minority Interest": 118.3,
        "Goodwill And Other Intangible Assets": 81.9, "Tangible Book Value": -38.1,
    })  # fmt: skip
    book, tangible = _common_book_and_tangible(c)
    assert book.iloc[0] == 43.8  # common equity, preferred excluded
    assert 19 < tangible.iloc[0] < 21  # 43.8 - 29% x 81.9, instead of Yahoo's -38.1


def test_without_minorities_tangible_equals_yahoo():
    c = frame(**{"Stockholders Equity": 100.0, "Common Stock Equity": 100.0, "Tangible Book Value": 70.0,
                 "Goodwill And Other Intangible Assets": 30.0})  # fmt: skip
    book, tangible = _common_book_and_tangible(c)
    assert (book.iloc[0], tangible.iloc[0]) == (100.0, 70.0)


def test_intangibles_recovered_from_yahoo_tangible_book_and_missing_rows():
    c = frame(**{"Stockholders Equity": 100.0, "Tangible Book Value": 60.0})
    book, tangible = _common_book_and_tangible(c)
    assert (book.iloc[0], tangible.iloc[0]) == (100.0, 60.0)
    book, tangible = _common_book_and_tangible(frame(**{"Stockholders Equity": 100.0}))
    assert book.iloc[0] == 100.0 and pd.isna(tangible.iloc[0])  # unknown intangibles: no guess


def test_chart_triggers_on_book_value_alone(cfg):
    raw = make_raw(cfg, n=2)
    for y in cfg.fiscal_years:  # big book, but intangibles wipe out tangible book
        raw.loc[0, f"Stockholders Equity_{y}"] = 4e9
        raw.loc[0, f"Goodwill And Other Intangible Assets_{y}"] = 5e9
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "NAV_to_Price"] < 0 < 40 <= df.loc[0, "BV_to_Price"]
    assert bool(df.loc[0, "Show_NAV_Chart"]) and not bool(df.loc[1, "Show_NAV_Chart"])


SPLITS = pd.Series([1.5], index=pd.DatetimeIndex(["2025-10-10"], tz="America/Toronto"))
ENDS = {"TTM": "2026-06-30", "Y0": "2025-12-31", "Y1": "2024-12-31", "Y2": "2023-12-31"}


def test_split_factors_apply_only_before_the_split():
    assert split_factors(SPLITS, ENDS) == {"Y1": 1.5, "Y2": 1.5}
    assert split_factors(pd.Series(dtype=float), ENDS) == {}


def test_unadjusted_share_counts_are_restated():
    row = {"sharesOutstanding": 2.23e9, "Ordinary Shares Number_Y0": 2.2e9, "Ordinary Shares Number_Y1": 1.5e9,
           "Diluted Average Shares_Y1": 1.52e9}  # fmt: skip
    assert _apply_splits(row, {"Y1": 1.5}, "BN.TO") == ["Y1"]
    assert row["Ordinary Shares Number_Y1"] == 2.25e9
    assert row["Diluted Average Shares_Y1"] == pytest.approx(2.28e9)  # every share count on today's basis


def test_already_adjusted_share_counts_are_left_alone():
    row = {"sharesOutstanding": 2.23e9, "Ordinary Shares Number_Y1": 2.2e9}
    assert _apply_splits(row, {"Y1": 1.5}, "X") == []
    assert row["Ordinary Shares Number_Y1"] == 2.2e9


def _asset_df() -> pd.DataFrame:
    base = {
        "Ranked_Model": True,
        "Data_Suspect": False,
        "Insufficient_Data": False,
        "Industry_Excluded_Kept": False,
        "Peak_Margin": False,
        "Distress": False,
        "Earnings_Collapse": False,
        "Financial_Mode": False,
        "Business_Model": "operating",
        "Value_Score": 75.0,
        "Quality_Score": 55.0,
        "EPV_Per_Share_Yield": 8.0,
        "ROIC_Norm": 10.0,
        "FCF_Positive_Share": 100.0,
        "NetDebt_to_EBIT": 0.5,
        "Net_Cash_to_Cap": 10.0,
        "Share_Reduction_Rate": 1.5,
        "Per_Share_Compounding": 4.0,
        "Book_Per_Share_Compounding": 6.0,
        "NAV_to_Price": 115.0,
        "BV_to_Price": 120.0,
        "Equity_to_Assets": 50.0,
    }
    return pd.DataFrame([base.copy() for _ in range(6)], index=range(6))


def test_asset_play_qualifies_above_100_percent_or_growing_into_it(cfg):
    from fq.ranking import asset_play

    df = _asset_df()
    # Row 0: NAV/Price = 115%, BV/Price = 120%, compounding 1% -> qualifies now
    df.loc[0, ["NAV_to_Price", "BV_to_Price", "Book_Per_Share_Compounding"]] = [115.0, 120.0, 1.0]
    # Row 1: NAV/Price = 82%, compounding 9%/yr -> 106% in 3 years: qualifies (growing into it)
    df.loc[1, ["NAV_to_Price", "Book_Per_Share_Compounding"]] = [82.0, 9.0]
    # Row 2: NAV/Price = 72%, compounding 6%/yr -> 86% in 3 years: rejected
    df.loc[2, ["NAV_to_Price", "Book_Per_Share_Compounding"]] = [72.0, 6.0]
    # Row 3: NAV/Price = 120%, but shrinking book (-2%/yr) -> rejected
    df.loc[3, ["NAV_to_Price", "Book_Per_Share_Compounding"]] = [120.0, -2.0]
    # Row 4: NAV/Price = 115%, but BV/Price = 85% -> rejected (BV under 100%)
    df.loc[4, ["NAV_to_Price", "BV_to_Price", "Book_Per_Share_Compounding"]] = [115.0, 85.0, 1.0]
    # Row 5: NAV/Price = 60%, compounding 25%/yr -> under the 70% floor: rejected
    df.loc[5, ["NAV_to_Price", "Book_Per_Share_Compounding"]] = [60.0, 25.0]

    flags, scores = asset_play(df, cfg)
    assert list(flags) == [True, True, False, False, False, False]
    assert scores.iloc[0] > 0 and scores.iloc[1] > 0 and pd.isna(scores.iloc[2])


def test_asset_play_strips_goodwill_and_delevers_net_debt(cfg):
    from fq.ranking import asset_play

    df = _asset_df()
    # Row 0: Goodwill trap (BV/Price = 140%, but tangible NAV/Price = -25%) -> rejected
    df.loc[0, ["BV_to_Price", "NAV_to_Price"]] = [140.0, -25.0]
    # Row 1: Raw NAV/Price = 120%, but net debt is 70% of market cap (> 0.5 * 120%) -> rejected
    df.loc[1, ["NAV_to_Price", "Net_Cash_to_Cap"]] = [120.0, -70.0]
    # Row 2: Raw NAV/Price = 110%, net debt = 40% of market cap -> de-levered NAV = (110+40)/(100+40) = 107.1% >= 100%
    df.loc[2, ["NAV_to_Price", "Net_Cash_to_Cap"]] = [110.0, -40.0]
    # Row 3: High NetDebt/EBIT (> 3x) -> rejected
    df.loc[3, ["NAV_to_Price", "NetDebt_to_EBIT"]] = [130.0, 4.2]

    flags, _ = asset_play(df, cfg)
    assert list(flags.iloc[:4]) == [False, False, True, False]


def test_asset_play_accepts_high_debt_when_interest_well_covered(cfg):
    from fq.ranking import asset_play

    df = _asset_df()
    # High NetDebt/EBIT (5.0 > 3.0) but strong interest coverage (3.0 >= 2.5) with net debt <= 50% of NAV -> qualifies
    df.loc[0, ["NAV_to_Price", "NetDebt_to_EBIT", "Interest_Coverage", "Net_Cash_to_Cap"]] = [120.0, 5.0, 3.0, -25.0]
    # High NetDebt/EBIT (5.0 > 3.0) and poor interest coverage (1.8 < 2.5) -> rejected
    df.loc[1, ["NAV_to_Price", "NetDebt_to_EBIT", "Interest_Coverage", "Net_Cash_to_Cap"]] = [120.0, 5.0, 1.8, -25.0]

    flags, _ = asset_play(df, cfg)
    assert flags.iloc[0] and not flags.iloc[1]


def test_asset_play_favours_net_cash_and_buybacks_and_renders_in_html(cfg):
    from fq.ranking import apply_dual_ranking, asset_play
    from fq.report.html import build_site
    from fq.report.screens import asset_play_list

    df = _asset_df()
    df["Ticker"] = [f"T{i}" for i in range(len(df))]
    # Row 0: moderate net cash & buybacks
    df.loc[0, ["NAV_to_Price", "Net_Cash_to_Cap", "Share_Reduction_Rate", "Book_Per_Share_Compounding"]] = [
        110.0,
        5.0,
        0.5,
        6.0,
    ]
    # Row 1: same NAV & book compounding, but strong net cash (45%) & heavy buybacks (6%/yr)
    df.loc[1, ["NAV_to_Price", "Net_Cash_to_Cap", "Share_Reduction_Rate", "Book_Per_Share_Compounding"]] = [
        110.0,
        45.0,
        6.0,
        6.0,
    ]
    # Row 2: passive bank (0% share reduction) -> excluded
    df.loc[2, ["Financial_Mode", "Business_Model", "Equity_to_Assets", "Share_Reduction_Rate"]] = [
        True,
        "bank",
        9.0,
        0.0,
    ]
    # Row 3: cannibal bank (3%/yr share reduction, Equity/Assets 10%) -> excluded since Nov 5, 2026: banks are judged
    # in their own list, never on Below book
    df.loc[3, ["Financial_Mode", "Business_Model", "Equity_to_Assets", "Share_Reduction_Rate"]] = [
        True,
        "bank",
        10.0,
        3.0,
    ]
    # Row 4: a holding at 80% tangible book but 110% book with goodwill -> included, judged on book
    df.loc[4, ["Financial_Mode", "Business_Model", "NAV_to_Price", "BV_to_Price", "Equity_to_Assets"]] = [
        True,
        "holding",
        80.0,
        110.0,
        40.0,
    ]
    df.loc[4, "Book_Per_Share_Compounding"] = 1.0

    flags, scores = asset_play(df, cfg)
    assert bool(flags.loc[0]) and bool(flags.loc[1]) and not bool(flags.loc[2]) and not bool(flags.loc[3])
    assert bool(flags.loc[4])
    assert scores.loc[1] > scores.loc[0]
    df["Asset_Play"], df["Asset_Play_Score"] = flags, scores
    ordered = asset_play_list(df)
    assert list(ordered["Ticker"]).index("T1") < list(ordered["Ticker"]).index("T0")

    raw = make_raw(cfg, n=4)
    for y in cfg.fiscal_years:
        raw.loc[0, f"Tangible Book Value_{y}"] = 8e9
        raw.loc[0, f"Stockholders Equity_{y}"] = 8e9
        raw.loc[0, f"Common Stock Equity_{y}"] = 8e9
    ranked = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    assert "Asset_Play" in ranked.columns and "Effective_NAV_to_Price" in ranked.columns
    html = build_site(ranked, cfg).read_text(encoding="utf-8")
    assert 'data-panel="hl-asset-plays">Below book</button>' in html
    assert 'id="hl-asset-plays"' in html
