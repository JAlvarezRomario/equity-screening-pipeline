"""Tests for the "Ascending ROIC" highlight list (`report.screens.ascending_roic_list`)."""

import numpy as np
import pandas as pd

from fq.report import highlights as hl
from fq.report import screens as sc


def _row(
    ticker: str,
    rank: float = 10.0,
    roic=(24.0, 20.0, 16.0, 12.0),
    margin=(22.0, 21.0, 20.0, 19.0),
    revenue=(400.0, 300.0, 200.0, 100.0),
    **over,
) -> dict:
    """A passing Ascending ROIC row before `over`."""
    out = {
        "Ticker": ticker,
        "Rank": rank,
        "ROIC_Norm": 20.0,
        "ROIC_Trend": 8.0,
        "Incremental_ROIC": 25.0,
        "EPV_Per_Share_Yield": 6.0,
        "Owner_Yield": 6.5,
        "Treasury_Yield": 4.2,
        "Sovereign_Spread": 0.0,
        "NetDebt_to_EBIT": 1.0,
        "Interest_Coverage": 10.0,
        "Cyclical": False,
        "Financial_Mode": False,
        "Business_Model": "operating",
        "Rank_Group": "operating",
        "Peak_Margin_Ratio": 1.0,
        "industry": "Packaged Foods",
        "Per_Share_Compounding": 10.0,
    }
    for p, val in zip(("Y0", "Y1", "Y2", "Y3"), roic, strict=True):
        out[f"ROIC_{p}"] = val
    for p, val in zip(("Y0", "Y1", "Y2", "Y3"), margin, strict=True):
        out[f"operatingMargins_{p}"] = val
    for p, val in zip(("Y0", "Y1", "Y2", "Y3"), revenue, strict=True):
        out[f"Total Revenue_{p}"] = val
    out.update(over)
    return out


def test_ascending_roic_requires_strict_roic_ascent_and_allows_small_margin_revenue_dips():
    df = pd.DataFrame(
        [
            _row("PASS_STRICT", rank=1),
            _row("PASS_DIP", rank=2, margin=(20.0, 18.5, 20.0, 19.0), revenue=(350.0, 280.0, 300.0, 250.0)),
            _row("FLAT_ROIC", rank=3, roic=(20.0, 20.0, 16.0, 12.0)),
            _row("DIP_ROIC", rank=4, roic=(24.0, 18.0, 19.0, 12.0)),
            _row("NEG_ROIC", rank=5, roic=(-1.0, -2.0, -3.0, -4.0), ROIC_Norm=-2.0),
            _row("BIG_MARGIN_DIP", rank=6, margin=(22.0, 16.0, 20.0, 19.0)),
            _row("BIG_REV_DIP", rank=7, revenue=(400.0, 200.0, 300.0, 100.0)),
        ]
    )
    out = sc.ascending_roic_list(df, 20)
    assert out["Ticker"].tolist() == ["PASS_STRICT", "PASS_DIP"]


def test_ascending_roic_checks_reinvestment_yield_cyclical_and_industries():
    df = pd.DataFrame(
        [
            _row("OK", rank=1),
            _row("NO_REINVEST", rank=2, Incremental_ROIC=np.nan),
            _row("LOW_REINVEST", rank=3, Incremental_ROIC=10.0),
            _row("LOW_YIELD", rank=4, EPV_Per_Share_Yield=4.0),
            _row("SPREAD_FAIL", rank=5, EPV_Per_Share_Yield=6.0, Sovereign_Spread=3.0),
            _row("CYCLICAL", rank=6, Cyclical=True),
            _row("BANK", rank=7, Financial_Mode=True, Business_Model="bank", Rank_Group="financial"),
            _row("BIOTECH", rank=8, industry="Biotechnology"),
            _row("ASSET_MGR", rank=9, industry="Asset Management"),
            _row("LEVERED", rank=10, NetDebt_to_EBIT=5.0, Interest_Coverage=2.0),
            _row("BREAKING", rank=11, Revenue_Change_Recent=-20.0),
        ]
    )
    out = sc.ascending_roic_list(df, 20)
    assert out["Ticker"].tolist() == ["OK"]


def test_ascending_roic_tab_is_registered_in_sources():
    assert "Ascending ROIC" in hl.TAB
    assert hl.TAB["Ascending ROIC"].group == "sources"
    assert "Ascending ROIC" in hl.THE_LIST_SOURCES
