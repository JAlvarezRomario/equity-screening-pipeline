"""Graham's net current asset value, Buffett's reinvestment runway in US terms and pricing power through inflation
(shown, and tested by the lists; never scored)."""

import numpy as np
import pandas as pd

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.metrics.quality import _gross_margin_held


def test_ncav_is_current_assets_less_every_liability_per_share_and_blank_for_a_bank_or_homebuilder(cfg):
    raw = make_raw(cfg, n=3)
    raw.loc[1, "industry"] = "Banks - Regional"
    raw.loc[2, "industry"] = "Residential Construction"
    for row in (0, 1, 2):
        for y in cfg.fiscal_years:
            raw.loc[row, f"Current Assets_{y}"] = 6e9
            raw.loc[row, f"Total Liabilities Net Minority Interest_{y}"] = 2e9
            raw.loc[row, f"Minority Interest_{y}"] = 0.5e9
    df = calculate_metrics(raw, cfg)
    shares = raw.loc[0, "Ordinary Shares Number_TTM"]
    expected = (6e9 - 2e9 - 0.5e9) / shares / raw.loc[0, "currentPrice"] * 100
    assert np.isclose(df.loc[0, "NCAV_to_Price"], expected)
    assert df.loc[1, "Financial_Mode"] and pd.isna(df.loc[1, "NCAV_to_Price"])
    assert pd.isna(df.loc[2, "NCAV_to_Price"])


def test_ncav_is_blank_without_current_assets(cfg):
    df = calculate_metrics(make_raw(cfg, n=1), cfg)
    assert pd.isna(df.loc[0, "NCAV_to_Price"])


def _margins(*rows):
    return pd.DataFrame(list(rows), columns=["TTM", "Y0", "Y1", "Y2", "Y3"], dtype=float)


def test_gross_margin_held_compares_the_newest_margin_with_the_peak_inflation_year():
    margin = _margins(
        [45, 44, 42, 41, 40],  # rebuilt since the shock: +5
        [38, 38, 39, 40, 48],  # fell since the shock: -10
        [45, 44, 42, 41, 40],  # no year with a cost shock: blank
        [45, np.nan, np.nan, np.nan, 40],  # two margins: blank
        [45, 44, 42, 41, 40],  # a bank: blank
    )
    inflation = {"TTM": 0.027, "Y0": 0.029, "Y1": 0.041, "Y2": 0.08}
    df = pd.DataFrame({f"Inflation_US_{p}": [v] * 5 for p, v in inflation.items()})
    df["Inflation_US_Y3"] = [0.047, 0.047, 0.047, 0.047, 0.047]
    df.loc[2, [f"Inflation_US_{p}" for p in (*inflation, "Y3")]] = 0.025
    df["Financial_Mode"] = [False, False, False, False, True]
    held = _gross_margin_held(df, margin)
    assert held.iloc[0] == 45 - 41  # Y2 had the highest inflation (8%), never the calendar year
    assert held.iloc[1] == 38 - 40
    assert held.iloc[2:].isna().all()


def test_incremental_roic_is_in_us_terms(cfg):
    raw = make_raw(cfg, n=1)
    for y in cfg.fiscal_years:  # invested capital grows, so the capital added is measured
        raw.loc[0, f"Stockholders Equity_{y}"] = raw.loc[0, f"Total Revenue_{y}"] * 0.8
    usd = calculate_metrics(raw, cfg)
    local = raw.copy()
    local["financialCurrency"] = local["currency"] = "EGP"  # 20% inflation in the fallback table, far above US
    egp = calculate_metrics(local, cfg)
    excess = egp.loc[0, "Excess_Inflation_Y0"]
    nominal = usd.loc[0, "Incremental_ROIC"]
    assert usd.loc[0, "Excess_Inflation_Y0"] == 0 and excess > 0.1
    assert np.isclose(egp.loc[0, "Incremental_ROIC"], ((1 + nominal / 100) / (1 + excess) - 1) * 100)
