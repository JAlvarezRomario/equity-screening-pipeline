"""Fixes from an external review: FX on price refresh, leases, splits vs ADRs, check order, growth spend."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.evaluate import evaluate_snapshot
from fq.fetch import _align_share_basis, _restate_fx, _statement_columns
from fq.metrics import calculate_metrics, growth_profile
from fq.ranking import dislocation
from fq.settings import Settings


def test_price_refresh_moves_statements_to_todays_rate(cfg):
    raw = make_raw(cfg, n=2).assign(financialCurrency="EUR", currency="EUR", Fin_Rate=1.0)
    for p in cfg.fiscal_years:
        raw[f"FX_Hist_{p}"] = 1.0
    raw.loc[1, "Fin_Rate"] = np.nan  # cached before the rate was stored
    before = raw.copy()
    assert _restate_fx(raw, 0, {"EUR": 1.1}, cfg)
    assert raw.loc[0, "Total Revenue_Y0"] == pytest.approx(before.loc[0, "Total Revenue_Y0"] * 1.1)
    assert raw.loc[0, "Total Debt_TTM"] == pytest.approx(before.loc[0, "Total Debt_TTM"] * 1.1)
    assert raw.loc[0, "FX_Hist_Y2"] == pytest.approx(1 / 1.1)  # USD of the time unchanged
    assert raw.loc[0, "Ordinary Shares Number_Y0"] == before.loc[0, "Ordinary Shares Number_Y0"]
    assert raw.loc[0, "Fin_Rate"] == 1.1
    assert not _restate_fx(raw, 1, {"EUR": 1.1}, cfg)
    assert raw.loc[1].equals(before.loc[1])


def _leased(cfg, ticker="EUCO.MC"):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "Ticker"] = ticker
    for p in cfg.fiscal_years:
        raw.loc[0, f"Capital Lease Obligations_{p}"] = 2e8
        raw.loc[0, f"Current Capital Lease Obligation_{p}"] = 4e7
    return raw


def test_ifrs_ttm_without_current_leases_borrows_the_annual_share(cfg):
    raw = _leased(cfg)
    plain = calculate_metrics(raw, cfg)
    raw.loc[0, "Current Capital Lease Obligation_TTM"] = np.nan  # only the annual report splits it
    raw.loc[0, "Capital Lease Obligations_TTM"] = 3e8
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Lease_Cost_TTM"] == pytest.approx(3e8 * 4e7 / 2e8)
    assert df.loc[0, "Net_Debt"] == pytest.approx(plain.loc[0, "Net_Debt"] - 1e8)  # TTM leases left debt too
    assert bool(df.loc[0, "Lease_Adjusted"])


def test_ttm_quarter_without_a_lease_row_takes_the_annual_leases(cfg):
    raw = _leased(cfg, ticker="USCO")  # US GAAP
    plain = calculate_metrics(raw, cfg)
    raw.loc[0, "Capital Lease Obligations_TTM"] = np.nan  # not broken out in the quarter
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Net_Debt"] == pytest.approx(plain.loc[0, "Net_Debt"])  # Y0's leases left debt
    assert bool(df.loc[0, "Lease_Adjusted"])
    raw.loc[0, "Capital Lease Obligations_TTM"] = 0.0  # reported as none: nothing to take out
    assert calculate_metrics(raw, cfg).loc[0, "Net_Debt"] == pytest.approx(plain.loc[0, "Net_Debt"] + 2e8)


def test_leases_kept_as_debt_are_not_also_charged_to_fcf(cfg):
    raw = _leased(cfg)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Capital Lease Obligations_{p}"] = np.nan  # current part known, total not
    df = calculate_metrics(raw, cfg)
    assert df["Lease_Cost_Y0"].isna().all() and not df["Lease_Adjusted"].any()


def test_split_after_the_last_report_is_not_an_adr_ratio():
    row = {"currentPrice": 25.0, "marketCap": 1e10, "Ordinary Shares Number_TTM": 1e8}  # 4:1 split since
    assert _align_share_basis(row, ["TTM"], "X", {"TTM": 4.0}) is None
    assert row["Ordinary Shares Number_TTM"] == 1e8  # left to `_apply_splits`
    assert _align_share_basis(row, ["TTM"], "ADR") == pytest.approx(0.25)  # no split: an ADR ratio


def test_sparse_newest_quarter_is_not_mixed_into_the_balance_sheet():
    q = pd.DataFrame(
        {
            "Total Assets": [np.nan, 100.0, 90.0],
            "Total Debt": [9.0, 7.0, 6.0],
            "Cash And Cash Equivalents": [np.nan, 3, 2],
        },
        index=pd.to_datetime(["2026-03-31", "2025-12-31", "2025-09-30"]),
    ).T
    out, ends, _ = _statement_columns(
        pd.DataFrame(), q, pd.DataFrame(), ["Total Assets", "Total Debt", "Cash And Cash Equivalents"],
        Settings(), 1.0, "latest",
    )  # fmt: skip
    assert (out["Total Debt_TTM"], out["Cash And Cash Equivalents_TTM"]) == (7.0, 3.0)
    assert ends["TTM"] == "2025-12-31"


def test_unit_error_in_the_annual_report_does_not_blank_a_sound_ttm(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "Total Revenue_Y0"] *= 1000  # thousands read as units
    df = calculate_metrics(raw, cfg)
    assert pd.isna(df.loc[0, "Total Revenue_Y0"])
    assert df.loc[0, "Total Revenue_TTM"] == raw.loc[0, "Total Revenue_TTM"]


def test_three_steady_rises_count_like_a_dip_then_two():
    years = ["Y0", "Y1", "TTM"]
    row = pd.Series({"X_Y1": 10.0, "X_Y0": 11.0, "X_TTM": 12.0})
    assert growth_profile(row, "X", years)["status"] == "RECENT_GROWTH_STABLE"
    jump = pd.Series({"X_Y1": 1.0, "X_Y0": 2.0, "X_TTM": 10.0})
    assert growth_profile(jump, "X", years)["status"] == "RECENT_GROWTH"


def test_unranked_models_are_told_why_they_are_not_flagged(cfg):
    df = pd.DataFrame(
        {"Ticker": ["OP", "REIT"], "Value_Score": 90.0, "Quality_Score": 70.0, "Revenue_Change_Recent": 3.0,
         "EBIT_Change_Recent": 2.0, "EPS_Rev_90d": 0.0, "Ranked_Model": [True, False]}
    )  # fmt: skip
    flag, text = dislocation(df, cfg)
    assert flag.tolist() == [True, False] and "not in the rankings" in text[1]


def test_headline_ic_leaves_out_unranked_models():
    n = 40
    snap = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "Value_Score": np.linspace(100, 1, n),
                         "Ranked_Model": [True] * 30 + [False] * 10})  # fmt: skip
    fwd = pd.Series(np.r_[np.linspace(30, 0, 30), np.full(10, -50.0)], index=snap["Ticker"])
    _, ic = evaluate_snapshot(snap, fwd, top=5)
    assert ic["Value_Score"] == pytest.approx(1.0)
