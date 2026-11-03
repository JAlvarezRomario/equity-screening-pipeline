"""Behaviour that keeps the system working whatever Yahoo serves and whenever it is run."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq import health
from fq.fetch import _annual_anchor, _contiguous_quarters, _statement_columns, annual_slots
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.format import period_label, report_periods
from fq.report.html import build_site
from fq.report.links import slug
from fq.settings import Settings


def stmt(dates: list[str], **rows: list[float]) -> pd.DataFrame:
    """A yfinance-shaped statement: rows = concepts, columns = period ends (newest first)."""
    return pd.DataFrame(rows, index=pd.to_datetime(dates)).T


QUARTERS = ["2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"]


def test_annual_slots_are_elapsed_years_not_positions():
    # FY ending June; the 2023 report is missing, so Y2 must stay empty and 2022 lands in Y3.
    frame = stmt(["2025-06-30", "2024-06-30", "2022-06-30"], **{"Total Revenue": [3, 2, 1]}).T
    slots = annual_slots(frame, 5)
    assert sorted(slots) == ["Y0", "Y1", "Y3"]
    assert slots["Y3"] == pd.Timestamp("2022-06-30")


def test_statements_share_the_income_statement_years():
    # LOG.MC: the balance sheet has a nearly empty newer year (shares only) the income statement does not have yet.
    income = stmt(["2024-09-30", "2023-09-30"], **{"Total Revenue": [2, 1]})
    balance = stmt(["2025-09-30", "2024-09-30", "2023-09-30"], **{"Total Assets": [None, 20, 10]})
    anchor = _annual_anchor(income, pd.DataFrame(), balance)
    assert anchor == pd.Timestamp("2024-09-30")
    slots = annual_slots(balance.T, 5, anchor)
    assert slots == {"Y0": pd.Timestamp("2024-09-30"), "Y1": pd.Timestamp("2023-09-30")}  # the newer year left out


def test_contiguous_quarters_detects_a_missing_quarter():
    good = stmt(QUARTERS, x=[1, 2, 3, 4, 5]).T
    assert _contiguous_quarters(good)
    gap = stmt(["2026-03-31", "2025-12-31", "2025-06-30", "2025-03-31"], x=[1, 2, 3, 4]).T
    assert not _contiguous_quarters(gap)


def cols(annual, quarterly, ttm=None, concept="Total Revenue", method="sum", cfg=None):
    out, ends, _meta = _statement_columns(
        annual, quarterly, ttm if ttm is not None else pd.DataFrame(), [concept], cfg or Settings(), 1.0, method
    )
    return out, ends


def test_ttm_is_the_sum_of_four_contiguous_quarters():
    q = stmt(QUARTERS, **{"Total Revenue": [10, 10, 10, 10, 99]})
    out, ends = cols(pd.DataFrame(), q)
    assert out["Total Revenue_TTM"] == 40 and ends["TTM"] == "2026-03-31"


def test_missing_quarter_or_value_falls_back_to_yahoo_ttm_then_nothing():
    gap = stmt(["2026-03-31", "2025-12-31", "2025-06-30", "2025-03-31"], **{"Total Revenue": [10, 10, 10, 10]})
    ttm = stmt(["2026-03-31"], **{"Total Revenue": [123]})
    assert cols(pd.DataFrame(), gap, ttm)[0]["Total Revenue_TTM"] == 123
    assert cols(pd.DataFrame(), gap)[0]["Total Revenue_TTM"] is None
    # A NaN quarter must not be silently summed as 3 quarters.
    hole = stmt(QUARTERS, **{"Total Revenue": [10, np.nan, 10, 10, 10]})
    assert cols(pd.DataFrame(), hole)[0]["Total Revenue_TTM"] is None


def test_balance_items_use_newest_quarter_with_a_value():
    q = stmt(QUARTERS, **{"Total Debt": [np.nan, 7, 6, 5, 4]})
    out, _ = cols(pd.DataFrame(), q, concept="Total Debt", method="latest")
    assert out["Total Debt_TTM"] == 7


def test_aliases_and_composites_cover_renamed_rows():
    annual = stmt(["2025-12-31"], **{"Operating Revenue": [50], "Common Stock Equity": [8]})
    out, _ = cols(annual, pd.DataFrame(), concept="Total Revenue")
    assert out["Total Revenue_Y0"] == 50
    out, _ = cols(annual, pd.DataFrame(), concept="Stockholders Equity", method="latest")
    assert out["Stockholders Equity_Y0"] == 8
    debt = stmt(
        ["2025-12-31"],
        **{"Long Term Debt And Capital Lease Obligation": [30], "Current Debt And Capital Lease Obligation": [5]},
    )
    out, _ = cols(debt, pd.DataFrame(), concept="Total Debt", method="latest")
    assert out["Total Debt_Y0"] == 35


def test_old_cache_schema_is_refused(cfg):
    raw = make_raw(cfg, n=2)
    health.check_schema(raw)
    with pytest.raises(health.CacheError, match="fetch again"):
        health.check_schema(raw.drop(columns="Schema_Version"))
    with pytest.raises(health.CacheError):
        health.check_schema(raw.assign(Schema_Version=1))


def test_health_report_flags_stale_cache_and_missing_concepts(cfg):
    raw = make_raw(cfg, n=6)
    assert health.report(raw, cfg) == []
    old = raw.assign(Fetched_At="2020-01-01 00:00", Stale_Annual=True)
    for col in [c for c in old.columns if c.startswith("Total Debt_")]:
        old[col] = np.nan
    messages = " ".join(health.report(old, cfg))
    assert "days old" in messages and "Total Debt" in messages and "15 months" in messages


def test_report_periods_drop_slots_nobody_has(cfg):
    raw = make_raw(cfg, n=3)
    for col in [c for c in raw.columns if c.endswith("_Y4")]:
        raw[col] = np.nan
    assert report_periods(raw, cfg) == ["Y3", "Y2", "Y1", "Y0", "TTM"]


def test_slot_without_ttm_and_per_company_labels(cfg):
    df = calculate_metrics(make_raw(cfg, n=2), cfg)
    assert period_label(df.iloc[0], "Y0") == "Dec-25" and period_label(df.iloc[0], "TTM").startswith("TTM Jun-26")
    assert period_label(pd.Series(dtype=object), "Y2") == "n/a"


def test_pipeline_runs_three_years_from_now_and_with_fewer_years(tmp_path):
    """Nothing depends on today's date or on a particular number of years."""
    cfg = Settings(n_years=3, data_dir=tmp_path / "d", output_dir=tmp_path / "o", label="future")
    raw = make_raw(cfg, n=5, latest_end="2028-12-31")
    ranked = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    index = build_site(ranked, cfg)
    page = (index.parent / "companies" / f"{slug(ranked.loc[0, 'Ticker'])}.html").read_text(encoding="utf-8")
    assert "Dec-28" in page and "Dec-25" not in page
    assert "2028" in index.read_text(encoding="utf-8")


def test_missing_period_columns_do_not_break_metrics(cfg):
    raw = make_raw(cfg, n=4)
    for col in [c for c in raw.columns if c.endswith(("_Y3", "_Y4", "_TTM"))]:
        raw = raw.drop(columns=col)
    ranked = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    assert len(ranked) == 4 and ranked["FCFy_Norm"].notna().all()
