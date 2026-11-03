"""The long share-count and dividend record: fetch helpers, derived figures and the company chart (no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.fetch import yearly_dividends, yearly_shares
from fq.metrics import calculate_metrics
from fq.metrics.long_history import long_history, parse_history
from fq.report.charts import long_history_figure


def _series(points: dict[str, float], tz: str | None = None) -> pd.Series:
    idx = pd.to_datetime(list(points))
    return pd.Series(list(points.values()), index=idx.tz_localize(tz) if tz else idx, dtype=float)


def test_yearly_shares_median_of_late_points_drops_spikes():
    pts = _series({"2020-03-31": 120, "2020-11-30": 100, "2020-12-15": 900, "2020-12-31": 100, "2021-12-31": 98},
                  tz="America/New_York")  # fmt: skip
    out = yearly_shares(pts, pd.Series(dtype=float), ref=98)
    assert out == {"2020": 100.0, "2021": 98.0}  # Q4 median, the one-day spike out


def test_yearly_shares_drops_points_on_another_basis():
    # Swire: counts of every class (3.6x) mixed among economic shares, even as a year's only late point.
    pts = _series({f"2025-{m:02d}-15": 100.0 for m in range(1, 13)} | {"2025-11-17": 360.0, "2026-01-06": 101.0,
                   "2026-01-10": 360.0})  # fmt: skip
    out = yearly_shares(pts, pd.Series(dtype=float), ref=101)
    assert out == {"2025": 100.0, "2026": 101.0}


def test_yearly_shares_split_adjusts_unrestated_points_only():
    # A 2:1 split in 2022: 2019-2021 reported raw (50), 2021 once already restated (100).
    pts = _series({"2019-12-31": 50, "2020-12-31": 50, "2021-12-31": 100, "2023-12-31": 101})
    splits = _series({"2022-06-01": 2.0})
    out = yearly_shares(pts, splits, ref=101)
    assert out == {"2019": 100.0, "2020": 100.0, "2021": 100.0, "2023": 101.0}  # gap year 2022 stays empty


def test_yearly_shares_rescaled_to_the_quoted_basis():
    # An ADR worth 4 shares: the series counts ordinary shares, the price is per ADR.
    pts = _series({"2022-12-31": 400, "2023-12-31": 380, "2024-12-31": 360})
    out = yearly_shares(pts, pd.Series(dtype=float), ref=90)
    assert out == pytest.approx({"2022": 100.0, "2023": 95.0, "2024": 90.0})
    assert yearly_shares(pd.Series(dtype=float), pd.Series(dtype=float), ref=90) == {}


def test_yearly_dividends_complete_years_only():
    divs = _series({"2023-03-01": 0.5, "2023-09-01": 0.5, "2024-03-01": 0.6, "2024-09-01": 0.6, "2025-03-01": 0.7},
                   tz="Europe/Madrid")  # fmt: skip
    out = yearly_dividends(divs, pd.Timestamp("2025-10-30"))
    assert out == pytest.approx({"2023": 1.0, "2024": 1.2})
    assert yearly_dividends(pd.Series(dtype=float), pd.Timestamp("2025-10-30")) == {}


def _frame(shares: dict | None, dps: dict | None, fetched: str = "2026-10-30 10:00") -> pd.DataFrame:
    return pd.DataFrame([{"Ticker": "X", "currency": "EUR", "Fetched_At": fetched,
                          "Shares_Hist": json.dumps(shares) if shares else None,
                          "DPS_Hist": json.dumps(dps) if dps else None}])  # fmt: skip


def test_derived_figures(cfg):
    shares = {str(y): 1000 * 0.98 ** (y - 2015) for y in range(2015, 2027)}
    dps = {str(y): 1.0 * 1.05 ** (y - 2004) for y in range(2004, 2026)}
    dps["2009"] = dps["2008"] * 0.5  # a cut: the streak starts after it
    df = _frame(shares, dps)
    long_history(df, cfg)
    assert df["Shares_Change_10Y"].iloc[0] == pytest.approx(-2.0)
    assert df["Dividend_Years_No_Cut"].iloc[0] == 17  # 2009..2025
    assert df["DPS_Growth_10Y"].iloc[0] == pytest.approx(5.0)
    assert not np.isnan(df["DPS_Growth_20Y"].iloc[0])


def test_stopped_dividend_has_no_streak_or_growth(cfg):
    dps = {str(y): 1.0 for y in range(2005, 2020)}  # nothing since 2020
    df = _frame({"2024": 10, "2025": 10}, dps)
    long_history(df, cfg)
    assert df["Dividend_Years_No_Cut"].iloc[0] == 0
    assert np.isnan(df["DPS_Growth_10Y"].iloc[0])
    assert np.isnan(df["Shares_Change_10Y"].iloc[0])  # under 5 years of shares


def test_pipeline_tolerates_rows_without_history(cfg):
    df = calculate_metrics(make_raw(cfg, n=4), cfg)
    assert df["Dividend_Years_No_Cut"].isna().all()
    assert parse_history("not json").empty and parse_history(None).empty


def test_chart_draws_what_exists():
    shares = {"2020": 1e9, "2021": 0.98e9, "2022": 0.95e9}
    dps = {"2021": 1.0, "2022": 1.1, "2023": 1.2, "2024": 1.3}
    both = long_history_figure(_frame(shares, dps), "X")
    assert [t.type for t in both.data] == ["bar", "scatter"]
    assert "EUR" in both.data[0].name
    assert [t.type for t in long_history_figure(_frame(None, dps), "X").data] == ["bar"]
    assert [t.type for t in long_history_figure(_frame(shares, None), "X").data] == ["scatter"]
    assert long_history_figure(_frame({"2024": 1e9, "2025": 1e9}, {"2025": 1.0}), "X") is None


def test_a_payment_slipping_across_the_year_end_is_no_cut(cfg):
    # KO: five payments in one year (0.54), so the next (0.40) looked cut; and a missed payment made up next year.
    dps = {"2016": 0.30, "2017": 0.32, "2018": 0.54, "2019": 0.40, "2020": 0.30, "2021": 0.52, "2022": 0.44}
    df = _frame(None, dps, fetched="2023-05-01 10:00")
    long_history(df, cfg)
    assert df["Dividend_Years_No_Cut"].iloc[0] == 7
    halved = {"2019": 1.0, "2020": 1.0, "2021": 0.5, "2022": 0.5}
    df = _frame(None, halved, fetched="2023-05-01 10:00")
    long_history(df, cfg)
    assert df["Dividend_Years_No_Cut"].iloc[0] == 2  # 2021 and 2022, after the cut
