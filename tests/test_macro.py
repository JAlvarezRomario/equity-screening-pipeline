"""Offline tests for `fq.macro`, `index_charts.macro_yields_figure`, and `fq.report.macro_page`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fq import macro
from fq.report import index_charts
from fq.report.macro_page import build_macro

SAMPLE_MULTPL_HTML = """
<html><body>
<table id="datatable">
  <tr><th>Date</th><th>Value</th></tr>
  <tr><td>Jun 1, 2026</td><td>3.96%</td></tr>
  <tr><td>Jan 1, 1970</td><td>6.34%</td></tr>
  <tr><td>Dec 1, 1969</td><td>6.45%</td></tr>
</table>
</body></html>
"""


def _sample_yields() -> pd.DataFrame:
    dates = pd.to_datetime(["1970-01-31", "1970-02-28", "2026-06-30"])
    return pd.DataFrame(
        {
            "treasury_10y": [7.75, 6.90, 4.42],
            "sp500_earnings_yield": [6.34, 6.52, 3.96],
            "spread_pp": [-1.41, -0.38, -0.46],
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_parse_multpl_table_clips_before_1970():
    s = macro.parse_multpl_table(SAMPLE_MULTPL_HTML)
    assert list(s.index) == [pd.Timestamp("1970-01-31"), pd.Timestamp("2026-06-30")]
    assert list(s.values) == [6.34, 3.96]


def test_load_macro_yields_caches_and_falls_back(tmp_path: Path, monkeypatch):
    cache = tmp_path / "macro_yields.parquet"
    ty = pd.Series([7.75, 4.42], index=pd.to_datetime(["1970-01-31", "2026-06-30"]), name="treasury_10y")
    ey = pd.Series([6.34, 3.96], index=pd.to_datetime(["1970-01-31", "2026-06-30"]), name="sp500_earnings_yield")

    monkeypatch.setattr(macro, "fetch_treasury_history", lambda start=macro.START_DATE: ty)
    monkeypatch.setattr(macro, "fetch_sp500_earnings_yield", lambda start=macro.START_DATE: ey)

    df = macro.load_macro_yields(cache, max_age_days=7, offline=False)
    assert cache.is_file()
    assert list(df["spread_pp"]) == [-1.41, -0.46]

    # Offline reads from cache without calling fetchers
    def _boom(*_args, **_kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr(macro, "fetch_treasury_history", _boom)
    monkeypatch.setattr(macro, "fetch_sp500_earnings_yield", _boom)
    cached = macro.load_macro_yields(cache, max_age_days=0, offline=True)
    assert cached.equals(df)

    # Online failure with stale cache falls back to cache
    stale = macro.load_macro_yields(cache, max_age_days=-1, offline=False)
    assert stale.equals(df)


def test_macro_yields_figure_has_three_traces():
    fig = index_charts.macro_yields_figure(_sample_yields())
    assert len(fig.data) == 3
    assert fig.data[0].name == "S&P 500 Earnings Yield (TTM)"
    assert fig.data[1].name == "10Y US Treasury Yield (^TNX)"
    assert "Spread" in fig.data[2].name


def test_build_macro_writes_page(tmp_path: Path):
    out = build_macro(tmp_path, _sample_yields(), copy_static=True)
    assert out == tmp_path / "macro.html"
    html = out.read_text(encoding="utf-8")
    assert "Market yields" in html
    assert "3.96%" in html
    assert "4.42%" in html
    assert "-0.46 pp" in html
    assert "macro_yields_chart" in html
