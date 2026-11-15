"""Offline tests for `fq.macro`, `index_charts` yield figures, and `fq.report.macro_page`."""

from __future__ import annotations

import io
import zipfile
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
  <tr><td>Jan 1, 1960</td><td>5.90%</td></tr>
  <tr><td>Dec 1, 1959</td><td>5.75%</td></tr>
</table>
</body></html>
"""

SAMPLE_MOF_CSV = """Interest Rate,,,(Unit : %)
Date,1Y,5Y,10Y
2013/1/30,0.1,0.2,0.74
2013/1/31,0.1,0.2,0.76
2026/10/30,1.659,2.237,2.817
"""


def _sample_xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            '<sheet name="Non-Consolidated(AVG)~202203" sheetId="1" r:id="rId1"/>'
            '<sheet name="Consolidated(AVG)~202203" sheetId="2" r:id="rId2"/>'
            '<sheet name="Consolidated(Weighted avg.)" sheetId="3" r:id="rId3"/>'
            "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet2.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet3.xml"/>'
            "</Relationships>",
        )
        z.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1"><v>1989</v></c><c r="B1"><v>1</v></c><c r="I1"><v>50.0</v></c></row>'
            '<row r="2"><c r="A2"><v>2013</v></c><c r="B2"><v>1</v></c><c r="I2"><v>99.0</v></c></row>'
            "</sheetData></worksheet>",
        )
        z.writestr(
            "xl/worksheets/sheet2.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1"><v>2005</v></c><c r="B1"><v>1</v></c><c r="D1"><v>40.0</v></c></row>'
            '<row r="2"><c r="A2"><v>2013</v></c><c r="B2"><v>1</v></c><c r="D2"><v>80.0</v></c></row>'
            "</sheetData></worksheet>",
        )
        z.writestr(
            "xl/worksheets/sheet3.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1"><v>2013</v></c><c r="B1"><v>1</v></c><c r="D1"><v>25.0</v></c></row>'
            '<row r="2"><c r="A2"><v>2026</v></c><c r="B2"><v>10</v></c><c r="D2"><v>20.0</v></c></row>'
            "</sheetData></worksheet>",
        )
    return buf.getvalue()


def _sample_yields() -> pd.DataFrame:
    dates = pd.to_datetime(["1960-01-31", "1960-02-29", "2026-06-30"])
    return pd.DataFrame(
        {
            "treasury_10y": [4.72, 4.49, 4.42],
            "sp500_earnings_yield": [5.90, 6.02, 3.96],
            "spread_pp": [1.18, 1.53, -0.46],
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _sample_japan_yields() -> pd.DataFrame:
    dates = pd.to_datetime(["2013-01-31", "2026-10-31"])
    return pd.DataFrame(
        {
            "jgb_10y": [0.76, 2.817],
            "topix_earnings_yield": [4.0, 5.0],
            "spread_pp": [3.24, 2.183],
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_parse_multpl_table_clips_before_1960():
    s = macro.parse_multpl_table(SAMPLE_MULTPL_HTML)
    assert list(s.index) == [pd.Timestamp("1960-01-31"), pd.Timestamp("2026-06-30")]
    assert list(s.values) == [5.90, 3.96]


def test_fetch_treasury_history_backfills_from_fred(monkeypatch):
    tnx_close = pd.Series(
        [4.10, 4.42],
        index=pd.to_datetime(["1962-01-31", "2026-06-30"]),
    )
    monkeypatch.setattr(macro.yf, "download", lambda *_a, **_kw: pd.DataFrame({"Close": tnx_close}))
    gs10 = pd.Series(
        [4.72, 3.95],
        index=pd.to_datetime(["1960-01-31", "1962-01-31"]),
        name="treasury_10y",
    )
    monkeypatch.setattr(macro, "fetch_fred_gs10", lambda start=macro.START_DATE: gs10)

    res = macro.fetch_treasury_history()
    assert res.loc[pd.Timestamp("1960-01-31")] == 4.72
    assert res.loc[pd.Timestamp("1962-01-31")] == 4.10
    assert res.loc[pd.Timestamp("2026-06-30")] == 4.42


def test_parse_mof_jgb_and_jpx_xlsx():
    jgb = macro.parse_mof_jgb_csv(SAMPLE_MOF_CSV)
    assert jgb.loc[pd.Timestamp("2013-01-31")] == 0.76
    assert jgb.loc[pd.Timestamp("2026-10-31")] == 2.817

    ey = macro.parse_jpx_perpbr_xlsx(_sample_xlsx_bytes())
    assert ey.loc[pd.Timestamp("1989-01-31")] == 2.0  # 100 / 50.0 (Non-Consolidated AVG)
    assert ey.loc[pd.Timestamp("2005-01-31")] == 2.5  # 100 / 40.0 (Consolidated AVG)
    assert ey.loc[pd.Timestamp("2013-01-31")] == 4.0  # 100 / 25.0 (Consolidated Weighted overrides)
    assert ey.loc[pd.Timestamp("2026-10-31")] == 5.0


def test_load_macro_yields_caches_and_falls_back(tmp_path: Path, monkeypatch):
    cache = tmp_path / "macro_yields.parquet"
    ty = pd.Series([7.75, 4.42], index=pd.to_datetime(["1960-01-31", "2026-06-30"]), name="treasury_10y")
    ey = pd.Series([6.34, 3.96], index=pd.to_datetime(["1960-01-31", "2026-06-30"]), name="sp500_earnings_yield")

    monkeypatch.setattr(macro, "fetch_treasury_history", lambda start=macro.START_DATE: ty)
    monkeypatch.setattr(macro, "fetch_sp500_earnings_yield", lambda start=macro.START_DATE: ey)

    df = macro.load_macro_yields(cache, max_age_days=7, offline=False)
    assert cache.is_file()
    assert list(df["spread_pp"]) == [-1.41, -0.46]

    def _boom(*_args, **_kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr(macro, "fetch_treasury_history", _boom)
    monkeypatch.setattr(macro, "fetch_sp500_earnings_yield", _boom)
    cached = macro.load_macro_yields(cache, max_age_days=0, offline=True)
    assert cached.equals(df)

    stale = macro.load_macro_yields(cache, max_age_days=-1, offline=False)
    assert stale.equals(df)


def test_load_japan_yields_caches(tmp_path: Path, monkeypatch):
    cache = tmp_path / "macro_japan_yields.parquet"
    jgb = macro.parse_mof_jgb_csv(SAMPLE_MOF_CSV)
    ey = macro.parse_jpx_perpbr_xlsx(_sample_xlsx_bytes())
    monkeypatch.setattr(macro, "fetch_japan_jgb_history", lambda: jgb)
    monkeypatch.setattr(macro, "fetch_japan_earnings_yield", lambda: ey)

    df = macro.load_japan_yields(cache, max_age_days=7, offline=False)
    assert cache.is_file()
    assert round(float(df.loc[pd.Timestamp("2013-01-31"), "spread_pp"]), 2) == 3.24


def test_macro_figures_have_three_traces():
    us_fig = index_charts.macro_yields_figure(_sample_yields())
    assert len(us_fig.data) == 3
    assert us_fig.data[0].name == "S&P 500 Earnings Yield (TTM)"

    jp_fig = index_charts.japan_yields_figure(_sample_japan_yields())
    assert len(jp_fig.data) == 3
    assert jp_fig.data[0].name == "TSE Prime / 1st Sec. Earnings Yield"
    assert jp_fig.data[1].name == "10Y JGB Yield (MOF)"


def test_build_macro_writes_us_and_japan_sections(tmp_path: Path):
    out = build_macro(tmp_path, _sample_yields(), japan_yields=_sample_japan_yields(), copy_static=True)
    assert out == tmp_path / "macro.html"
    html = out.read_text(encoding="utf-8")
    assert "United States" in html
    assert "Japan" in html
    assert "3.96%" in html
    assert "5.00%" in html
    assert "2.82%" in html
    assert "macro_yields_chart" in html
    assert "macro_japan_chart" in html
