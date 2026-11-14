"""The macro yields page (`macro.html`): US (1960–present) and Japan (2013–present) equity vs. 10Y bond yields."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fq.report import index_charts
from fq.report.company_page import STATIC_FILES, TEMPLATE_DIR, _env


def _section_context(
    df: pd.DataFrame,
    ey_col: str,
    bond_col: str,
    chart_html: str,
) -> dict:
    clean = df.sort_index()
    latest = clean.iloc[-1]
    spread = clean["spread_pp"].astype(float)
    return {
        "start_date": f"{clean.index[0]:%B %Y}",
        "latest_date": f"{clean.index[-1]:%B %Y}",
        "n_months": len(clean),
        "latest_ey": f"{float(latest[ey_col]):.2f}%",
        "latest_ty": f"{float(latest[bond_col]):.2f}%",
        "latest_spread": f"{float(latest['spread_pp']):+.2f} pp",
        "latest_spread_pos": float(latest["spread_pp"]) >= 0,
        "median_spread": f"{float(spread.median()):+.2f} pp",
        "min_spread": f"{float(spread.min()):+.2f} pp",
        "max_spread": f"{float(spread.max()):+.2f} pp",
        "chart": chart_html,
    }


def build_macro(
    out_dir: Path,
    yields: pd.DataFrame,
    japan_yields: pd.DataFrame | None = None,
    copy_static: bool = False,
) -> Path:
    """Write `<out_dir>/macro.html` with US and (when available) Japan yield comparisons."""
    if yields.empty and (japan_yields is None or japan_yields.empty):
        raise ValueError("Cannot build macro.html from empty yields DataFrames")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_static:
        for name in STATIC_FILES:
            shutil.copy2(TEMPLATE_DIR / name, out_dir / name)

    us = (
        _section_context(yields, "sp500_earnings_yield", "treasury_10y", index_charts.macro_yields_html(yields))
        if not yields.empty
        else None
    )
    jp = (
        _section_context(
            japan_yields, "topix_earnings_yield", "jgb_10y", index_charts.japan_yields_html(japan_yields)
        )
        if japan_yields is not None and not japan_yields.empty
        else None
    )
    latest_dates = [d.index.max() for d in (yields, japan_yields) if d is not None and not d.empty]
    page = (
        _env()
        .get_template("macro.html.j2")
        .render(
            root="",
            us=us,
            jp=jp,
            data_date=f"{max(latest_dates):%Y-%m-%d}",
        )
    )
    path = out_dir / "macro.html"
    path.write_text(page, encoding="utf-8")
    return path
