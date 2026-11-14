"""The macro yields page (`macro.html`): 10-year US Treasury yield vs. S&P 500 TTM earnings yield since 1970."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fq.report import index_charts
from fq.report.company_page import STATIC_FILES, TEMPLATE_DIR, _env


def build_macro(out_dir: Path, yields: pd.DataFrame, copy_static: bool = False) -> Path:
    """Write `<out_dir>/macro.html` from a monthly DataFrame (`treasury_10y`, `sp500_earnings_yield`, `spread_pp`)."""
    if yields.empty:
        raise ValueError("Cannot build macro.html from an empty yields DataFrame")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_static:
        for name in STATIC_FILES:
            shutil.copy2(TEMPLATE_DIR / name, out_dir / name)

    clean = yields.sort_index()
    latest = clean.iloc[-1]
    spread = clean["spread_pp"].astype(float)
    page = (
        _env()
        .get_template("macro.html.j2")
        .render(
            root="",
            start_date=f"{clean.index[0]:%B %Y}",
            latest_date=f"{clean.index[-1]:%B %Y}",
            data_date=f"{clean.index[-1]:%Y-%m-%d}",
            n_months=len(clean),
            latest_ey=f"{float(latest['sp500_earnings_yield']):.2f}%",
            latest_ty=f"{float(latest['treasury_10y']):.2f}%",
            latest_spread=f"{float(latest['spread_pp']):+.2f} pp",
            latest_spread_pos=float(latest["spread_pp"]) >= 0,
            median_spread=f"{float(spread.median()):+.2f} pp",
            min_spread=f"{float(spread.min()):+.2f} pp",
            max_spread=f"{float(spread.max()):+.2f} pp",
            chart=index_charts.macro_yields_html(clean),
        )
    )
    path = out_dir / "macro.html"
    path.write_text(page, encoding="utf-8")
    return path
