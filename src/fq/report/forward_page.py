"""The forward test's page (`forward.html`): what the test is, the cumulative-return chart and results table since
the pivot, and the frozen members. The backward months stay in the ledger but are not shown (user's choice, Oct 31,
2026: hindsight on the lists' own picks misled more than it told). Built from `fq.forward`'s cohort and ledger; no network."""

import shutil
from pathlib import Path

import pandas as pd

from fq import forward
from fq.report import index_charts
from fq.report.company_page import STATIC_FILES, TEMPLATE_DIR, _env
from fq.report.format import fmt
from fq.report.highlights import label
from fq.report.links import slug

# (label, column, format, suffix, help, better when up)
RESULT_COLUMNS = (
    ("Companies", "Members", ".0f", "", "Companies frozen in the list on the start date, each bought for the same amount.", True),
    ("Return", "Return", "+.1f", "%", "Total return since the start, in US dollars, dividends reinvested.", True),
    ("Of which dividends", "Dividends", "+.1f", " pp", "How much of the return the dividends added: the total return minus the return from the price alone, in percentage points.", True),
    ("vs SPY", "vs SPY", "+.1f", " pp", "Return minus the S&P 500 ETF's, in percentage points.", True),
    ("vs VT", "vs VT", "+.1f", " pp", "Return minus the total world ETF's, in percentage points.", True),
    ("Per year", "Annualised", "+.1f", "%", "Compound return per year; shown once a full year has passed.", True),
    ("Worst fall", "Max_Drawdown", ".1f", "%", "The largest fall from a previous high, on month-end values.", True),
    ("Months ahead of VT", "Months_Ahead", ".0f", "%", "Share of months in which the list returned more than the world ETF.", True),
    ("Companies ahead of VT", "Members_Ahead", ".0f", "%", "Share of the list's companies whose own return beats the world ETF's.", True),
    ("Still listed", "Still_Listed", ".0f", "", "How many of the frozen companies the live list on the main page still shows today.", True),
    ("Stale", "Stale", ".0f", "", "Companies without a price at the last month-end (delisted or missing on Yahoo), held at their last value.", False),
)  # fmt: skip


def _cell(value, spec: str, suffix: str, up_good: bool, signed: bool) -> dict:
    cls = "num"
    if signed and pd.notna(value) and value != 0:
        cls += " pos" if (value > 0) == up_good else " neg"
    return {"text": fmt(value, spec, suffix=suffix), "cls": cls}


def result_rows(table: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """(headers, rows) of the results table: best excess over the world ETF first, the benchmarks in their place."""
    columns = [c for c in RESULT_COLUMNS if c[1] in table.columns]
    order = table.sort_values("vs VT", ascending=False, kind="stable")
    headers = [
        {"label": label, "help": help_, "first": "desc" if up else "asc"} for label, _, _, _, help_, up in columns
    ]
    rows = []
    for r in order.itertuples(index=False):
        rec = r._asdict()
        cells = [_cell(rec.get(col), spec, suffix, up, signed=spec.startswith("+") or col == "Max_Drawdown")
                 for _, col, spec, suffix, _, up in columns]  # fmt: skip
        rows.append({"name": label(rec["Series"]), "benchmark": rec["Kind"] == "benchmark",
                     "portfolio": rec["Series"] == forward.PORTFOLIO_LIST, "cells": cells})  # fmt: skip
    return headers, rows


def member_groups(members: pd.DataFrame, pages: set[str]) -> list[dict]:
    """One fold per list: its frozen companies in entry order, each with its return and whether it is still listed.
    Lists are named as the front page names them (`highlights.label`); the cohort keeps the names they were frozen
    under."""
    groups = []
    for name, part in members.groupby("List", sort=False):
        has_live = "Still_Listed" in part.columns and name != forward.PORTFOLIO_LIST
        rows = [{"ticker": r.Ticker, "slug": slug(r.Ticker) if slug(r.Ticker) in pages else None,
                 "name": r.Name if isinstance(r.Name, str) else "", "country": r.Country if isinstance(r.Country, str) else "",
                 "position": int(r.Position), "fwd": _cell(r.Return_Fwd, "+.1f", "%", True, True),
                 "stale": bool(r.Stale), "listed": bool(r.Still_Listed) if has_live else None}
                for r in part.itertuples(index=False)]  # fmt: skip
        still = sum(1 for r in rows if r["listed"]) if has_live else None
        groups.append({"name": label(name), "rows": rows, "n": len(rows), "still": still})
    return groups


def _phase(cohort: pd.DataFrame, ledger: pd.DataFrame, start: pd.Timestamp, live, div_id: str) -> dict:
    """One phase's chart and results (`ledger` holds only its marks); no chart before its first month-end."""
    values = forward.series_values(cohort, ledger, start)
    headers, rows = result_rows(forward.summary(values, cohort, ledger, live))
    months = (values.index[-1].year - start.year) * 12 + values.index[-1].month - start.month
    return {"start": f"{start:%d %B %Y}", "last": f"{values.index[-1]:%d %B %Y}", "months": months,
            "chart": index_charts.forward_html(values.rename(columns=label), div_id) if len(values) > 1 else "", "headers": headers,
            "rows": rows if len(values) > 1 else []}  # fmt: skip


def build_placeholder(out_dir: Path) -> Path:
    """Write `<out_dir>/forward.html` before the cohort is frozen: when the test starts and when its first result
    comes (`forward.START`, `forward.FROZEN_ON`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = forward.START + pd.offsets.MonthEnd(1)
    page = (
        _env()
        .get_template("forward_soon.html.j2")
        .render(
            root="",
            pivot=f"{forward.last_close(forward.START):%d %B %Y}",
            frozen_on=f"{forward.FROZEN_ON:%d %B %Y}",
            first_fwd=f"{forward.last_close(first):%d %B %Y}",
        )  # fmt: skip
    )
    path = out_dir / "forward.html"
    path.write_text(page, encoding="utf-8")
    return path


def build_forward(
    out_dir: Path,
    cohort: pd.DataFrame,
    ledger: pd.DataFrame,
    live: pd.DataFrame | None = None,
    copy_static: bool = False,
) -> Path:
    """Write `<out_dir>/forward.html`; `live` (today's published lists) marks who is still on each list. Company
    links point at `companies/<slug>.html` where such a page exists."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_static:
        for name in STATIC_FILES:
            shutil.copy2(TEMPLATE_DIR / name, out_dir / name)
    (start, main), *later_groups = forward.pivot_groups(cohort)
    _, fwd_ledger = forward.phases(ledger, start)
    fwd = _phase(main, fwd_ledger, start, live, index_charts.FORWARD_ID)
    later = []
    for when, part in later_groups:  # lists bought at a pivot of their own, measured apart
        phase = _phase(part, forward.phases(ledger, when)[1], when, live, f"{index_charts.FORWARD_ID}_{when:%Y%m%d}")
        later.append(phase | {"pivot": f"{forward.last_close(when):%d %B %Y}", "anchor": f"forward-{when:%Y-%m}",
                              "first": f"{forward.last_close(when + pd.offsets.MonthEnd(1)):%d %B %Y}",
                              "lists": [label(n) for n in part["List"].unique()]})  # fmt: skip
    members = forward.member_table(cohort, ledger, live)
    pages = {p.stem for p in (out_dir / "companies").glob("*.html")}
    no_fx = sorted(set(ledger.loc[ledger.get("No_FX", pd.Series(False, index=ledger.index)).astype(bool), "Ticker"]))
    first_fwd = start + pd.offsets.MonthEnd(1)
    check = fwd_ledger.get("Check", pd.Series(False, index=fwd_ledger.index)).astype(bool)
    late = cohort[cohort["List"].isin(forward.LATE_LISTS)].drop_duplicates("List")
    page = (
        _env()
        .get_template("forward.html.j2")
        .render(
            root="",
            pivot=f"{forward.last_close(start):%d %B %Y}",
            published_on=f"{main['Published_On'].iloc[0]:%d %B %Y}",
            first_fwd=f"{forward.last_close(first_fwd):%d %B %Y}",
            first_fwd_run=f"{first_fwd + pd.Timedelta(days=1):%d %B}",
            fwd=fwd,
            gated_out=cohort.attrs.get("gated_out"),
            gate_date=f"{cohort.attrs['gate_date']:%d %B %Y}" if "gate_date" in cohort.attrs else "",
            n_lists=cohort.loc[~cohort["List"].isin([forward.PORTFOLIO_LIST, *forward.LATE_LISTS]), "List"].nunique(),
            late_lists=[
                {
                    "name": label(r.List),
                    "date": f"{r.Published_On:%d %B %Y}",
                    "own": r.Start != start,
                    "pivot": f"{forward.last_close(r.Start):%d %B %Y}",
                }
                for r in late.itertuples()
            ],  # fmt: skip
            later=later,
            check=sorted(set(fwd_ledger.loc[check, "Ticker"])),
            groups=member_groups(members, pages),
            no_fx=no_fx,
            data_date=f"{(ledger['Date'].max() if len(ledger) else start):%Y-%m-%d}",
        )
    )
    path = out_dir / "forward.html"
    path.write_text(page, encoding="utf-8")
    return path
