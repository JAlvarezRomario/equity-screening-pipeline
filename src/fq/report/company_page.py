"""Company pages: rendering and the fingerprint that leaves an unchanged page alone.

Only this module and the files in `PAGE_INPUTS` decide a page's look, so editing the front page's code (`html.py`,
`screens.py`, `index_charts.py`) re-renders no company page (user's point, Nov 1, 2026: the Repeaters tab cost
an 80-minute re-render). What a page shows of the lists travels in its row, which the fingerprint hashes."""

import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs_version

from fq.metrics.long_history import parse_history
from fq.report import charts
from fq.report.cards import check_cards, grade_card, holders_card
from fq.report.format import fmt
from fq.report.links import external_links, slug
from fq.report.notes import data_notes, owner_earnings_caption
from fq.report.pills import company_flags
from fq.report.tiles import (
    HEADLINE_FACTS,
    company_facts,
    headline_facts,
    is_financial,
    key_tiles,
    model_line,
    net_cash_pill,
    rank_counts,
    score_panel,
    score_tiles,
    unreliable_reasons,
)
from fq.settings import Settings

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_FILES = ("style.css", "main.js")


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # The cartesian build (scatter, bar, histogram: every chart the site draws) is a third of the full one, which
    # cost 1.5 MB on every first visit (Oct 31, 2026).
    env.globals["plotly_src"] = f"https://cdn.plot.ly/plotly-cartesian-{get_plotlyjs_version()}.min.js"
    return env


PAGE_MANIFEST = ".pages.json"  # ticker -> fingerprint of the inputs its page was written from (never deployed)
_REPORT_DIR = Path(__file__).parent


def _read_manifest(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# The files a company page is built from (`_page_inputs_version`). The front page's and the test page's own code
# (`html.py`, `screens.py`, `index_charts.py`, `forward_page.py`, their templates) and the static `main.js` /
# `style.css` (linked, not embedded) are not: editing them leaves the company pages as they are (user's point,
# Oct 31, 2026: a cosmetic change to the front page's map re-rendered all 8,000 pages).
PAGE_INPUTS = (
    "charts.py", "format.py", "tiles.py", "pills.py", "cards.py", "notes.py", "company_page.py", "links.py",
    "base.html.j2", "company.html.j2",
)  # fmt: skip


def _page_inputs_version(cfg: Settings, periods, treasury_yield, counts) -> bytes:
    """What every company page shares: its report code and templates (`PAGE_INPUTS`), settings, periods, the
    Treasury yield and the ranks' "of N". Any change re-renders every page."""
    digest = hashlib.sha1()
    for f in sorted([*_REPORT_DIR.glob("*.py"), *TEMPLATE_DIR.glob("*")]):
        if f.is_file() and f.name in PAGE_INPUTS:
            digest.update(f.name.encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    digest.update(repr((cfg, periods, treasury_yield, counts)).encode())
    return digest.digest()


def _page_fingerprint(shared: bytes, item: tuple) -> str:
    """Hash of one page's inputs: the shared part, the row's values, previous / next and the held mark."""
    prev, next_item, row, is_held = item
    digest = hashlib.sha1(shared)
    digest.update(row.to_json(default_handler=str, double_precision=15).encode())
    digest.update(repr((prev, next_item, is_held)).encode())
    return digest.hexdigest()


def _data_date(row: pd.Series) -> str | None:
    """The company's newest data (price refresh or full fetch), shown on its page instead of the build time,
    so a page whose figures did not change stays byte for byte the same (and is not re-deployed)."""
    stamps = pd.to_datetime(pd.Series([row.get("Price_At"), row.get("Fetched_At")]), errors="coerce").dropna()
    return stamps.max().strftime("%Y-%m-%d") if not stamps.empty else None


def _render_company_batch(args: tuple) -> None:
    comp_dir, cfg, periods, treasury_yield, stamp, counts, items = args
    env = _env()
    company_tpl = env.get_template("company.html.j2")
    for prev, next_item, row, is_held in items:
        ticker = row["Ticker"]
        tiles, drivers = key_tiles(row), score_tiles(row)
        shown_above = frozenset(c[1] for c in HEADLINE_FACTS) | {t["col"] for t in [*tiles, *drivers]}
        holders = holders_card(row)
        if holders and holders["line"]:  # the card says the institutions' share; "More figures" need not
            shown_above |= {"heldPercentInstitutions"}
        notes = data_notes(row, cfg, periods)
        mood = row.get("Mood_Check")  # the plain verdict (`ranking.dislocation`), lifted into "Why it's here"
        unreliable = unreliable_reasons(row)
        # The mood line ("Cheap (Value 99)...") reads the numbers as right: left out when they may not be.
        verdict = [] if unreliable else [n for n in notes if isinstance(mood, str) and n == mood]
        groups = company_facts(row, also_skip=shown_above, counts=counts)
        row_df = row.to_frame().T
        page = company_tpl.render(
            ticker=ticker,
            name=row.get("shortName"),
            country=row.get("country"),
            industry=row.get("industry"),
            model=model_line(row),
            scores=score_panel(row, counts),
            unreliable=unreliable,
            flags=company_flags(row),
            tiles=tiles,
            drivers=drivers,
            history_wide=len(parse_history(row.get("DPS_Hist"))) >= charts.LONG_HISTORY_MIN_YEARS,
            on_lists=row.get("On_Lists") if isinstance(row.get("On_Lists"), str) else None,
            verdict=verdict,
            charts=charts.company_charts_html(row_df, ticker, cfg, periods, treasury_yield=treasury_yield),
            nav_note=(
                f"Now: book {fmt(row.get('BV_to_Price'), '.0f')}%, tangible "
                f"{fmt(row.get('NAV_to_Price'), '.0f')}% of the price."
            ),
            caption=owner_earnings_caption(row),
            headline=headline_facts(row),
            n_ranked=counts.get("Rank", 0),
            links=external_links(ticker),
            more=[g for g in groups if g[0] not in ("Ranking", "Value criteria", "Quality criteria")],
            prev=prev,
            next=next_item,
            notes=[n for n in notes if n not in verdict],
            summary=row.get("longBusinessSummary"),
            website=row.get("website") if pd.notna(row.get("website")) and str(row.get("website")).strip() else None,
            net_cash_pill=net_cash_pill(row),
            grade=grade_card(row),
            checks=check_cards(row),
            holders=holders,
            is_financial=is_financial(row),
            held=is_held,
            date=stamp,
            data_date=_data_date(row),
        )
        (comp_dir / f"{slug(ticker)}.html").write_text(page, encoding="utf-8")


def write_company_pages(
    df: pd.DataFrame,
    cfg: Settings,
    out_dir: Path,
    periods,
    treasury_yield,
    stamp: str,
    held: frozenset[str] = frozenset(),
    tickers: set[str] | None = None,
) -> None:
    """Write `out_dir/companies/<slug>.html` for every row of `df` (only `tickers`, when given) whose inputs changed
    since its page was written (`.pages.json`); `df` is in the full ranking's order, which previous / next follow."""
    comp_dir = out_dir / "companies"
    counts = rank_counts(df)
    # Previous / next company in the full ranking's order (the order of `df`, as the index table).
    order = [
        {"ticker": t, "slug": slug(t), "rank": fmt(r, ",.0f")} for t, r in zip(df["Ticker"], df["Rank"], strict=True)
    ]
    items = [
        (order[i - 1] if i > 0 else None, order[i + 1] if i + 1 < len(order) else None, row, row["Ticker"] in held)
        for i, (_, row) in enumerate(df.iterrows())
    ]
    if tickers is not None:
        items = [it for it in items if it[2]["Ticker"] in tickers]
    # A page whose inputs are those it was last written from is left as it is (`.pages.json`).
    manifest_path = out_dir / PAGE_MANIFEST
    manifest = _read_manifest(manifest_path)
    shared = _page_inputs_version(cfg, periods, treasury_yield, counts)
    prints = {it[2]["Ticker"]: _page_fingerprint(shared, it) for it in items}
    n_all = len(items)
    items = [
        it
        for it in items
        if manifest.get(it[2]["Ticker"]) != prints[it[2]["Ticker"]]
        or not (comp_dir / f"{slug(it[2]['Ticker'])}.html").exists()
    ]

    if items:
        # FQ_RENDER_WORKERS caps the processes: each holds Plotly and its pages, and four ran an 8 GB machine out of
        # memory (Oct 31, 2026).
        n_workers = min(os.cpu_count() or 1, 8, int(os.environ.get("FQ_RENDER_WORKERS") or 8))
        if len(items) >= 50 and n_workers > 1:
            chunk_size = (len(items) + n_workers - 1) // n_workers
            chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
            tasks = [(comp_dir, cfg, periods, treasury_yield, stamp, counts, chunk) for chunk in chunks]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_render_company_batch, tasks))
        else:
            _render_company_batch((comp_dir, cfg, periods, treasury_yield, stamp, counts, items))
    if tickers is None:  # pages of tickers no longer ranked are pruned by `build_site`; forget them too
        manifest = {t: p for t, p in manifest.items() if t in set(df["Ticker"])}
    manifest.update(prints)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    log.info("Site written to %s (%d company pages rendered, %d unchanged)", out_dir.resolve(), len(items),
             n_all - len(items))  # fmt: skip
