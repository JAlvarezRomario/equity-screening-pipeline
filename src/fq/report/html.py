"""Static HTML site: index with quadrant + rankings, one interactive page per company.

Which companies each highlight tab shows is decided in `report.highlights`; this module only renders them."""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from fq import grading
from fq.report import dividend_record, index_charts
from fq.report import screens as sc
from fq.report.cards import check_pill, grade_pill
from fq.report.company_page import STATIC_FILES, TEMPLATE_DIR, _env, write_company_pages
from fq.report.format import benchmark_note, fmt, headers, period_range, report_periods, summary_columns, summary_rows
from fq.report.highlights import (
    DEEP_MIN_MGMT,
    GROUPS,
    MIN_ALL_PASS,
    SHORTLIST,
    TABS,
    THE_LIST_CASH_MGMT,
    THE_LIST_MIN_GRADE,
    THE_LIST_N,
    Highlights,
    Tab,
    fails,
)
from fq.report.links import slug
from fq.settings import Settings
from fq.universe import _read_lines, sovereign_spread

log = logging.getLogger(__name__)

# Left out of the combined ranking table: the row order is the combined rank, and the scores beside it say the rest.
HIDDEN_IN_RANKING = {"Value_Rank_Bench", "Quality_Rank_Bench"}


RANKING_REST = "ranking-rest.js"
_REST_OPEN = '<script type="text/html" id="ranking-rest">'


def _write_index(index: str, out_dir: Path) -> None:
    """Write index.html with the full ranking's lazy rows in their own file (`RANKING_REST`), so the front page does
    not carry ~6 MB nobody reads until a search, filter, sort or "Show all" (`main.js` `ensureAllRows` loads it as a
    script, which works from a local file too). The `?v=` hash keeps a cached copy from meeting another day's page."""
    rest_path = out_dir / RANKING_REST
    start = index.find(_REST_OPEN)
    if start < 0:
        rest_path.unlink(missing_ok=True)
    else:
        end = index.index("</script>", start)
        rows = index[start + len(_REST_OPEN) : end]
        script = f"window.FQ_RANKING_REST = {json.dumps(rows, ensure_ascii=False)};\n"
        rest_path.write_text(script, encoding="utf-8")
        version = hashlib.sha256(script.encode("utf-8")).hexdigest()[:10]
        stub = f'<script type="text/html" id="ranking-rest" data-src="{RANKING_REST}?v={version}">'
        index = index[:start] + stub + index[end:]
    (out_dir / "index.html").write_text(index, encoding="utf-8")


# Full-ranking rows built into the page's table; the rest wait as inert text until a search, filter, sort or
# "Show all" needs them (`main.js` `ensureAllRows`): ~6 MB of rows made phones parse for seconds.
RANKING_EAGER_ROWS = 500

# The one sentence shown under the Highlights heading; `TOP_LIST_NOTE` follows it in a fold.
TOP_LIST_LEAD = ("Every list leaves out companies whose figures need checking, and companies with fewer than "
                 f"{sc.RECORD_MIN_YEARS} annual reports or listed under {sc.LISTED_MIN_YEARS} years. Those that fall "
                 "short on the moat and "
                 "management grade (see above), or whose numbers an AI check finds flattered, are folded away under "
                 "each list, not deleted.")  # fmt: skip
# Under the Highlights heading, once for every list (`ranking.trusted`).
TOP_LIST_NOTE = ("Scores are percentiles (0 to 100). Every list leaves out suspect figures, peak-cycle margins, "
                 "distressed balance sheets, collapsing earnings, declining revenue, earnings far ahead of cash, shares that trade under {liquid} a day, companies with fewer than {record_years} annual reports on file or a share or dividend history under {listed_years} years (a new issue's seller picks the date and the figures), and business "
                 "models still to confirm. The rankings hold profitable operating companies only: banks, insurers, "
                 "REITs and utilities are judged against their own kind in the By kind lists, holdings in Below "
                 "book. Every list also leaves out "
                 "the industries AI substitutes or disrupts (software, IT services, consulting, staffing, advertising, "
                 "publishing, education, business services, internet content, gaming, health information, entertainment "
                 "and broadcasting). Cheap and good, Cheapest and Best businesses are pure score rankings (the two "
                 "named cheap also ask an owner yield on the share price above {cheap_yield}%); the other "
                 "lists keep only the best {top} of the combined ranking (of its own kind, for the By kind lists). "
                 "Each company a "
                 "list shows is then graded 1 to 5 on its moat and its management by an AI model (Gemini), from its "
                 "own knowledge and this site's figures: those with both grades at least {both} (Boring businesses, "
                 "In trouble now, Cheapest and Best businesses set bars of their own), are the list; the others "
                 "stay on it, folded away under the table, and the list takes the next company that qualifies in "
                 "their place, so its rows that pass stay full while enough qualify. Cheap and good, Cheapest and "
                 "Best businesses walk further down their ranking until at least {min_pass} pass every test. A grade changes no score "
                 "and no rank.")  # fmt: skip


# Countries the Boring businesses rule quotes for the sovereign spread (`lists/sovereign_spread.txt`), read at each report.
SPREAD_EXAMPLES = ("Brazil", "South Africa", "India", "China", "Japan", "Germany")


def _rule_values(cfg: Settings, hurdle: float, has_bench: bool, borrow_rate: float, treasury: float | None) -> dict:
    """The thresholds the tabs' sentences quote (`templates/list_rules.html.j2`), formatted once."""
    g = lambda v: f"{v:g}"  # noqa: E731
    return {
        "among": " among the benchmark's companies" if has_bench else "",
        "treasury": "n/a" if treasury is None else f"{treasury:.2f}",
        "top": f"{cfg.highlight_top_fraction:.0%}",
        "n": g(sc.SHORTLIST_N), "fcf": g(sc.SHORTLIST_MIN_FCF_POS), "beneish": g(sc.SHORTLIST_MAX_BENEISH),
        "accruals": g(sc.SHORTLIST_MAX_ACCRUALS), "incremental": g(sc.SHORTLIST_MIN_INCREMENTAL_ROIC),
        "net_debt": g(sc.SHORTLIST_MAX_NET_DEBT_EBIT), "cover": g(sc.SHORTLIST_MIN_INTEREST_COVER),
        "equity": g(sc.SHORTLIST_MIN_EQUITY_ASSETS), "roic": g(sc.SHORTLIST_MIN_ROIC),
        "value": g(sc.SHORTLIST_MIN_VALUE), "quality": g(sc.SHORTLIST_MIN_QUALITY),
        "estimate_cut": g(-cfg.estimate_cut_pct), "revenue_fall": g(-cfg.dislocation_revenue_change_min),
        "ebit_fall": g(-cfg.dislocation_ebit_change_min),
        "net_debt_equity": g(sc.SHORTLIST_MAX_NET_DEBT_EQUITY), "leader_share": g(sc.SHORTLIST_LEADER_SHARE * 100),
        "leader_min": g(sc.SHORTLIST_LEADER_MIN), "hurdle": g(round(hurdle, 1)),
        "spreads": ", ".join(f"{c} {sovereign_spread(c):.1f}" for c in SPREAD_EXAMPLES),
        "grade_both": g(grading.MIN_BOTH), "grade_top": g(grading.SHORTLIST_MIN_TOP),
        "grade_best": g(grading.BEST_MIN_BOTH), "cheapest_both": g(grading.CHEAPEST_MIN_BOTH),
        "cheap_yield": g(sc.CHEAP_MIN_SHARE_YIELD),
        "beta_max": g(sc.LOW_BETA_MAX), "beta_revenue": g(-sc.LOW_BETA_MIN_REVENUE_WORST),
        "beta_gm_sd": g(sc.LOW_BETA_MAX_GROSS_MARGIN_SD), "beta_roic": g(sc.LOW_BETA_MIN_ROIC), "beta_roic_var": g(sc.LOW_BETA_MAX_ROIC_VARIABILITY),
        "trouble_both": g(grading.TROUBLE_MIN_BOTH), "trouble_top": g(grading.TROUBLE_MIN_TOP),
        "asset_min_roic": g(cfg.asset_play_min_roic),
        "deep_pb": g(sc.DEEP_DISCOUNT_MAX_PRICE_TO_BOOK), "deep_mgmt": g(DEEP_MIN_MGMT),
        "net_cash_min": g(cfg.net_cash_min_pct), "income_dip": g(round((1 - sc.INCOME_HELD) * 100)),
        "min_years": cfg.improver_min_years,
        "operating_min_roic": g(sc.OPERATING_MIN_ROIC), "operating_min_yield": g(sc.OPERATING_MIN_YIELD),
        "operating_max_dip": g(sc.OPERATING_MAX_DIP * 100),
        "fast_min_growth": g(sc.FAST_MIN_OE_GROWTH), "fast_min_yield": g(sc.FAST_MIN_YIELD),
        "fast_min_revenue": g(sc.FAST_MIN_REVENUE_GROWTH),
        "list_min_incremental": g(sc.LIST_MIN_INCREMENTAL_ROIC), "max_margin_loss": g(sc.MAX_MARGIN_LOSS),
        "borrow_rate": f"{borrow_rate:.2f}",
        "boring_industries": ", ".join(_read_lines(sc.BORING_INDUSTRIES)).lower(),
        "income_wht": g(sc.INCOME_MAX_WHT), "income_years": sc.INCOME_MIN_YEARS,
        "income_yield": g(sc.INCOME_MIN_YIELD), "income_roic": g(sc.INCOME_MIN_ROIC),
        "income_payout": g(sc.INCOME_MAX_PAYOUT * 100), "income_growth_cap": g(sc.INCOME_GROWTH_CAP),
        "special_ratio": g(dividend_record.SPECIAL_RATIO),
        "recession_fall": g(sc.RECESSION_MAX_FALL),
        "recessions": " and ".join(str(y) for y, _, only in dividend_record.recessions() if not only),
        "recession_exempt": "; ".join(
            f"{y} not for {', '.join(sorted(m + 's' for m in exempt))}" for y, exempt, _ in dividend_record.recessions()
            if exempt
        ),
        "downturns": "; ".join(
            f"{y} for {', '.join(sorted(only)).lower()}" for y, _, only in dividend_record.recessions() if only
        ),
        "min_pass": MIN_ALL_PASS, "the_list_n": THE_LIST_N, "the_list_grade": g(THE_LIST_MIN_GRADE),
        "the_list_cash_mgmt": g(THE_LIST_CASH_MGMT),
        "reit_cover": g(sc.REIT_MIN_INTEREST_COVER), "reit_net_debt": g(sc.REIT_MAX_NET_DEBT_EBIT),
        "utility_cover": g(sc.UTILITY_MIN_INTEREST_COVER), "utility_net_debt": g(sc.UTILITY_MAX_NET_DEBT_EBIT),
    }  # fmt: skip


def _list_rows(h: Highlights, tab: Tab) -> list[dict]:
    """A tab's rows: its top `n`, plus the top `n` among companies of zero-withholding countries marked `extra`
    (shown only by the "No dividend tax" switch, so the list stays full when it is on); below the grade bar last."""
    base, zero = h.shown(tab.name), h.shown(tab.name, zero=True)
    extra = zero[~zero["Ticker"].isin(base["Ticker"])]
    both = pd.concat([base, extra]) if len(extra) else base
    both = both.assign(_fails=fails(both)).sort_values(["_fails", tab.sort], kind="stable")
    rows = summary_rows(both, tab.columns)
    extras = set(extra["Ticker"])
    for r, (_, row) in zip(rows, both.iterrows(), strict=True):
        r["extra"] = r["ticker"] in extras
        r["fails"] = bool(row["_fails"])
        r["pills"].append(grade_pill(row))
        if (check := check_pill(row)) is not None:
            r["pills"].append(check)
        if isinstance(row.get("The_List_Why"), str):  # The List: each list it came from, and its place there
            for seat in row["The_List_Why"].split(" · "):
                src, _, place = seat.rpartition(" #")
                r["pills"].append({"label": seat, "cls": "pill-src",
                                   "why": f"Place {place} among {src}'s companies that pass every test; The List takes "
                                          "each list's best in turn."})  # fmt: skip
        missing = sc.untested(row) if tab.key in sc.UNTESTED_TABS else []
        if missing:
            r["pills"].append({"label": f"untested: {len(missing)}", "cls": "",
                               "why": f"No figure for: {', '.join(missing)}. The test that reads it was passed "
                                      "untested (a missing figure passes); check it in the filings."})  # fmt: skip
    return rows


def _shares(values: pd.Series, weights: pd.Series | None = None) -> pd.Series:
    """Each value's share of the whole, %, largest first (weighted by `weights` when given; blanks left out)."""
    w = pd.Series(1.0, index=values.index) if weights is None else pd.to_numeric(weights, errors="coerce")
    keep = values.notna() & w.notna() & (w > 0)
    totals = w[keep].groupby(values[keep]).sum()
    return (totals / totals.sum() * 100).sort_values(ascending=False) if totals.sum() > 0 else totals


def list_mix(rows: pd.DataFrame, market: pd.DataFrame, top: int = 3) -> list[dict]:
    """The List's mix against the market (user's choice, Nov 9, 2026: over 5 years the countries, sizes and currencies
    can move results as much as the picks): for country, size class (`index_charts.CAP_CLASSES`) and currency, The
    List's largest shares (each row counts once: equal weights, as a reader would buy them) beside the same share of
    `market` weighted by market cap (the benchmark's companies, close to a global index), then "other"."""
    names = [c for c, _, _ in index_charts.CAP_CLASSES]

    def size(df: pd.DataFrame) -> pd.Series:
        caps = pd.to_numeric(df["Market Cap"], errors="coerce")
        return index_charts.market_cap_class(caps).map(lambda k: names[int(k)]).where(caps > 0)

    out = []
    for label, pick in (
        ("Countries", lambda df: df["country"]),
        ("Size", size),
        ("Currencies", lambda df: df["currency"]),
    ):
        try:
            mine, theirs = _shares(pick(rows)), _shares(pick(market), market["Market Cap"])
        except KeyError:  # a frame without the column
            continue
        if mine.empty:
            continue
        parts = [f"{k} {v:.0f}% (market {theirs.get(k, 0.0):.0f}%)" for k, v in mine.head(top).items()]
        rest = mine.iloc[top:].sum()
        if rest > 0.5:
            parts.append(
                f"other {rest:.0f}% (market {100 - theirs.reindex(mine.head(top).index).fillna(0).sum():.0f}%)"
            )
        out.append({"label": label, "text": "; ".join(parts)})
    return out


def tab_views(h: Highlights) -> list[dict]:
    """Every tab as the index page shows it, in `TABS` order: label, panel id, heading, group, prose macro, rows,
    total (the stored `name` stays out of sight, `Tab.label`)."""
    return [{"name": t.label or t.name, "panel": t.panel, "heading": t.label or t.heading or t.name, "key": t.key,
             "group": t.group, "ranked": t.ranked, "rows": _list_rows(h, t), "total": h.total(t.name),
             "headers": headers(t.columns), "funnel": h.funnel(t.name),
             "mix": list_mix(h.shown(t.name), h.frame) if t.key == "the_list" else []} for t in TABS]  # fmt: skip


def tab_groups(top: str) -> dict[str, dict[str, str]]:
    """The rows of tabs (`highlights.GROUPS`): key -> heading and its one line, the best fraction filled in."""
    return {k: {"heading": head, "note": note.format(top=top)} for k, (head, note) in GROUPS.items()}


def _newest_data(df: pd.DataFrame) -> str | None:
    """The run's newest data (price refresh or full fetch), as each company page states its own (`_data_date`)."""
    cols = [c for c in ("Price_At", "Fetched_At") if c in df.columns]
    stamps = (
        pd.concat([pd.to_datetime(df[c], errors="coerce") for c in cols]) if cols else pd.Series(dtype="datetime64[ns]")
    )
    newest = stamps.max()
    return newest.strftime("%Y-%m-%d") if pd.notna(newest) else None


def build_site(
    df: pd.DataFrame,
    cfg: Settings,
    out_dir: Path | None = None,
    bench: pd.DataFrame | None = None,
    held: frozenset[str] = frozenset(),
    index_only: bool = False,
    companies_only: bool = False,
    tickers: set[str] | None = None,
    forward: str | None = None,
    highlights: Highlights | None = None,
) -> Path:
    """Write the full site into `out_dir` (default `cfg.site_dir`); returns the index path.

    `bench` (the ranked benchmark) is drawn behind the run in the quadrant and histograms; `held`
    (the portfolio's tickers) are marked in the tables and on their own pages.
    `index_only`: generate only index.html and static assets, skipping all company pages (~1.5s).
    `companies_only`: generate only company pages in companies/.
    `tickers`: generate only company pages for specific tickers.
    `forward`: the label of the header's link to forward.html (the forward test, `report.forward_page`), None for none.
    `highlights`: the tabs' members when the caller computed them already (`Highlights.compute(df, cfg, bench)`).
    """
    out_dir = Path(out_dir or cfg.site_dir)
    h = highlights or Highlights.compute(df, cfg, bench)
    comp_dir = out_dir / "companies"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Only prune obsolete files that are no longer in df (never wipe valid existing company pages)
    if not index_only and tickers is None and not companies_only:
        valid = {slug(t) for t in df["Ticker"]}
        for old in comp_dir.glob("*.html"):
            if old.stem not in valid:
                old.unlink()

    for name in STATIC_FILES:
        shutil.copy2(TEMPLATE_DIR / name, out_dir / name)

    env = _env()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    df = h.frame
    periods = report_periods(df, cfg)
    # Companies a list shows (extras left out) and the holdings carry their grade card on their page.
    df = df.assign(
        Grade_Listed=df["Ticker"].isin(h.listed() | set(held or ())),
        Grade_Shortlisted=df["Ticker"].isin(h.shown(SHORTLIST)["Ticker"]),
        On_Lists=df["Ticker"].map(h.places()),  # in the row, so a page follows its lists (fingerprint)
    )
    columns = [c for c in summary_columns(df) if c[1] not in HIDDEN_IN_RANKING]
    has_bench = bench is not None and not bench.empty
    top = f"{cfg.highlight_top_fraction:.0%}"

    index = env.get_template("index.html.j2").render(
        date=stamp,
        period=period_range(df, periods),
        n_companies=len(df),
        n_countries=int(df["country"].nunique()) if "country" in df.columns else 0,
        n_ranked=int(df["Rank"].notna().sum()) if "Rank" in df.columns else len(df),
        data_date=_newest_data(df),
        fresh={
            "statements_days": cfg.max_cache_age_days,
            "per_run": cfg.max_full_fetches,
            "price_days": cfg.price_refresh_days,
        },  # fmt: skip
        vq_corr=fmt(index_charts.map_correlation(df), "+.2f"),
        n_map=len(index_charts.map_points(df)),
        bench_note=benchmark_note(df),
        has_bench=has_bench,
        quadrant=index_charts.quadrant_html(df, bench),
        histograms=index_charts.histogram_divs(df, bench, held=held),
        tabs=tab_views(h),
        groups=tab_groups(top),
        rule=_rule_values(cfg, h.hurdle, has_bench, h.borrow_rate, h.treasury_yield),
        top_list_lead=TOP_LIST_LEAD,
        top_list_note=TOP_LIST_NOTE.format(
            top=top,
            liquid=f"${cfg.list_min_traded_usd / 1e3:,.0f}k",
            record_years=sc.RECORD_MIN_YEARS,
            listed_years=sc.LISTED_MIN_YEARS,
            cheap_yield=f"{sc.CHEAP_MIN_SHARE_YIELD:g}",
            both=f"{grading.MIN_BOTH:g}",
            min_pass=MIN_ALL_PASS,
        ),
        industries=sc.industry_averages(df, bench),
        headers=headers(columns, groups=True),
        rows=summary_rows(df, columns, groups=True),
        eager_rows=RANKING_EAGER_ROWS,
        held=held,
        forward=forward,
        held_slugs=sorted(slug(t) for t in held if t in set(df["Ticker"])),
    )
    if not companies_only:
        _write_index(index, out_dir)
        df.to_csv(out_dir / "full_analysis.csv", index=False)

    if index_only:
        log.info("Front page written to %s (skipped company pages)", (out_dir / "index.html").resolve())
        return out_dir / "index.html"

    write_company_pages(df, cfg, out_dir, periods, h.treasury_yield, stamp, held=held, tickers=tickers)
    return out_dir / "index.html"
