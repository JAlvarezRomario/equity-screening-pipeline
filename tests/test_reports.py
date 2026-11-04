import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.html import build_site
from fq.report.links import slug


def _ranked(cfg):
    return apply_dual_ranking(calculate_metrics(make_raw(cfg), cfg), cfg)


def test_html_site(cfg):
    df = _ranked(cfg)
    index = build_site(df, cfg)
    html = index.read_text(encoding="utf-8")
    assert 'id="quadrant_plot"' in html and 'id="growth_plot"' not in html and "report.pdf" not in html
    # Three full top lists, each with the number of analysts, as the growth list (and the full ranking).
    assert all(f"<h3>{name}</h3>" in html for name in ("Cheap and good", "Cheapest", "Best businesses"))
    assert html.count(">Analysts</th>") >= 4
    # Header subtitle saying what the reader gets and the breadth; no Munger quotation, Lynch line or tagline.
    assert "measured with one yardstick" in html and "Charlie Munger" not in html
    assert "What this is for" in html and "not a buy list" in html
    assert "Peter Lynch" not in html and "how likely it is to last" not in html
    for ticker in df["Ticker"]:
        assert f"companies/{slug(ticker)}.html" in html
        page = (index.parent / "companies" / f"{slug(ticker)}.html").read_text(encoding="utf-8")
        assert "dataroma.com/m/stock.php?sym=" in page and "roic.ai/quote/" in page
        assert "company_chart" in page
    # Yahoo text must be escaped, not injected.
    page = (index.parent / "companies" / f"{slug(df.loc[0, 'Ticker'])}.html").read_text(encoding="utf-8")
    assert "<b>things</b>" not in page and "&lt;b&gt;things&lt;/b&gt;" in page
    assert (index.parent / "full_analysis.csv").exists()
    assert 'class="held"' not in html and 'id="holdings-toggle"' not in html and "window.FQ_HELD = []" in html


def test_holdings_are_marked(cfg):
    df = _ranked(cfg)
    held = df.loc[0, "Ticker"]
    index = build_site(df, cfg, held=frozenset({held}))
    html = index.read_text(encoding="utf-8")
    table = html[html.index('id="ranking-table"') :]
    assert table.count('data-held="1"') == 1 and 'id="holdings-toggle"' in html
    # The one Holdings switch: the combined ranking filtered to the holdings, highlighted in the charts.
    assert f'window.FQ_HELD = ["{slug(held)}"]' in html
    assert "fq-holdings" in html  # the quadrant redraws on the switch
    pages = {t: (index.parent / "companies" / f"{slug(t)}.html").read_text(encoding="utf-8") for t in df["Ticker"]}
    assert "In portfolio" in pages[held] and all("In portfolio" not in p for t, p in pages.items() if t != held)


def test_front_page_code_does_not_re_render_company_pages(cfg, monkeypatch, tmp_path):
    """Company pages are re-rendered only when their own inputs change (`company_page.PAGE_INPUTS`): editing the front
    page's code (`html.py`, its lists in `screens.py`, its map), its template or the static files leaves their
    fingerprint alone (Nov 1, 2026: the Repeaters tab re-rendered 7,940 pages); editing the company template does not."""
    from fq.report import company_page

    report = tmp_path / "report"
    templates = report / "templates"
    templates.mkdir(parents=True)
    for name in ("charts.py", "index_charts.py", "format.py", "html.py", "screens.py", "company_page.py"):
        (report / name).write_text("x = 1")
    for name in ("company.html.j2", "index.html.j2", "style.css"):
        (templates / name).write_text("<p>1</p>")
    monkeypatch.setattr(company_page, "_REPORT_DIR", report)
    monkeypatch.setattr(company_page, "TEMPLATE_DIR", templates)
    version = lambda: company_page._page_inputs_version(cfg, ["Y0"], 4.0, {"Rank": 3})  # noqa: E731
    before = version()
    for f in (report / "index_charts.py", report / "html.py", report / "screens.py", templates / "index.html.j2",
              templates / "style.css"):  # fmt: skip
        f.write_text("changed")
    assert version() == before  # front page only: company pages stay as they are
    (templates / "company.html.j2").write_text("<p>2</p>")
    assert version() != before
    before = version()
    (report / "company_page.py").write_text("changed")
    assert version() != before


def test_ranking_rows_beyond_the_first_wait_as_text(cfg, monkeypatch):
    from fq.report import html as html_mod

    df = _ranked(cfg)
    monkeypatch.setattr(html_mod, "RANKING_EAGER_ROWS", 1)
    index = build_site(df, cfg)
    html = index.read_text(encoding="utf-8")
    table = html[html.index('id="ranking-table"') :]
    body = table.split("<tbody>")[1].split("</tbody>")[0]
    stub = table.split('id="ranking-rest"')[1].split("</script>")[0]
    assert 'data-src="ranking-rest.js?v=' in stub and "<tr" not in stub  # the rows are not on the front page
    script = (index.parent / html_mod.RANKING_REST).read_text(encoding="utf-8")
    rest = json.loads(script.removeprefix("window.FQ_RANKING_REST = ").rstrip().removesuffix(";"))
    assert body.count("<tr") == 1 and rest.count("<tr") == len(df) - 1  # every company is still there
    assert f'data-total="{len(df)}"' in table and 'id="ranking-show-all"' in table

    monkeypatch.setattr(html_mod, "RANKING_EAGER_ROWS", len(df))  # everything eager: no stale file left
    build_site(df, cfg)
    assert not (index.parent / html_mod.RANKING_REST).exists()


def test_front_page_and_company_pages_say_the_same(cfg):
    """Every figure the full ranking and a company page both show agrees, every pill of a ticker is a badge on its
    page (one vocabulary, Oct 31, 2026), and a listed company's page names its tabs and row as the index shows them."""
    from bs4 import BeautifulSoup

    df = _ranked(cfg)
    df.loc[0, "heldPercentInsiders"] = 0.3  # owner-led: a pill, so a badge too
    index = build_site(df, cfg)
    soup = BeautifulSoup(index.read_text(encoding="utf-8"), "html.parser")
    table = soup.find(id="ranking-table")
    heads = [th.get_text(strip=True) for th in table.thead.find_all("th")]
    num = lambda s: "".join(ch for ch in (s or "") if ch.isdigit() or ch in ".-")  # noqa: E731
    compared = 0
    for tr in table.tbody.find_all("tr"):
        shown = dict(zip(heads, [td.get_text(" ", strip=True) for td in tr.find_all("td")], strict=True))
        page_file = index.parent / "companies" / tr.find("a")["href"].split("/")[-1]
        page = BeautifulSoup(page_file.read_text(encoding="utf-8"), "html.parser")
        on_page = {s.select_one(".label").get_text(strip=True): s.select_one(".score-num").get_text(strip=True)
                   for s in page.select(".score")}  # fmt: skip
        on_page |= {t.select_one(".tile-label").get_text(strip=True): t.select_one(".tile-value").get_text(strip=True)
                    for t in page.select(".tile")}  # fmt: skip
        on_page |= {c.span.get_text(strip=True): c.get_text(" ", strip=True)[len(c.span.get_text(strip=True)) :]
                    for c in page.select(".headline .chip")}  # fmt: skip
        for col, label in {"Value Score": "Value", "Quality Score": "Quality", "Owner Yield": "Owner yield",
                           "Return Quality": "Return quality", "Market Cap (B$)": "Market cap",
                           "Analysts": "Analysts"}.items():  # fmt: skip
            if num(shown[col]):  # a figure the index shows, the page shows too, the same
                assert num(shown[col]) == num(on_page.get(label)), (page_file.name, col)
                compared += 1
        badges = {b.get_text(strip=True) for b in page.select(".badges .badge")}
        for pill in tr.select(".flag-tag"):
            word = pill.get_text(strip=True).replace("★ ", "").replace("◆ ", "")
            if word in ("bank", "insurer", "holding", "held", "not graded") or word.startswith("moat "):
                continue  # the yardstick is the page's model line, the grade its card, held its portfolio tag
            assert word[0].upper() + word[1:] in badges, (page_file.name, word)
    assert compared >= 4 * len(df)  # the figures were really read, not skipped
    owner_led = (index.parent / "companies" / f"{slug(df.loc[0, 'Ticker'])}.html").read_text(encoding="utf-8")
    assert "Owner-led" in owner_led
    first = (index.parent / "companies" / f"{slug(df.sort_values('Rank')['Ticker'].iloc[0])}.html").read_text(
        encoding="utf-8"
    )
    assert "On the lists" in first and "Cheap and good #1" in first


def test_industry_averages_over_the_benchmark(cfg):
    from fq.report.screens import industry_averages

    df = _ranked(cfg)
    df.loc[[0, 1], "industry"] = "Tools"
    df.loc[[0, 1], "FCFy_Norm"] = [30.0, 0.0]  # Tools averages 15%, above every Software company
    view = industry_averages(df)
    assert [r["industry"] for r in view["rows"]] == ["Tools", "Software - Application"]  # highest FCF yield first
    assert view["rows"][0]["cells"][0] == "15.0%" and view["scope"] is None  # median of two = their midpoint
    assert view["headers"][2] == "FCF Yield Norm (median)"

    bench = df.copy()
    bench.loc[:, "industry"] = "Tools"  # the averages come from the benchmark, the best ranked from the run
    view = industry_averages(df.iloc[[2]], bench)
    assert [(r["industry"], r["n"]) for r in view["rows"]] == [("Tools", len(bench))]
    assert view["rows"][0]["best"] is None and view["scope"] is not None


def test_nav_chart_only_for_flagged_companies(cfg):
    raw = make_raw(cfg, n=2)
    for y in cfg.fiscal_years:
        raw.loc[0, f"Tangible Book Value_{y}"] = 2e10
    df = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    index = build_site(df, cfg)
    pages = {t: (index.parent / "companies" / f"{slug(t)}.html").read_text(encoding="utf-8") for t in df["Ticker"]}
    flagged = raw.loc[0, "Ticker"]
    assert 'id="nav_chart"' in pages[flagged] and "100% = today" in pages[flagged]
    assert all('id="nav_chart"' not in p for t, p in pages.items() if t != flagged)


def test_chart_bars_are_owner_earnings_and_the_caption_says_why(cfg):
    from fq.metrics import calculate_metrics
    from fq.report.charts import company_chart_figure
    from fq.report.format import report_periods as periods_of
    from fq.report.notes import owner_earnings_caption

    raw = make_raw(cfg, n=4)
    raw.loc[1, ["industry", "sector"]] = ["Banks - Regional", "Financial Services"]
    raw.loc[2, ["industry", "sector"]] = ["REIT - Retail", "Real Estate"]
    raw.loc[3, ["industry", "sector"]] = ["Insurance - Property & Casualty", "Financial Services"]
    df = calculate_metrics(raw, cfg)
    row = df.iloc[0]
    fig = company_chart_figure(df, row["Ticker"], cfg, periods_of(df, cfg))
    bars = next(t for t in fig.data if t.name.startswith("Owner earnings yield"))
    i = list(bars.x).index(cfg.latest)
    expected = row[f"Owner_Earnings_{cfg.latest}"] * row["Parent_Share"] / row[f"Ordinary Shares Number_{cfg.latest}"]
    assert bars.y[i] == pytest.approx(expected / row["currentPrice"] * 100)
    assert any(t.name.startswith("Earnings power/share") for t in fig.data)
    texts = [" ".join(t for _, t in owner_earnings_caption(df.iloc[k])) for k in range(4)]
    assert "growth added back" in texts[0] or "bought growth" in texts[0]
    assert "deposits and loans" in texts[1] and "depreciation" in texts[2] and "Investment gains" in texts[3]


def test_ranking_is_lean_and_company_pages_show_every_scored_criterion(cfg):
    """The combined ranking shows the two scores, earnings power and context only (user's choice, Oct 21, 2026); each
    company page lists every criterion of its model, so the method is never out of sight."""
    from fq.report.format import SUMMARY_COLUMNS, column_help
    from fq.report.tiles import COMPANY_FACTS, company_facts, rank_counts

    assert [c[1] for c in SUMMARY_COLUMNS] == ["Value_Score", "Quality_Score", "Owner_Yield", "Return_Quality", "Market Cap",
                                               "numberOfAnalystOpinions", "Above_52W_Low", "Dividend_WHT"]  # fmt: skip
    scored = {c for crit in (cfg.value_criteria, cfg.quality_criteria, cfg.financial_value_criteria,
                             cfg.financial_quality_criteria) for c, *_ in crit}  # fmt: skip
    assert scored <= {c[1] for c in COMPANY_FACTS}
    assert "Higher is better" in column_help("Return_Quality") and "Higher is better" in column_help("Owner_Yield")
    assert "Scored" not in column_help("Above_52W_Low")

    df = _ranked(cfg)
    groups = dict(company_facts(df.iloc[0], counts=rank_counts(df)))
    quality = [f["label"] for f in groups["Quality criteria"]]
    assert len(quality) == len(cfg.quality_criteria) and quality[0] == "Return quality"
    assert [f["label"] for f in groups["Value criteria"]] == ["Owner yield"]
    rank = groups["Ranking"][0]
    assert rank["label"] == "Rank" and rank["value"].endswith(f" of {df['Rank'].notna().sum()}")

    html = build_site(df, cfg).read_text(encoding="utf-8")
    head = html[html.index('id="ranking-table"') :].split("</thead>")[0]
    assert ">Value Score</th>" in head and ">Owner Yield</th>" in head and 'data-help="' in head
    assert ">ROIC Norm</th>" not in head and ">After-tax Yield</th>" not in head


def test_company_page_leads_with_scores_tiles_and_neighbours(cfg):
    """Scores with an exact sentence, six key tiles not repeated below, flags only when set, and previous / next in
    the full ranking's order."""
    from fq.report.pills import company_flags
    from fq.report.tiles import KEY_TILES, key_tiles, score_position

    assert score_position(80, 60) == "Above the median company on both Value and Quality."
    assert score_position(20, 77) == "Below the median company on Value, above it on Quality."
    assert score_position(50, 10) == "At the median company on Value, below it on Quality."
    assert score_position(None, 10).startswith("Not enough data")

    df = _ranked(cfg)
    df.loc[1, "website"] = "https://example.com"
    row = df.iloc[0].copy()
    assert [t["label"] for t in key_tiles(row)] == [t[0] for t in KEY_TILES]
    row["Financial_Mode"] = True
    assert "Equity / assets" in [t["label"] for t in key_tiles(row)]
    for flag in ("Dislocation", "Peak_Margin", "Distress", "Data_Suspect", "Insufficient_Data"):
        row[flag] = False
    row["Cyclical"] = True
    assert [f[0] for f in company_flags(row)] == ["Cyclical"]

    index = build_site(df, cfg)
    tickers = list(df["Ticker"])
    page = (index.parent / "companies" / f"{slug(tickers[1])}.html").read_text(encoding="utf-8")
    assert f'href="{slug(tickers[0])}.html" rel="prev"' in page and f'href="{slug(tickers[2])}.html" rel="next"' in page
    first = (index.parent / "companies" / f"{slug(tickers[0])}.html").read_text(encoding="utf-8")
    assert 'rel="prev"' not in first
    drivers = page[page.index('aria-label="What the scores are built from"') :].split("</section>")[0]
    context = page[page.index('aria-label="Key figures"') :].split("</section>")[0]
    rest = page[page.index('class="card chart-card"') :]
    # Every scored criterion is a driver tile with how it ranks; the context tiles follow; neither is repeated below.
    assert "Owner yield" in drivers and "Return quality" in drivers and "better than" in drivers
    assert "Earnings power / EV" in context and "<dt>Owner yield</dt>" not in rest
    assert "<dt>Earnings power / EV, before interest</dt>" not in rest and "Value criteria" not in rest
    assert 'id="company_chart"' in page and 'id="returns_chart"' in page and 'id="business_chart"' in page
    assert 'class="company-website"' in page and 'href="https://example.com"' in page


def test_combined_ranking_shows_no_rank_columns_and_no_implied_return(cfg):
    html = build_site(_ranked(cfg), cfg).read_text(encoding="utf-8")
    head = html[html.index('id="ranking-table"') :].split("</thead>")[0]
    assert ">Value Score</th>" in head and "Value Rank" not in head and "Quality Rank" not in head
    assert "Implied" not in html  # removed on Oct 21, 2026 (user's choice)


def test_every_highlight_list_says_how_many_companies_qualified(cfg):
    html = build_site(_ranked(cfg), cfg).read_text(encoding="utf-8")
    panels = html[html.index('id="highlights"') :].split('id="map"')[0].split("<section")[1:]
    # Three top lists, Shortlist, Repeaters, Operating companies, Fast growers, Cannibals, Below book, Net-nets,
    # Net cash, Off the radar, Dividend carry, Temporary trouble.
    assert len(panels) == 14
    assert all('class="list-count muted"' in p for p in panels)
    ranked = [p for p in panels if any(f'id="{i}"' in p for i in ("hl-top-1", "hl-top-2", "hl-top-3"))]
    assert len(ranked) == 3 and all("eligible companies" in p and "qualif" not in p for p in ranked)
    assert all("qualif" in p for p in panels if p not in ranked)
    # Three rows of tabs; two shown by their plain label, stored under their frozen name.
    assert all(f'class="tab-group" aria-hidden="true">{g}<' in html for g in ("Start here", "By style", "Raw scores"))
    assert "Untaxed dividends" in html and ">Dividend carry<" not in html
    assert "Operating companies" in html and "Fast growers" in html and "Steady growers, net cash" not in html


def test_revenue_and_shares_axes_start_at_zero(cfg):
    from fq.report.charts import business_figure
    from fq.report.format import report_periods

    df = _ranked(cfg)
    fig = business_figure(df, df.loc[0, "Ticker"], cfg, report_periods(df, cfg))
    assert fig.layout.yaxis.rangemode == "tozero" and fig.layout.yaxis2.rangemode == "tozero"
    assert fig.layout.yaxis2.range is None  # no zoomed range: a 1% change looks like 1%
    assert fig.layout.yaxis.showline is True and fig.layout.yaxis2.showline is True
    assert fig.layout.yaxis.title.text.startswith("Revenue")
    assert fig.layout.yaxis2.title.text == "Shares (M)"


def test_returns_figure_financial_mode_has_no_operating_margin(cfg):
    from fq.report.charts import returns_figure

    df = _ranked(cfg).copy()
    df.loc[0, "Financial_Mode"] = True
    fig = returns_figure(df, df.loc[0, "Ticker"], cfg, ["Y1", "Y0"])
    trace_names = [t.name for t in fig.data]
    assert "ROE" in trace_names
    assert "Operating margin" not in trace_names


def test_market_cap_histogram_counts_every_company_on_a_log_scale(cfg):
    import numpy as np

    from fq.report.index_charts import CAP_STEP, histogram_divs, market_cap_bins, market_cap_figure

    df = _ranked(cfg)
    df["Market Cap"] = [0.4, 3.0, 12.0, 85.0, 640.0, 4200.0][: len(df)]
    fig = market_cap_figure(df)
    assert abs(sum(fig.data[0].y) - 100) < 1e-9  # percentages summing to 100
    edges = market_cap_bins(df["Market Cap"])
    assert np.allclose(np.diff(edges), CAP_STEP) and 10 ** edges[0] <= 0.4 and 10 ** edges[-1] >= 4200
    assert list(fig.layout.xaxis.ticktext)[:3] == ["0.1", "1", "10"] and fig.layout.yaxis.title.text == "% of companies"
    assert any("companies" in t for t in fig.data[0].text)  # hover names the range and the count

    # With a benchmark: both in the same bins, as percentages of their own companies.
    bench = df.copy()
    bench["Market Cap"] = bench["Market Cap"] * 3
    both = market_cap_figure(df, bench)
    assert len(both.data) == 2 and abs(sum(both.data[0].y) - 100) < 1e-9 and abs(sum(both.data[1].y) - 100) < 1e-9
    assert both.data[0].x.tolist() == both.data[1].x.tolist()
    assert market_cap_figure(df.drop(columns="Market Cap")) is None

    divs = histogram_divs(df)
    titles = [t for t, _ in divs]
    assert titles[-1] == "Market cap (B$, log scale)"
    caps = int((pd.to_numeric(df["Market Cap"], errors="coerce") > 0).sum())
    assert f"n = {caps:,}" in divs[-1][1] and all("n = " in div for _, div in divs)  # every chart says its count
    html = build_site(df, cfg).read_text(encoding="utf-8")
    assert 'id="hist_cap"' in html and f"{len(titles)} measures" in html


def test_every_page_carries_the_disclaimers_and_no_csv_link(cfg):
    index = build_site(_ranked(cfg), cfg)
    pages = [index, *sorted((index.parent / "companies").glob("*.html"))[:1]]
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert "Not investment, financial, tax or legal advice" in html
        assert "Yahoo Finance" in html
        assert 'content="noindex,nofollow"' in html
    assert "full_analysis.csv" not in index.read_text(encoding="utf-8")


def test_histograms_superimpose_holdings_in_orange(cfg):
    from fq.report.index_charts import HOLDINGS_COLOR, histogram_divs, market_cap_figure

    df = _ranked(cfg)
    held = frozenset([df.loc[0, "Ticker"]])
    cap_fig = market_cap_figure(df, held=held)
    assert len(cap_fig.data) == 2
    held_trace = cap_fig.data[1]
    assert held_trace.name == "Holdings"
    assert held_trace.visible is False
    assert held_trace.marker.color == HOLDINGS_COLOR
    assert abs(sum(held_trace.y) - 100) < 1e-9

    divs = histogram_divs(df, held=held)
    assert len(divs) >= 3
    for _title, div in divs:
        assert "Holdings" in div
        assert HOLDINGS_COLOR in div

    html = build_site(df, cfg, held=held).read_text(encoding="utf-8")
    assert "Holdings switch adds your portfolio in orange" in html


def test_treasury_guide_line_replaces_six_percent(cfg):
    from fq.metrics import calculate_metrics
    from fq.report.charts import company_chart_figure
    from fq.report.format import report_periods as periods_of

    raw = make_raw(cfg, n=2)
    df = calculate_metrics(raw, cfg)
    row = df.iloc[0]
    # Test with treasury yield well separated from earnings power yield (~3.8%)
    ty = 8.0
    fig = company_chart_figure(df, row["Ticker"], cfg, periods_of(df, cfg), treasury_yield=ty)
    shapes = fig.layout.shapes
    # A dashed horizontal guide at 8.0% (10Y treasury) and a dotted guide at 15%
    hlines = [s for s in shapes if s.type == "line" and s.y0 == s.y1]
    y_levels = [s.y0 for s in hlines]
    assert ty in y_levels
    assert 6 not in y_levels
    assert 15 in y_levels

    # Check that the 10Y treasury line is dashed with width 1.5
    ty_line = next(s for s in hlines if s.y0 == ty)
    assert ty_line.line.dash == "dash"
    assert ty_line.line.width == 1.5

    # Check annotation
    texts = [a.text for a in fig.layout.annotations if a.text]
    assert any("10Y" in t for t in texts)


def test_mobile_and_theme_adaptations(cfg):
    df = _ranked(cfg)
    index = build_site(df, cfg)
    html = index.read_text(encoding="utf-8")
    assert 'class="card dot-chart-card" id="map"' in html
    assert 'class="card dot-chart-card"' in html
    assert 'class="nav-map"' in html

    css = (index.parent / "style.css").read_text(encoding="utf-8")
    assert "@media (prefers-color-scheme: dark)" in css
    assert ".dot-chart-card" in css
    assert ".nav-map" in css

    js = (index.parent / "main.js").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in js
    assert "offsetParent" in js


def test_chart_years_skips_unpopulated_history_and_keeps_data(cfg):
    import numpy as np
    import pandas as pd

    from fq.report.charts import business_figure, returns_figure
    from fq.report.format import chart_years

    # Company lacking Y4 data (e.g. 2021)
    row_no_y4 = pd.Series(
        {
            "Ticker": "TEST_NO_Y4",
            "PeriodEnd_Y4": "2021-12-31",
            "Total Revenue_Y4": np.nan,
            "Net Income_Y4": np.nan,
            "ROIC_Y4": np.nan,
            "PeriodEnd_Y3": "2022-12-31",
            "Total Revenue_Y3": 100.0,
            "Net Income_Y3": 10.0,
            "ROIC_Y3": 15.0,
            "PeriodEnd_Y2": "2023-12-31",
            "Total Revenue_Y2": 110.0,
            "Net Income_Y2": 12.0,
            "ROIC_Y2": 16.0,
            "PeriodEnd_Y1": "2024-12-31",
            "Total Revenue_Y1": 120.0,
            "Net Income_Y1": 14.0,
            "ROIC_Y1": 17.0,
            "PeriodEnd_Y0": "2025-12-31",
            "Total Revenue_Y0": 130.0,
            "Net Income_Y0": 15.0,
            "ROIC_Y0": 18.0,
        }
    )
    periods = ["Y4", "Y3", "Y2", "Y1", "Y0"]
    years_no_y4 = chart_years(row_no_y4, periods)
    assert years_no_y4 == ["Y3", "Y2", "Y1", "Y0"]

    # Company with Y4 data
    row_with_y4 = row_no_y4.copy()
    row_with_y4["Total Revenue_Y4"] = 90.0
    row_with_y4["ROIC_Y4"] = 14.0
    years_with_y4 = chart_years(row_with_y4, periods)
    assert years_with_y4 == ["Y4", "Y3", "Y2", "Y1", "Y0"]

    # Missing PeriodEnd (e.g. n/a)
    row_missing_date = row_no_y4.copy()
    row_missing_date["PeriodEnd_Y3"] = np.nan
    assert chart_years(row_missing_date, periods) == ["Y2", "Y1", "Y0"]

    # Verify connectgaps=True in returns_figure and business_figure
    df = pd.DataFrame([row_with_y4])
    r_fig = returns_figure(df, "TEST_NO_Y4", cfg, periods)
    for tr in r_fig.data:
        assert tr.connectgaps is True

    b_fig = business_figure(df, "TEST_NO_Y4", cfg, periods)
    shares_trace = next(t for t in b_fig.data if t.name == "Shares (M)")
    assert shares_trace.connectgaps is True


def _cannibal_ready(df, cfg, rows):
    """Rows that pass every Cannibal test: 12%/yr fewer shares, bought back from free cash flow, cheap and good."""
    for i in rows:
        df.loc[i, ["Share_Reduction_Rate", "Value_Score", "Quality_Score", "ROIC_Norm", "Excess_Cash"]] = [
            12.0, 90.0, 80.0, 15.0, 0.0]  # fmt: skip
        df.loc[i, ["Data_Suspect", "Distress", "Cyclical"]] = False
        df.loc[i, "Ordinary Shares Number_TTM"] = 1e8
        for y in cfg.annual_years:
            df.loc[i, f"FCF_Reported_{y}"] = 100.0
            df.loc[i, f"Repurchase Of Capital Stock_{y}"] = -80.0
            df.loc[i, f"Ordinary Shares Number_{y}"] = 1e8  # the paid test sees no shares retired
    return df


def test_cannibals_screen_and_list(cfg):
    from fq.ranking import cannibal
    from fq.report.html import build_site
    from fq.report.screens import cannibal_list

    df = _cannibal_ready(_ranked(cfg).copy(), cfg, range(5))
    df.loc[1, "Share_Reduction_Rate"] = 2.0  # below 7%
    df.loc[2, "Cyclical"] = True  # an upswing pays for buybacks at the top
    df.loc[3, "Quality_Score"] = 40.0  # a buyback in a poor business only shrinks it
    df.loc[4, ["Ordinary Shares Number_TTM", f"Ordinary Shares Number_{cfg.latest}"]] = 1e8 / 6  # sixfold fewer
    df.loc[4, "Market Cap"] = 1.0  # B$: nothing bought, so a consolidation
    for y in cfg.annual_years:
        df.loc[4, f"Repurchase Of Capital Stock_{y}"] = 0.0

    flagged = cannibal(df, cfg)
    assert list(flagged.iloc[:5]) == [True, False, False, False, False]
    df.loc[4, [f"Repurchase Of Capital Stock_{y}" for y in cfg.annual_years]] = np.nan
    assert bool(cannibal(df, cfg).iloc[4])  # no buyback figure at all: missing is not bad

    df["Cannibal"] = flagged
    clist = cannibal_list(df)
    assert len(clist) >= 1
    assert clist.iloc[0]["Ticker"] == df.loc[0, "Ticker"]

    html = build_site(df, cfg).read_text(encoding="utf-8")
    assert "hl-cannibals" in html
    assert "The Cannibals" in html


def test_company_charts_responsiveness_and_axes(cfg):
    from fq.report.charts import PAGE_LAYOUT, business_figure, returns_figure
    from fq.report.format import report_periods

    assert PAGE_LAYOUT.get("autosize") is True

    df = _ranked(cfg)
    ticker = df.iloc[0]["Ticker"]
    periods = report_periods(df, cfg)

    rf = returns_figure(df, ticker, cfg, periods)
    assert rf.layout.yaxis.automargin is True
    assert rf.layout.xaxis.automargin is True

    bf = business_figure(df, ticker, cfg, periods)
    assert bf.layout.yaxis.automargin is True
    assert bf.layout.yaxis2.automargin is True
    assert bf.layout.xaxis.automargin is True
    assert bf.layout.yaxis2.side == "right"
    assert bf.layout.margin.r >= 65


def test_build_site_versatility_index_only_and_tickers(cfg):
    df = _ranked(cfg)
    out_dir = cfg.site_dir / "test_versatile"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_site(df, cfg, out_dir=out_dir, index_only=True)
    assert index.exists()
    assert (out_dir / "index.html").exists()
    assert len(list((out_dir / "companies").glob("*.html"))) == 0

    sample_ticker = df.loc[0, "Ticker"]
    build_site(df, cfg, out_dir=out_dir, tickers={sample_ticker})
    pages = list((out_dir / "companies").glob("*.html"))
    assert len(pages) == 1
    assert pages[0].stem == slug(sample_ticker)


def test_cannibal_buybacks_must_be_paid_from_free_cash_flow(cfg):
    from fq.ranking import cannibal

    df = _cannibal_ready(_ranked(cfg).copy(), cfg, range(4))
    for y in cfg.annual_years:
        df.loc[1, f"Repurchase Of Capital Stock_{y}"] = -150.0  # more than the cash it made: borrowed
    df.loc[2, "NetDebt_to_EBIT"] = 9.0  # leverage is not tested: Value prices it
    df.loc[3, "Excess_Cash"] = 1000.0
    for y in cfg.annual_years:
        df.loc[3, f"Repurchase Of Capital Stock_{y}"] = -150.0  # more than the cash made, but a cash pile pays it
    flagged = cannibal(df, cfg)
    assert list(flagged.iloc[:4]) == [True, False, True, True]


def test_net_cash_list(cfg):
    from fq.ranking import trusted
    from fq.report.screens import net_cash_list

    df = _ranked(cfg).copy()
    df = df[df["Rank"].notna() & trusted(df)].head(5).copy()
    df.loc[:, ["Net_Cash_to_Cap", "Share_Reduction_Rate", "FCF_Positive_Share"]] = [40.0, 1.0, 100.0]
    df["In_Best_Fraction"] = [True, True, True, True, False]  # the last is outside the best fifth
    rows = df.index
    df.loc[rows[1], "Net_Cash_to_Cap"] = 10.0  # too little cash
    df.loc[rows[2], "Share_Reduction_Rate"] = -3.0  # issuing shares: the pile is diluted
    df.loc[rows[3], "FCF_Positive_Share"] = 50.0  # the pile is burned
    assert list(net_cash_list(df, len(df))["Ticker"]) == [df.loc[rows[0], "Ticker"]]
    assert 'data-panel="hl-net-cash">Net cash</button>' in build_site(_ranked(cfg), cfg).read_text(encoding="utf-8")


def test_net_net_list_needs_net_current_assets_above_the_price_without_the_best_fraction(cfg):
    from fq.ranking import trusted
    from fq.report.pills import tags
    from fq.report.screens import net_net_list

    df = _ranked(cfg).copy()
    df = df[df["Rank"].notna() & trusted(df)].head(5).copy()
    df.loc[:, ["NCAV_to_Price", "Owner_Yield", "Share_Reduction_Rate"]] = [120.0, 5.0, 0.0]
    df["Financial_Mode"] = False
    df["In_Best_Fraction"] = False  # no best-fraction cut: net-nets rarely score high on Quality
    rows = df.index
    df.loc[rows[1], "NCAV_to_Price"] = 80.0  # the fixed assets are not free
    df.loc[rows[2], "Owner_Yield"] = -1.0  # burning the net current assets
    df.loc[rows[3], "Share_Reduction_Rate"] = -3.0  # diluting them
    df.loc[rows[4], "Financial_Mode"] = True  # a lender's current assets mean nothing
    assert list(net_net_list(df, len(df))["Ticker"]) == [df.loc[rows[0], "Ticker"]]
    assert "net-net" in [t["word"] for t in tags(df.loc[rows[0]])]
    assert "net-net" not in [t["word"] for t in tags(df.loc[rows[1]])]
    html = build_site(_ranked(cfg), cfg).read_text(encoding="utf-8")
    assert 'data-panel="hl-net-nets">Net-nets</button>' in html and "Graham&#39;s net-nets" in html


def test_list_filters_for_reinvestment_and_pricing_power(cfg):
    from fq.report.screens import LIST_MIN_INCREMENTAL_ROIC, MAX_MARGIN_LOSS, _earns_its_keep, _margin_held

    df = pd.DataFrame({
        "Incremental_ROIC": [LIST_MIN_INCREMENTAL_ROIC, LIST_MIN_INCREMENTAL_ROIC - 1, np.nan],
        "Gross_Margin_Held": [-MAX_MARGIN_LOSS, -MAX_MARGIN_LOSS - 0.5, np.nan],
    })  # fmt: skip
    assert list(_earns_its_keep(df)) == [True, False, True]  # a blank passes, as in every list
    assert list(_margin_held(df)) == [True, False, True]


def test_shortlist_applies_every_test_and_counts_its_funnel(cfg):
    from fq.ranking import trusted
    from fq.report.screens import shortlist_funnel, shortlist_list

    df = _ranked(cfg).copy()
    one = df[df["Rank"].notna() & trusted(df)].head(1)
    df = pd.concat([one] * 22, ignore_index=True)
    df["Ticker"] = [f"T{i}" for i in range(len(df))]
    df["Rank"] = np.arange(1.0, len(df) + 1)
    clean = {"Per_Share_Compounding": 5.0, "Share_Reduction_Rate": 1.0, "Owner_Earnings_Compounding": 8.0,
             "FCF_Positive_Share": 100.0, "Beneish_M": -2.5, "Accruals": -3.0, "Incremental_ROIC": 20.0,
             "NetDebt_to_EBIT": 1.0, "Interest_Coverage": 10.0, "Equity_to_Assets": 10.0, "EPS_Rev_90d": 0.0,
             "Revenue_Change_Recent": 2.0, "EBIT_Change_Recent": 2.0, "ROIC_Norm": 20.0, "Value_Score": 80.0,
             "Quality_Score": 80.0, "Financial_Mode": False, "industry": "Drug Manufacturers - General",
             "Business_Model": "operating", "Cyclical": False, "Industry_Leader": False, "Net_Debt": -100.0,
             "Common Stock Equity_TTM": 1000.0, "Common Stock Equity_Y0": 1000.0, "Owner_Yield": 8.0}  # fmt: skip
    for col, value in clean.items():
        df[col] = value
    df["In_Best_Fraction"] = True
    rows = df.index
    breaks = [
        ("Share_Reduction_Rate", -3.0),  # issuing shares
        ("Owner_Earnings_Compounding", -2.0),  # owner earnings shrinking
        ("FCF_Positive_Share", 50.0),  # profit not turned into cash
        ("Beneish_M", -1.0),  # manipulation warning
        ("Accruals", 8.0),  # earnings well ahead of cash
        ("Incremental_ROIC", 4.0),  # growth that does not earn its keep
        ("NetDebt_to_EBIT", 5.0),  # leveraged (and interest cover below)
        ("EPS_Rev_90d", -12.0),  # estimates cut
        ("ROIC_Norm", 8.0),  # low returns
    ]
    for row, (col, value) in zip(rows[1:], breaks, strict=False):
        df.loc[row, col] = value
    df.loc[rows[7], "Interest_Coverage"] = 2.0
    df.loc[rows[10], ["Financial_Mode", "Equity_to_Assets"]] = [True, 3.0]  # a thinly capitalised bank
    df.loc[rows[11], ["Beneish_M", "Incremental_ROIC", "Accruals"]] = np.nan  # missing figures pass
    df.loc[rows[12], "In_Best_Fraction"] = False  # outside the best fifth
    df.loc[rows[13], "industry"] = "Information Technology Services"  # fragile to AI
    df.loc[rows[14:17], "industry"] = "Restaurants"  # a global recession hits it hard
    df.loc[rows[15:17], "Industry_Leader"] = True
    df.loc[rows[15], "Net_Debt"] = 100.0  # a leader, but in debt: out (net cash needed)
    df.loc[rows[17], "Net_Debt"] = 600.0  # net debt 60% of equity
    df.loc[rows[18], ["Net_Debt", "Common Stock Equity_TTM", "Common Stock Equity_Y0"]] = [100.0, -50.0, -50.0]
    df.loc[rows[19:21], ["Financial_Mode", "Business_Model", "Net_Debt"]] = [True, "bank", np.nan]
    df.loc[rows[20], "Industry_Leader"] = True  # only the bank leading its industry stays
    df.loc[rows[21], "Owner_Yield"] = 4.0  # a good business that pays less than the bond hurdle
    picked = shortlist_list(df, len(df))
    assert set(picked["Ticker"]) == set(df.loc[[rows[0], rows[11], rows[16], rows[20]], "Ticker"])
    assert shortlist_list(df, len(df), hurdle=9.0).empty  # a Treasury above 6% raises the bar
    assert list(picked["Rank"]) == sorted(picked["Rank"])  # combined rank order
    assert len(shortlist_list(df, 1)) == 1
    funnel = [count for _, count in shortlist_funnel(df)]
    assert "beats the bond" in [label for label, _ in shortlist_funnel(df)]
    assert funnel[0] == len(df) and funnel[-1] == len(picked)
    assert all(a >= b for a, b in zip(funnel, funnel[1:], strict=False))
    html = build_site(_ranked(cfg), cfg).read_text(encoding="utf-8")
    assert 'data-panel="hl-shortlist">Shortlist</button>' in html and 'class="list-funnel muted"' in html


def test_industry_leader_is_the_top_tenth_and_always_the_top_three():
    from fq.report.screens import industry_leader

    big = pd.DataFrame(
        {"Ticker": [f"B{i}" for i in range(50)], "industry": "Big", "Total Revenue_TTM": range(50, 0, -1)}
    )
    small = pd.DataFrame(
        {"Ticker": [f"S{i}" for i in range(5)], "industry": "Small", "Total Revenue_Y0": range(5, 0, -1)}
    )
    banks = pd.DataFrame({"Ticker": ["K0", "K1", "K2", "K3"], "industry": "Banks", "Business_Model": "bank",
                          "Total Revenue_TTM": [1.0, 2.0, 3.0, 4.0], "Total Assets_TTM": [9.0, 8.0, 7.0, 6.0]})  # fmt: skip
    ref = pd.concat([big, small, banks], ignore_index=True)
    leader = dict(zip(ref["Ticker"], industry_leader(ref), strict=True))
    assert [leader[f"B{i}"] for i in range(6)] == [True] * 5 + [False]  # top tenth of 50
    assert [leader[f"S{i}"] for i in range(5)] == [True] * 3 + [False] * 2  # the top 3 of a small industry
    assert [leader[k] for k in ("K0", "K1", "K2", "K3")] == [True, True, True, False]  # banks by assets
    run = ref[ref["Ticker"].isin(["B5", "S0"])]
    assert list(industry_leader(run, ref)) == [False, True]  # judged among the reference, not the run


def test_screened_lists_keep_to_the_best_fifth_top_lists_do_not(cfg):
    from fq.ranking import trusted
    from fq.report.highlights import highlight_picks
    from fq.report.screens import cannibal_list, in_best_fraction
    from fq.settings import Settings

    df = _ranked(cfg).copy()
    best = in_best_fraction(df)
    assert best.sum() == np.ceil(Settings().highlight_top_fraction * df["Rank"].notna().sum())
    df["Cannibal"] = True
    assert set(cannibal_list(df, len(df))["Ticker"]) == set(df.loc[best, "Ticker"])
    picks = highlight_picks(6.0)
    total = {name: len(pick(df, len(df))) for name, (pick, _) in picks.items()}
    assert best.sum() < total["Cheapest"] == total["Best businesses"] <= trusted(df).sum()


def test_highlight_lists_leave_out_models_to_confirm(cfg):
    from fq.ranking import top_n, trusted

    df = _ranked(cfg).copy()
    df["Model_Review"] = None
    best = df.sort_values("Value_Rank").index[0]
    df.loc[best, "Model_Review"] = "holding: operating income is under half of pre-tax income"
    assert not trusted(df)[best]
    assert df.loc[best, "Ticker"] not in set(top_n(df, "Value_Rank", 5)["Ticker"])
    assert df.loc[best, "Ticker"] in set(df["Ticker"])  # still in the full ranking


def test_dividend_withholding_lookup_and_switch(cfg, monkeypatch):
    import fq.universe as universe
    from fq.report.html import build_site
    from fq.report.pills import list_pills

    assert universe.dividend_wht("ANY", "United Kingdom") == 0
    assert universe.dividend_wht("ANY", "Japan") == pytest.approx(15.315)
    assert np.isnan(universe.dividend_wht("ANY", "Atlantis"))
    monkeypatch.setattr(universe, "_dividend_wht_table", lambda: ({"unitedkingdom": 0.0}, {"HSBA.L": 10.0}))
    assert universe.dividend_wht("hsba.l", "United Kingdom") == 10.0  # the issuer's own rate wins
    monkeypatch.undo()

    assert [(p["label"], p["cls"]) for p in list_pills(pd.Series({"Dividend_WHT": 0.0}))] == [("◆ 0% WHT", "pill-wht0")]
    df = _ranked(cfg).copy()
    df["country"] = "United States"
    df.loc[0, "country"] = "Singapore"
    html = build_site(df, cfg).read_text(encoding="utf-8")
    assert 'id="wht-toggle"' in html and html.count('data-wht0="1"') >= 1


def test_bond_hurdle_and_owner_led_pill_and_zero_withholding_badge():
    from fq.report.pills import company_flags, list_pills
    from fq.report.screens import bond_hurdle

    assert bond_hurdle(4.99) == 6.0 and bond_hurdle(7.2) == 7.2 and bond_hurdle(None) == 6.0
    labels = lambda row: [p["label"] for p in list_pills(pd.Series(row))]  # noqa: E731
    assert labels({"heldPercentInsiders": 0.20}) == ["owner-led"]
    assert labels({"heldPercentInsiders": 0.19}) == [] and labels({"heldPercentInsiders": np.nan}) == []
    assert [f[0] for f in company_flags(pd.Series({"Dividend_WHT": 0.0}))] == ["0% WHT"]
    assert company_flags(pd.Series({"Dividend_WHT": 15.0})) == []


def test_temporary_trouble_finds_good_businesses_that_fell_and_stopped_falling(cfg):
    from fq.report.screens import normal_earnings_yield, trouble_funnel, trouble_list

    df = _ranked(cfg).copy()
    one = df[df["Rank"].notna()].head(1)
    df = pd.concat([one] * 14, ignore_index=True)
    df["Ticker"] = [f"T{i}" for i in range(len(df))]
    df["Rank"] = np.arange(1.0, len(df) + 1)
    good = {"EBIT_Change_Recent": -30.0, "Earnings_Collapse": False, "ROIC_Norm": 20.0, "Quality_Score": 70.0,
            "FCF_Positive_Share": 100.0, "Financial_Mode": False, "Business_Model": "operating",
            "industry": "Apparel Retail", "Net_Debt": -100.0, "Common Stock Equity_TTM": 1000.0,
            "EPV_Yield": 9.0, "Pretax Income_Q0": 90.0, "Total Unusual Items_Q0": 0.0, "Pretax Income_Q1": 120.0,
            "Total Unusual Items_Q1": 0.0, "Pretax Income_Q4": 100.0, "Total Unusual Items_Q4": 0.0}  # fmt: skip
    for flag in ("Data_Suspect", "Insufficient_Data", "Industry_Excluded_Kept", "Peak_Margin", "Distress",
                 "Revenue_Decline"):  # fmt: skip
        df[flag] = False
    df["Model_Review"] = None
    for col, value in good.items():
        df[col] = value
    rows = df.index
    df.loc[rows[1], "EBIT_Change_Recent"] = 5.0  # nothing fell
    df.loc[rows[2], "ROIC_Norm"] = 8.0  # not a good business
    df.loc[rows[3], "Revenue_Decline"] = True  # dying
    df.loc[rows[4], "Distress"] = True  # bad balance sheet
    df.loc[rows[5], "industry"] = "Information Technology Services"  # fragile to AI
    df.loc[rows[6], "Pretax Income_Q0"] = 40.0  # still falling: -60% on the year, worse than the -30% drop
    df.loc[rows[7], "EPV_Yield"] = 4.0  # not cheap on normal earnings
    # A bank's EBIT and revenue say little.
    df.loc[rows[8], ["Financial_Mode", "Business_Model"]] = [True, "bank"]
    df.loc[rows[8], "Total Revenue_TTM"] = 0.5 * df.loc[rows[8], "Total Revenue_TTM"]
    df.loc[rows[9], ["EBIT_Change_Recent", "Earnings_Collapse"]] = [np.nan, True]  # a lasting fall in profit
    df.loc[rows[10], ["Pretax Income_Q0", "Pretax Income_Q1", "Pretax Income_Q4"]] = np.nan  # no quarters: passes
    df.loc[rows[11], "Data_Suspect"] = True  # figures to check
    before = df.loc[rows[12], ["Total Revenue_Y0", "Total Revenue_Y1", "Total Revenue_Y2"]].mean()
    df.loc[rows[12], "Total Revenue_TTM"] = 0.8 * before  # customers leaving: 80% of the three years before
    df.loc[rows[13], "Total Revenue_Y0"] = np.nan  # one of the years missing: the other two are enough
    df.loc[rows[13], "Total Revenue_TTM"] = 0.9 * df.loc[rows[13], ["Total Revenue_Y1", "Total Revenue_Y2"]].mean()
    # A lasting fall's yield is rebuilt from its flows without the cap (the synthetic company's is 5.5%), so the
    # EPV_Yield set above is not what it is judged on; a 5% hurdle keeps it.
    assert normal_earnings_yield(df).loc[rows[9]] != 9.0
    picked = trouble_list(df, len(df), hurdle=5.0)
    assert set(picked["Ticker"]) == set(df.loc[[rows[0], rows[9], rows[10], rows[13]], "Ticker"])
    assert list(picked["Rank"]) == sorted(picked["Rank"]) and "Normal_Earnings_Yield" in picked.columns
    assert trouble_list(df, len(df), hurdle=10.0).empty
    steps = trouble_funnel(df, 5.0)
    funnel = [count for _, count in steps]
    assert funnel[0] == len(df) and funnel[-1] == len(picked)
    labels = [label for label, _ in steps]
    held = labels.index("customers still there")
    assert funnel[held - 1] - funnel[held] == 1  # only the company whose revenue fell to 80%
    assert all(a >= b for a, b in zip(funnel, funnel[1:], strict=False))
    # Without a lasting fall the normal yield is the earnings power yield Value already uses.
    assert normal_earnings_yield(df.drop(rows[9])).equals(pd.to_numeric(df.drop(rows[9])["EPV_Yield"]))
    html = build_site(_ranked(cfg), cfg).read_text(encoding="utf-8")
    assert 'data-panel="hl-trouble">Temporary trouble</button>' in html


def test_published_lists_follow_the_tabs_and_the_liquidity_floor(cfg):
    from fq.report.highlights import highlight_picks, listable, published_lists

    df = _ranked(cfg).copy()
    names = set(published_lists(df, cfg)["List"])
    assert names <= set(highlight_picks(6.0)) and "Cheap and good" in names
    first = published_lists(df, cfg).query("List == 'Cheap and good'")["Ticker"].iloc[0]
    df["Traded_Value_USD"] = np.where(df["Ticker"] == first, 1_000.0, np.nan)  # one illiquid, the rest unknown
    assert first not in set(listable(df, cfg)["Ticker"]) and len(listable(df, cfg)) == len(df) - 1
    assert first not in set(published_lists(df, cfg)["Ticker"])
    assert first in set(df["Ticker"])  # still in the full ranking


def test_repeaters_count_lists_order_by_count_then_rank_and_need_a_pass():
    """Repeaters (Oct 31, 2026): on two or more lists, a screen counting every company it admits and a ranked list
    only the rows it shows; most lists first, then the combined rank; only a passing grade enters."""
    from fq.report import highlights as hl

    df = pd.DataFrame({"Ticker": list("ABCDE"), "Rank": [5.0, 1.0, 2.0, 3.0, 4.0],
                       "Grade_Pass": [True, True, True, None, False]})  # fmt: skip
    head = lambda tickers: lambda d, n: d[d["Ticker"].isin(list(tickers))].head(n)  # noqa: E731
    picks = {
        "Cheap and good": (head("BCDE"), 1),  # a ranked list: only B, the row it shows, counts
        "Fast growers": (head("ABCDE"), 1),  # a screen: everyone it admits counts, beyond the row shown
        "Net cash": (head("AD"), 20),
        "Cannibals": (head("AE"), 20),
    }
    every = hl.repeater_candidates(df, picks)
    assert list(every["Ticker"]) == ["A", "B", "D", "E"]  # A on 3; then B, D, E on 2 by rank; C on 1 only
    assert every.set_index("Ticker").at["A", "Repeat_Lists"] == "Fast growers, Cannibals, Net cash"
    shown = hl.repeater_list(df, 20, picks)
    assert list(shown["Ticker"]) == ["A", "B"]  # D ungraded and E below the bar stay out
    assert tuple(hl.highlight_picks(6.0)) == hl.TAB_ORDER and hl.REPEATERS in hl.TAB_ORDER


def test_momentum_chip_coloured_and_plain_tiles():
    """The 6-month momentum chip is green when up and red when down (user's choice, Nov 1, 2026); net cash reads
    "Net cash", not "0.0x"; a company paying nothing reads "None"; a near report counts the days."""
    from fq.report.tiles import headline_facts, key_tiles

    now = pd.Timestamp("2026-11-01")
    row = pd.Series({"Momentum_6M": -19.8, "Above_52W_Low": 6.0, "NetDebt_to_EBIT": -0.4, "Net_Cash_to_Cap": 13.3,
                     "Dividend_Yield": 0.0, "earningsTimestamp": pd.Timestamp("2026-11-12").timestamp()})  # fmt: skip
    chips = {label: (value, cls) for label, value, cls in headline_facts(row, now=now)}
    assert chips["Momentum 6M"] == ("-19.8%", "neg") and chips["Above 52w low"][1] == ""
    assert chips["Next report"] == ("12 Nov 2026 · in 11 days", "soon")
    assert headline_facts(pd.Series({"Momentum_6M": 4.2}), now=now) == [("Momentum 6M", "+4.2%", "pos")]
    tiles = {t["col"]: (t["value"], t["cls"]) for t in key_tiles(row)}
    assert tiles["NetDebt_to_EBIT"] == ("Net cash", "pos") and tiles["Dividend_Yield"] == ("None", "muted")
    debt = {t["col"]: t["value"] for t in key_tiles(pd.Series({"NetDebt_to_EBIT": 2.5, "Net_Cash_to_Cap": -20.0}))}
    assert debt["NetDebt_to_EBIT"] == "2.5x"
    far = pd.Series({"earningsTimestamp": pd.Timestamp("2027-02-01").timestamp()})
    assert headline_facts(far, now=now) == [("Next report", "01 Feb 2027", "")]


def test_front_page_says_how_fresh_each_kind_of_data_is(cfg):
    html = build_site(_ranked(cfg), cfg, index_only=True).read_text(encoding="utf-8")
    intro = html.split('class="card intro"')[1].split("</details>")[0]
    assert "How fresh the data is" in intro and "quarterly" in intro and "month-end" in intro
    assert f"every {cfg.max_cache_age_days} days" in intro and f"{cfg.price_refresh_days} days old" in intro


def test_insider_ownership_histogram_in_percent_with_the_owner_led_guide():
    from fq.report.index_charts import histogram_divs

    df = pd.DataFrame({"Ticker": ["A", "B", "C"], "heldPercentInsiders": [0.05, 0.3, 0.6]})
    (title, div), *_ = histogram_divs(df)
    assert title == "Insider ownership (%)" and "Owner-led" in div and "n = 3" in div
    assert "60" in div and '"end":100' in div


def test_unreliable_figures_are_labelled_on_the_company_page(cfg):
    """Suspect figures: a "Data not reliable" banner with the reason, faded scores, no rank out of N, no median
    sentence and no mood line; a company that is only in decline keeps a normal page."""
    df = _ranked(cfg)
    suspect, declining = df.loc[0, "Ticker"], df.loc[1, "Ticker"]
    for flag in ("Data_Suspect", "Insufficient_Data", "Accounting_Flag", "Revenue_Decline"):
        df[flag] = False
    df["Model_Review"] = None
    df.loc[0, ["Data_Suspect", "Suspect_Reasons"]] = [True, "owner earnings yield above 25%"]
    df.loc[1, "Revenue_Decline"] = True
    index = build_site(df, cfg)
    page = lambda t: (index.parent / "companies" / f"{slug(t)}.html").read_text(encoding="utf-8")  # noqa: E731
    bad, ok = page(suspect), page(declining)
    assert "Data not reliable" in bad and "owner earnings yield above 25%" in bad
    assert 'class="score-panel untrusted"' in bad and "After every trusted company" in bad
    assert "the median company" not in bad and " scored</dd>" not in bad
    assert "Data not reliable" not in ok and 'class="score-panel"' in ok and "the median company" in ok


def test_equity_to_assets_pill_is_not_a_share_of_the_market_cap():
    from fq.report.tiles import net_cash_pill

    financial = pd.Series({"Financial_Mode": True, "Business_Model": "bank", "Equity_to_Assets": 12.0})
    assert net_cash_pill(financial)["sub"] is None
    assert net_cash_pill(pd.Series({"Financial_Mode": False, "Net_Cash_to_Cap": 30.0}))["sub"] == "of cap"


def test_unreliable_companies_are_left_off_the_map(cfg):
    from fq.report.index_charts import quadrant_figure

    df = _ranked(cfg)
    for flag in ("Data_Suspect", "Insufficient_Data", "Accounting_Flag"):
        df[flag] = False
    df["Model_Review"] = None
    fig, meta = quadrant_figure(df)
    shown = sum(len(fig.data[m["trace"]].x) for m in meta)
    df.loc[0, "Data_Suspect"] = True
    fig, meta = quadrant_figure(df)
    assert sum(len(fig.data[m["trace"]].x) for m in meta) == shown - 1
    slugs = [c[0] for m in meta for c in fig.data[m["trace"]].customdata]
    assert slug(df.loc[0, "Ticker"]) not in slugs
