import ast
import json
from pathlib import Path

import pandas as pd

from conftest import make_raw
from fq import grading
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.highlights import Highlights, published_lists
from fq.report.html import build_site
from fq.report.links import slug


def _ranked(cfg):
    return apply_dual_ranking(calculate_metrics(make_raw(cfg), cfg), cfg)


def test_the_gate_is_both_at_three_or_either_at_four_and_a_quarter():
    assert grading.passes(3.0, 3.0) and grading.passes(4.0, 3.5)
    assert grading.passes(2.0, 4.25) and grading.passes(4.5, 1.0)  # one outstanding grade carries it
    assert not grading.passes(2.5, 4.0) and not grading.passes(3.0, 2.75) and not grading.passes(2.0, 2.0)
    assert grading.passes(None, 3.0) is None and grading.passes(float("nan"), 4.0) is None  # ungraded


def test_scores_are_read_from_the_verdict_whatever_the_markup():
    text = "I considered **Economic Moat Score:** 4.0 ...\n### Economic Moat Score: 3.5/5\n**Management Quality Score:** 4.5"
    assert grading.score_after(grading.MOAT_LABEL, text) == 3.5  # the last one: the verdict comes last
    assert grading.score_after(grading.MGMT_LABEL, text) == 4.5
    assert grading.score_after(grading.MGMT_LABEL, "no score here") is None


def test_due_when_new_on_a_newer_annual_report_or_a_year_old():
    now = pd.Timestamp("2026-10-31")
    ranked = pd.DataFrame({"Ticker": ["A", "B", "C", "D"],
                           "PeriodEnd_Y0": pd.to_datetime(["2025-12-31", "2026-06-30", "2025-12-31", "2025-12-31"])})  # fmt: skip
    grades = pd.DataFrame({"Ticker": ["B", "C", "D"],
                           "Graded_At": pd.to_datetime(["2026-10-01", "2025-10-01", "2026-10-01"]),
                           "Y0_End": pd.to_datetime(["2025-06-30", "2024-12-31", "2025-12-31"]),
                           "Prompt_Hash": grading.prompt_hash()})  # fmt: skip
    assert grading.due(["A", "B", "C", "D", "A"], ranked, grades, now) == ["A", "B", "C"]
    assert grading.due(["D"], ranked, grades.assign(Prompt_Hash="older"), now) == ["D"]  # another prompt or page
    assert grading.due(["D"], ranked, grading.load(Path("missing.parquet")), now) == ["D"]


def test_a_new_synthesis_prompt_rewrites_the_text_not_the_grades(tmp_path, monkeypatch):
    now = pd.Timestamp("2026-10-31")
    monkeypatch.setattr(grading, "PASS_HASH_ALIASES", frozenset({"alias"}))
    ranked = pd.DataFrame({"Ticker": ["A"], "shortName": ["Acme"], "PeriodEnd_Y0": pd.to_datetime(["2025-12-31"])})
    path = tmp_path / "grades.parquet"
    old = {"Ticker": "A", "Moat": 3.5, "Management": 4.25, "Synthesis": "Both reports converged.",
           "Report_1": "note one", "Report_2": "note two", "Graded_At": pd.Timestamp("2026-10-30"),
           "Y0_End": pd.Timestamp("2025-12-31"), "Prompt_Hash": "alias",
           "Tokens_In": 100, "Tokens_Out": 50, "Cost_USD": 0.1}  # fmt: skip
    grading.save(pd.DataFrame([old]), path)
    assert grading.due(["A"], ranked, grading.load(path), now) == []  # the passes stand: no costly re-grade
    assert grading.synthesis_due(grading.load(path)) == ["A"]
    seen = []

    def fake_call(client, model, prompt, max_out, label):
        seen.append((model, prompt))
        return "## Overview\n\nAcme makes anvils.", {"in": 10, "cached": 0, "out": 5, "usd": 0.01}

    monkeypatch.setattr(grading, "_call", fake_call)
    assert grading.resynthesize(["A"], ranked, path, client=object()) == 1
    row = grading.load(path).iloc[0]
    assert row["Synthesis"] == "## Overview\n\nAcme makes anvils."
    assert (row["Moat"], row["Management"], row["Graded_At"]) == (3.5, 4.25, pd.Timestamp("2026-10-30"))
    assert row["Prompt_Hash"] == grading.prompt_hash() and row["Synthesis_Hash"] == grading.synthesis_hash()
    assert round(row["Cost_USD"], 2) == 0.11 and grading.synthesis_due(grading.load(path)) == []
    model, prompt = seen[0]
    assert model == grading.SYNTHESIS_MODEL and "note one" in prompt and "3.50/5" in prompt
    # The reader is told about the company only: the prompt no longer asks to cite how the grades were reached.
    assert "converged, say so" not in prompt and "Never mention" in prompt


def test_a_new_synthesis_never_makes_passes_from_another_prompt_look_current(tmp_path, monkeypatch):
    # A capped run grades 60; the rest must stay due, not be stamped current by the synthesis rewrite.
    path = tmp_path / "grades.parquet"
    row = {"Ticker": "A", "Moat": 3.0, "Management": 3.0, "Report_1": "n1", "Report_2": "", "Synthesis": "old",
           "Graded_At": pd.Timestamp("2026-10-30"), "Prompt_Hash": "older", "Synthesis_Hash": "older",
           "Tokens_In": 1, "Tokens_Out": 1, "Cost_USD": 0.1}  # fmt: skip
    grading.save(pd.DataFrame([row]), path)
    assert grading.synthesis_due(grading.load(path)) == []  # its passes are due: re-graded, not rewritten
    monkeypatch.setattr(
        grading, "_call", lambda *a: ("## Overview\n\nNew.", {"in": 1, "cached": 0, "out": 1, "usd": 0})
    )
    grading.resynthesize(["A"], pd.DataFrame({"Ticker": ["A"]}), path, client=object())  # forced (`--resynthesize`)
    assert grading.load(path).iloc[0]["Prompt_Hash"] == "older"
    assert grading.due(["A"], pd.DataFrame({"Ticker": ["A"]}), grading.load(path), pd.Timestamp("2026-11-06")) == ["A"]


class _FakeClient:
    """Answers each grading pass with the next canned report, and the synthesis with a fixed text."""

    def __init__(self, reports):
        self.reports, self.prompts = list(reports), []

    def call(self, client, model, prompt, max_out, label):
        self.prompts.append((model, prompt))
        text = self.reports.pop(0) if model == grading.GRADE_MODEL else "## Overview\n\nSynthesis."
        return text, {"in": 10, "cached": 0, "out": 5, "usd": 0.01}


def _report(moat, mgmt):
    return f"Evidence.\n**{grading.MOAT_LABEL}:** {moat}/5\n**{grading.MGMT_LABEL}:** {mgmt}/5"


def test_the_second_pass_runs_only_when_the_first_clears_the_gate(monkeypatch):
    today = pd.Timestamp("2026-11-06")
    fake = _FakeClient([_report(2.5, 2.5)])  # fails the gate: one pass only
    monkeypatch.setattr(grading, "_call", fake.call)
    one = grading.grade_company(object(), "PERI", "Perion", "page", today)
    assert [m for m, _ in fake.prompts] == [grading.GRADE_MODEL, grading.SYNTHESIS_MODEL]
    assert (one["Moat"], one["Management"], one["Moat_1"]) == (2.5, 2.5, 2.5)
    assert pd.isna(one["Moat_2"]) and pd.isna(one["Mgmt_2"]) and one["Report_2"] == ""
    assert grading.NO_SECOND_NOTE in fake.prompts[-1][1]  # the synthesis is told there is one note only

    fake = _FakeClient([_report(4.0, 3.5), _report(3.0, 3.5)])  # clears it: two passes, averaged
    monkeypatch.setattr(grading, "_call", fake.call)
    two = grading.grade_company(object(), "MA", "Mastercard", "page", today)
    assert [m for m, _ in fake.prompts].count(grading.GRADE_MODEL) == 2
    assert (two["Moat"], two["Management"], two["Moat_2"]) == (3.5, 3.5, 3.0)


def test_the_grading_prompt_asks_for_dependency_and_caps_the_moat():
    prompt = grading.grade_prompt("Acme", "ACM", "- ROIC: 20%", pd.Timestamp("2026-11-06"))  # no stray braces
    assert "- ROIC: 20%" in prompt and "{" not in prompt.replace("{{", "")
    assert "II.B Revenue Dependency & Concentration" in prompt and "Dependency cap" in prompt
    assert "at most 3.0" in prompt and "at most 2.5" in prompt and "mutual lock-in" in prompt
    assert "The page is context, not the verdict" in prompt and "scores and ranks," not in prompt


def test_cost_follows_the_rates_in_force():
    before, after = pd.Timestamp("2026-11-30"), pd.Timestamp("2026-12-01")
    assert round(grading.cost_usd(grading.GRADE_MODEL, 5000, 0, 4000, before), 4) == 0.05
    assert round(grading.cost_usd(grading.GRADE_MODEL, 5000, 0, 4000, after), 4) == 0.10  # Argon doubles Dec 1
    assert grading.estimate_usd(10, after) == 2 * grading.estimate_usd(10, before)


def test_page_markdown_keeps_figures_definitions_and_chart_tables():
    traces = [{"name": "ROIC", "x": ["Y1", "Y0"], "y": {"dtype": "f8", "bdata": "AAAAAAAAJEAAAAAAAAA0QA=="}}]
    layout = {"xaxis": {"tickvals": ["Y1", "Y0"], "ticktext": ["Dec-24", "Dec-25"]}, "shapes": [{"a": 1}, {"b": 2}]}
    html = (f'<main><h1>Acme <span>ACM</span></h1><p>Makes anvils.</p>'
            f'<dl class="facts"><div data-help="Return on capital."><dt>ROIC</dt><dd>20%</dd></div></dl>'
            f'<section class="card grade-card"><p>Moat 5</p></section>'
            f'<script>Plotly.newPlot("returns_chart", {json.dumps(traces)}, {json.dumps(layout)}, {{}})</script></main>'
            f'<footer>Disclaimer</footer>')  # fmt: skip
    md = grading.page_markdown(html)
    assert "## Acme ACM" in md and "Makes anvils." in md and "- ROIC: 20% (Return on capital.)" in md
    assert "| series | Dec-24 | Dec-25 |" in md and "| ROIC | 10 | 20 |" in md  # binary arrays decoded
    assert "Moat 5" not in md and "Disclaimer" not in md  # an earlier grade is never read back


def test_page_markdown_leaves_out_our_conclusions():
    html = ('<main><ul class="badges"><li class="badge" data-help="Mastercard-league">Compounder</li></ul>'
            '<aside class="score-panel"><p class="score-sentence">Above the median.</p><dl class="ranks"><div>'
            '<dt>Rank</dt><dd>3 of 100</dd></div></dl></aside><span class="tag">In portfolio</span>'
            '<dl class="facts"><div data-help="Shares retired per year, %. Higher is better. Scored: Quality, weight 0.1.">'
            '<dt>Shares reduced / yr</dt><dd>+2.0% p82 · w0.1</dd></div></dl>'
            '<section class="callout"><h3>2 data notes</h3><ul class="notes"><li>Cheap (Value 90). Business held.</li>'
            '<li>TTM taken from Yahoo.</li></ul></section>'
            '<section class="card why-here"><p>On the lists Cheap and good #4</p></section>'
            '<div class="tile"><div class="tile-value">14.5%</div><div class="tile-label">Owner yield</div>'
            '<div class="tile-pct">better than 96% · all of Value</div></div></main>')  # fmt: skip
    md = grading.page_markdown(html)
    assert "- Shares reduced / yr: +2.0% (Shares retired per year, %.)" in md and "TTM taken from Yahoo." in md
    assert "- Owner yield: 14.5%" in md  # the figure stays, its rank does not
    for ours in ("Compounder", "Mastercard", "Above the median", "3 of 100", "In portfolio", "p82", "Scored",
                 "better", "Cheap (Value", "2 data notes", "On the lists", "Cheap and good #4", "96%"):  # fmt: skip
        assert ours not in md, ours


def _graded(cfg, df, moat, mgmt):
    """Store a grade for every company: moat / management per ticker from the two maps."""
    rows = [{"Ticker": t, "Moat": moat.get(t, 3.0), "Management": mgmt.get(t, 3.0), "Moat_1": moat.get(t, 3.0),
             "Mgmt_1": mgmt.get(t, 3.0), "Moat_2": moat.get(t, 3.0), "Mgmt_2": mgmt.get(t, 3.0),
             "Synthesis": "## Overview\n\nA **fine** business.\n\n## Moat Justification\n\nWide.",
             "Graded_At": pd.Timestamp("2026-10-31")} for t in df["Ticker"]]  # fmt: skip
    grading.save(pd.DataFrame(rows), cfg.data_dir / "grades.parquet")


def test_a_company_below_the_bar_stays_on_its_list_folded_under_it(cfg):
    df = _ranked(cfg)
    first = published_lists(df, cfg).query("List == 'Cheap and good'")["Ticker"].tolist()
    assert len(first) >= 2 and published_lists(df, cfg)["Grade_Pass"].isna().all()  # nothing graded yet
    low = first[0]
    _graded(cfg, df, {low: 2.0}, {low: 4.0})
    lists = published_lists(df, cfg)
    cheap = lists.query("List == 'Cheap and good'")
    assert set(cheap["Ticker"]) == set(first)  # kept, not replaced
    assert lists.loc[lists["Ticker"] == low, "Grade_Pass"].eq(False).all()
    html = build_site(df, cfg).read_text(encoding="utf-8")
    panel = html.split('id="hl-top-1"')[1].split("</section>")[0]
    main, fold = panel.split('class="explain below-bar"')
    assert f">{low}</a>" not in main and 'class="grade-fail"' not in main  # out of the list's own table
    rows = fold.split("<tbody>")[1].split("<tr")[1:]
    assert len(rows) == 1 and 'class="grade-fail"' in rows[0] and f">{low}</a>" in rows[0]  # in the fold, shaded
    assert "1 below the moat and management bar" in fold and "moat 2.0 · mgmt 4.0" in fold
    assert "moat 3.0 · mgmt 3.0" in panel and "pill-grade-low" in panel


def test_ungraded_companies_keep_their_place_with_a_pill(cfg):
    df = _ranked(cfg)
    html = build_site(df, cfg).read_text(encoding="utf-8")
    assert "not graded" in html and 'class="grade-fail"' not in html


def test_the_grade_card_is_on_listed_companies_pages_only(cfg):
    df = _ranked(cfg)
    listed = set(published_lists(df, cfg)["Ticker"])
    _graded(cfg, df, {}, {})
    index = build_site(df, cfg)
    for ticker in df["Ticker"]:
        page = (index.parent / "companies" / f"{slug(ticker)}.html").read_text(encoding="utf-8")
        assert ("grade-card" in page) == (ticker in listed)
        if ticker in listed:
            assert "Moat Justification" in page and "A fine business." in page and "**" not in page
    assert "written by an AI model (Gemini)" in index.read_text(encoding="utf-8")


def test_scores_and_ranks_never_read_the_grading():
    """The LLM sits outside the method: no module of `fq.metrics`, nor ranking, imports `fq.grading`."""
    src = Path(__file__).parents[1] / "src" / "fq"
    for path in [*src.joinpath("metrics").glob("*.py"), src / "ranking.py", src / "filters.py", src / "settings.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any("grading" in n or "genai" in n for n in names), path.name


def test_holdings_are_graded_listed_or_not(cfg, monkeypatch):
    from fq import pipeline

    df = _ranked(cfg)
    held = "HELD"
    df = pd.concat([df, df.iloc[[0]].assign(Ticker=held)], ignore_index=True)
    seen = {}
    monkeypatch.setattr(pipeline, "portfolio", lambda: frozenset({held, "NOT.RANKED"}))
    monkeypatch.setattr(grading, "due", lambda candidates, *a: seen.setdefault("c", candidates) and [])
    pipeline.grade_companies(cfg, df, Highlights.compute(df, cfg), dry_run=True)
    assert held in seen["c"] and "NOT.RANKED" not in seen["c"]


def test_the_shortlist_asks_both_at_three_and_one_at_four():
    assert grading.passes_shortlist(3.0, 4.0) and grading.passes_shortlist(4.5, 3.25)
    assert not grading.passes_shortlist(3.5, 3.75)  # clears the other lists' bar, not the Shortlist's
    assert not grading.passes_shortlist(2.5, 5.0) and grading.passes(2.5, 5.0)  # one outstanding grade is not enough
    assert grading.passes_shortlist(None, 4.0) is None


def test_new_only_grades_never_graded_companies_and_rewrites_no_synthesis(cfg, monkeypatch):
    # User's rule, Nov 5, 2026: new lists may be graded, but no re-grade is spent.
    from types import SimpleNamespace

    from fq import grading, pipeline

    path = cfg.data_dir / "grades.parquet"
    grading.save(pd.DataFrame({"Ticker": ["OLD"], "Graded_At": [pd.Timestamp("2020-01-01")]}), path)
    sent, rewritten = [], []
    monkeypatch.setattr(pipeline, "portfolio", lambda: set())
    monkeypatch.setattr(grading, "due", lambda candidates, *a: list(candidates))  # both due: OLD by age
    monkeypatch.setattr(grading, "run", lambda todo, *a: sent.extend(todo) or pd.DataFrame({"Cost_USD": []}))
    monkeypatch.setattr(grading, "synthesis_due", lambda grades: ["OLD"])
    monkeypatch.setattr(grading, "resynthesize", lambda stale, *a: rewritten.extend(stale) or len(stale))
    h = SimpleNamespace(grade_pool=lambda: ["NEW", "OLD"])
    ranked = pd.DataFrame({"Ticker": ["NEW", "OLD"]})
    pipeline.grade_companies(cfg, ranked, h, max_grades=150, new_only=True)
    assert sent == ["NEW"] and rewritten == []


def test_best_grades_take_both_at_least_four_and_one_above_by_owner_yield():
    import numpy as np

    from fq.report.highlights import best_grade_list

    assert grading.passes_best(4.0, 4.25) and grading.passes_best(5.0, 4.0)
    assert not grading.passes_best(4.0, 4.0) and not grading.passes_best(3.75, 5.0)
    assert not grading.passes_best(np.nan, 5.0)
    df = pd.DataFrame({
        "Ticker": ["LOW_YIELD", "HIGH_YIELD", "FOURS", "WEAK_MGMT", "UNGRADED"],
        "Grade_Moat": [4.5, 4.0, 4.0, 5.0, np.nan],
        "Grade_Mgmt": [4.5, 4.25, 4.0, 3.75, np.nan],
        "Owner_Yield": [3.0, 8.0, 9.0, 10.0, 11.0],
    })  # fmt: skip
    assert best_grade_list(df, 20)["Ticker"].tolist() == ["HIGH_YIELD", "LOW_YIELD"]


def test_temporary_trouble_turns_over_nothing_breaking_now_and_must_be_graded():
    import numpy as np

    from fq.report import screens as sc

    assert grading.passes_trouble(3.0, 3.5) and not grading.passes_trouble(3.25, 3.25)
    assert not grading.passes_trouble(2.75, 5.0) and not grading.passes_trouble(np.nan, 4.0)
    df = pd.DataFrame({"Ticker": ["FALLING", "FINE"], "Rank": [1.0, 2.0]})
    gates = sc._shortlist_gates
    try:
        sc._shortlist_gates = lambda d, hurdle: [
            ("a boring industry", pd.Series(True, index=d.index)),
            (sc.NOTHING_BREAKING, d["Ticker"] == "FINE"),
        ]
        sc.in_best_fraction, best = (lambda d: pd.Series(True, index=d.index)), sc.in_best_fraction
        try:
            assert sc.trouble_list(df)["Ticker"].tolist() == ["FALLING"]
            assert sc.shortlist_list(df)["Ticker"].tolist() == ["FINE"]
        finally:
            sc.in_best_fraction = best
    finally:
        sc._shortlist_gates = gates


def test_cheapest_walks_down_value_skipping_low_grades_and_holding_the_ungraded():
    import numpy as np

    from fq.report import highlights as hl

    df = pd.DataFrame({
        "Ticker": ["LOW", "GOOD", "NEW", "GOOD2", "GOOD3"],
        "Value_Rank": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Grade_Moat": [2.5, 3.0, np.nan, 4.0, 3.5],
        "Grade_Mgmt": [4.0, 3.0, np.nan, 3.0, 3.5],
    })  # fmt: skip
    out = hl.cheapest_list(df, 2)
    assert out["Ticker"].tolist() == ["GOOD", "NEW", "GOOD2"]  # LOW skipped, NEW waits for its grade
    assert out["Grade_Pass"].tolist() == [True, None, True]
    assert grading.passes_cheapest(np.nan, 3.0) is None and grading.passes_cheapest(3.0, 2.75) is False


def test_best_grades_deducts_sovereign_spread_from_owner_yield():
    from fq.report import highlights as hl

    df = pd.DataFrame(
        {
            "Ticker": ["EM_BANK", "US_GROWER"],
            "Grade_Moat": [4.5, 4.0],
            "Grade_Mgmt": [4.5, 4.25],
            "Owner_Yield": [10.0, 8.5],
            "Sovereign_Spread": [2.0, 0.0],  # EM_BANK net: 8.0%, US_GROWER net: 8.5%
            "Distress": [False, False],
            "Peak_Margin": [False, False],
            "Cyclical": [False, False],
            "Data_Suspect": [False, False],
            "Revenue_Decline": [False, False],
            "Accounting_Flag": [False, False],
            "Earnings_Collapse": [False, False],
        }
    )
    out = hl.best_grade_list(df, 2)
    assert out["Ticker"].tolist() == ["US_GROWER", "EM_BANK"]
