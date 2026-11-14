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
    assert not grading.passes(2.0, 4.25) and not grading.passes(4.5, 1.0)  # Nov 9, 2026: no outstanding-grade route
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
    assert "The figures are context, not the verdict" in prompt and "scores and ranks," not in prompt


def test_cost_follows_the_rates_in_force():
    before, after = pd.Timestamp("2026-11-30"), pd.Timestamp("2026-12-01")
    assert round(grading.cost_usd(grading.GRADE_MODEL, 5000, 0, 4000, before), 4) == 0.05
    assert round(grading.cost_usd(grading.GRADE_MODEL, 5000, 0, 4000, after), 4) == 0.10  # Argon doubles Dec 1
    assert grading.estimate_usd(10, after) == 2 * grading.estimate_usd(10, before)


def _grading_row(**over):
    row = {"Ticker": "ACM", "shortName": "Acme", "country": "Germany", "sector": "Industrials", "industry": "Tools",
           "Business_Model": "operating", "Financial_Mode": False, "longBusinessSummary": "Makes anvils.",
           "fullTimeEmployees": 1200, "heldPercentInsiders": 0.21, "PeriodEnd_Y0": "2025-12-31",
           "PeriodEnd_Y1": "2024-12-31", "PeriodEnd_TTM": "2026-06-30",
           "Total Revenue_Y1": 900e6, "Total Revenue_Y0": 1000e6, "Total Revenue_TTM": 1040e6,
           "EBIT_Y0": 150e6, "ROIC_Y0": 18.0, "ROIC_Norm": 17.5, "Beneish_M": -2.6, "Net Interest Income_Y0": -5e6,
           "DPS_Hist": json.dumps({"2009": 0.5, "2025": 1.25}),
           # Price and everything over a price, the market's mood and our own verdicts: none may reach the grader.
           "currentPrice": 87.65, "marketCap": 4.2e9, "Market Cap": 4.2, "trailingPE": 13.3, "Momentum_6M": 84.8,
           "Above_52W_Low": 97.0, "Owner_Yield": 9.2, "EPV_Yield": 8.1, "Dividend_Yield": 3.1, "BV_to_Price": 41.0,
           "numberOfAnalystOpinions": 7, "EPS_Rev_90d": -4.4, "Dividend_WHT": 26.4, "Sovereign_Spread": 0.28,
           "Value_Score": 91.0, "Quality_Score": 77.0, "Rank": 3.0, "Return_Quality": 15.2, "Compounder": True}  # fmt: skip
    return pd.Series(row | over)


def test_the_grader_reads_the_business_from_the_statements_and_never_a_price():
    md = grading.grade_input(_grading_row())
    assert "## Acme (ACM)" in md and "Makes anvils." in md and "Germany · Industrials · Tools" in md
    assert "| | Dec-24 | Dec-25 | TTM Jun-26 |" in md and "| Revenue | 900 | 1,000 | 1,040 |" in md
    assert "- Normalised return on invested capital (on equity for financials): 17.5%" in md
    assert "- Held by insiders: 21.0%" in md and "| Dividend per share | 0.5 | 1.25 |" in md
    assert "Net interest income" not in md  # an operating company's is the interest it pays, not a lender's income
    for price_or_verdict in ("87.65", "4,200", "13.3", "84.8", "97", "9.2", "8.1", "3.1", "41", "-4.4", "26.4",
                             "0.28", "91", "77", "15.2", "Price", "price", "Market", "P/E", "Momentum", "yield",
                             "Yield", "nalyst", "Score", "Rank", "Compounder", "WHT", "spread"):  # fmt: skip
        assert price_or_verdict not in md, price_or_verdict


def test_a_lender_shows_its_net_interest_income_and_the_input_is_in_the_hash():
    md = grading.grade_input(_grading_row(Business_Model="bank", Financial_Mode=True))
    assert "| Net interest income | – | -5 | – |" in md and "return on equity and book value" in md
    assert "TTM" not in grading.grade_input(_grading_row(TTM_Identical_To_Latest=True))  # TTM repeats Y0: left out
    hash_now = grading.prompt_hash()
    grading.INPUT_VERSION += 1
    try:
        assert grading.prompt_hash() != hash_now  # a new input makes every grade due
    finally:
        grading.INPUT_VERSION -= 1


def test_the_grading_prompt_caps_a_moat_ai_can_replace_and_says_the_price_is_left_out():
    prompt = (grading.PROMPT_DIR / "grade_prompt.md").read_text(encoding="utf-8")
    assert "AI substitution cap" in prompt and "No price, on purpose" in prompt
    assert "{company_figures}" in prompt and "yields are % of today's price" not in prompt


def _graded(cfg, df, moat, mgmt):
    """Store a grade for every company: moat / management per ticker from the two maps."""
    rows = [{"Ticker": t, "Moat": moat.get(t, 3.0), "Management": mgmt.get(t, 3.0), "Moat_1": moat.get(t, 3.0),
             "Mgmt_1": mgmt.get(t, 3.0), "Moat_2": moat.get(t, 3.0), "Mgmt_2": mgmt.get(t, 3.0),
             "Synthesis": "## Overview\n\nA **fine** business.\n\n## Moat Justification\n\nWide.",
             "Graded_At": pd.Timestamp("2026-10-31")} for t in df["Ticker"]]  # fmt: skip
    grading.save(pd.DataFrame(rows), cfg.data_dir / "grades.parquet")


def test_a_company_below_the_bar_stays_on_its_list_folded_under_it(cfg):
    df = _ranked(cfg)
    df["EPV_Per_Share_Yield"] = 10.0  # over the cheap lists' share-price bar: this test is the grade's
    df["Owner_Yield"] = 10.0
    df["ROIC_Norm"] = 20.0
    df["industry"] = "Packaged Foods"
    first = published_lists(df, cfg).query("List == 'Shortlist'")["Ticker"].tolist()
    assert len(first) >= 1 and published_lists(df, cfg)["Grade_Pass"].isna().all()  # nothing graded yet
    low = first[0]
    _graded(cfg, df, {t: 2.0 if t == low else 4.0 for t in df["Ticker"]}, {t: 4.0 for t in df["Ticker"]})
    lists = published_lists(df, cfg)
    assert low not in set(lists.query("List == 'Cheap and good'")["Ticker"])  # Cheap and good walks past a failed grade
    short = lists.query("List == 'Shortlist'")
    assert set(short["Ticker"]) == set(first)  # kept on its screened list, not replaced
    assert lists.loc[lists["Ticker"] == low, "Grade_Pass"].eq(False).all()
    html = build_site(df, cfg).read_text(encoding="utf-8")
    panel = html.split('id="hl-shortlist"')[1].split("</section>")[0]
    main, fold = panel.split('class="explain below-bar"')
    assert f">{low}</a>" not in main and 'class="grade-fail"' not in main  # out of the list's own table
    rows = fold.split("<tbody>")[1].split("<tr")[1:]
    assert len(rows) == 1 and 'class="grade-fail"' in rows[0] and f">{low}</a>" in rows[0]  # in the fold, shaded
    assert "1 below the moat and management bar" in fold and "moat 2.0 · mgmt 4.0" in fold
    assert "pill-grade-low" in panel


def test_ungraded_companies_keep_their_place_with_a_pill(cfg):
    df = _ranked(cfg)
    html = build_site(df, cfg).read_text(encoding="utf-8")
    assert "not graded" in html and 'class="grade-fail"' not in html


def test_the_grade_card_is_on_listed_companies_pages_only(cfg):
    from fq.report.screens import industry_best

    df = _ranked(cfg)
    _graded(cfg, df, {}, {})
    listed = set(published_lists(df, cfg)["Ticker"]) | set(industry_best(df))  # after the grades: walks + industry best
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
    monkeypatch.setattr(pipeline, "watchlist", lambda: frozenset())
    monkeypatch.setattr(grading, "due", lambda candidates, *a: seen.setdefault("c", candidates) and [])
    pipeline.grade_companies(cfg, df, Highlights.compute(df, cfg), dry_run=True)
    assert held in seen["c"] and "NOT.RANKED" not in seen["c"]

    watched = "WATCHED"
    df_w = pd.concat([df, df.iloc[[0]].assign(Ticker=watched)], ignore_index=True)
    seen.clear()
    monkeypatch.setattr(pipeline, "portfolio", lambda: frozenset())
    monkeypatch.setattr(pipeline, "watchlist", lambda: frozenset({watched, "NOT.RANKED"}))
    pipeline.grade_companies(cfg, df_w, Highlights.compute(df_w, cfg), dry_run=True)
    assert watched in seen["c"] and "NOT.RANKED" not in seen["c"]


def test_the_shortlist_asks_both_at_three_and_one_at_four():
    assert grading.passes_shortlist(3.0, 4.0) and grading.passes_shortlist(4.5, 3.25)
    assert not grading.passes_shortlist(3.5, 3.75)  # clears the other lists' bar, not the Shortlist's
    assert not grading.passes_shortlist(2.5, 5.0) and not grading.passes(
        2.5, 5.0
    )  # one outstanding grade is not enough
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


def test_fq_grade_grades_in_waves_until_nothing_changes(cfg, monkeypatch):
    # Nov 7, 2026: new grades put new names on lists filled from graded companies, due in turn (333, 4, 0).
    from fq import pipeline

    waves = iter([333, 4, 0, 99])
    calls = []
    monkeypatch.setattr(pipeline, "load_benchmark_run", lambda cfg: (pd.DataFrame(), None))
    monkeypatch.setattr(pipeline.Highlights, "compute", classmethod(lambda cls, *a: object()))
    monkeypatch.setattr(pipeline, "grade_companies", lambda *a, **k: calls.append(k) or next(waves))
    monkeypatch.setattr(pipeline, "check_companies", lambda *a, **k: 0)  # the checks have tests of their own
    monkeypatch.setattr(pipeline, "write_report", lambda *a, **k: calls.append("report"))
    pipeline.grade_only(cfg)
    assert len(calls) == 4 and calls[-1] == "report"  # three waves, the third changing nothing, then the site
    calls.clear()
    pipeline.grade_only(cfg, max_grades=10)  # a cap is one wave
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(pipeline, "grade_companies", lambda *a, **k: calls.append(k) or 1)
    pipeline.grade_only(cfg, report=False)
    assert len(calls) == pipeline.GRADE_WAVES  # bounded


def test_todays_grades_stay_current_under_the_management_line():
    # User's choice, Nov 7, 2026: the new line grades new companies only; no re-grade is spent.
    assert "ca61a6316a06" in grading.PASS_HASH_ALIASES and grading.prompt_hash() != "ca61a6316a06"
    assert "capital allocation across the whole record" in grading._prompt("grade_prompt.md")
    now = pd.Timestamp("2026-11-07")
    ranked = pd.DataFrame({"Ticker": ["A"], "PeriodEnd_Y0": pd.to_datetime(["2025-12-31"])})
    grades = pd.DataFrame({"Ticker": ["A"], "Graded_At": [pd.Timestamp("2026-11-07")],
                           "Y0_End": [pd.Timestamp("2025-12-31")], "Prompt_Hash": ["ca61a6316a06"]})  # fmt: skip
    assert grading.due(["A"], ranked, grades, now) == []


def test_the_client_has_a_call_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    client = grading._client()
    assert client._api_client._http_options.timeout == grading.CALL_TIMEOUT_MS


def test_best_businesses_walk_down_quality_keeping_both_grades_four_and_one_above():
    """Best businesses (Nov 7, 2026: it absorbed Best grades): down the Quality ranking, both grades at least 4 and
    either above, a lower grade skipped, the ungraded held marked; the order is Quality's, price plays no part."""
    import numpy as np

    from fq.report import highlights as hl

    assert grading.passes_best_walk(4.0, 4.25) and grading.passes_best_walk(5.0, 4.0)
    assert grading.passes_best_walk(4.0, 4.0) is False and grading.passes_best_walk(3.75, 5.0) is False
    assert grading.passes_best_walk(np.nan, 5.0) is None
    df = pd.DataFrame({
        "Ticker": ["FOURS", "BEST", "NEW", "WEAK_MGMT", "GOOD", "LATE"],
        "Quality_Rank": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Grade_Moat": [4.0, 4.5, np.nan, 5.0, 4.0, 5.0],
        "Grade_Mgmt": [4.0, 4.5, np.nan, 3.75, 4.25, 5.0],
        "Owner_Yield": [9.0, 1.0, 9.0, 9.0, 2.0, 20.0],
    })  # fmt: skip
    out = hl.best_business_list(df, 2)
    assert out["Ticker"].tolist() == ["BEST", "NEW", "GOOD"]  # LATE, the cheapest, is never reached
    assert out["Grade_Pass"].tolist() == [True, None, True]
    assert hl.TAB["Best businesses"].group == "outside"


def test_the_untested_pill_counts_data_holes_not_blanks_by_design():
    from fq.report import screens as sc

    row = pd.Series({"Beneish_M": float("nan"), "Accruals": -2.0, "FCF_Positive_Share": 100.0,
                     "Gross_Margin_Held": 0.5, "Incremental_ROIC": float("nan"), "EPS_Rev_90d": float("nan")})  # fmt: skip
    assert sc.untested(row) == ["Beneish M-score"]  # incremental ROIC and estimates may be blank by design


def test_boring_businesses_asks_nothing_breaking_now(monkeypatch):
    from fq.report import screens as sc

    df = pd.DataFrame({"Ticker": ["FALLING", "FINE"], "Rank": [1.0, 2.0]})
    monkeypatch.setattr(sc, "_shortlist_gates", lambda d, hurdle: [
        ("a boring industry", pd.Series(True, index=d.index)),
        (sc.NOTHING_BREAKING, d["Ticker"] == "FINE"),
    ])  # fmt: skip
    monkeypatch.setattr(sc, "in_best_fraction", lambda d: pd.Series(True, index=d.index))
    assert sc.shortlist_list(df)["Ticker"].tolist() == ["FINE"]


def test_cheapest_walks_down_value_skipping_low_grades_and_holding_the_ungraded():
    import numpy as np

    from fq.report import highlights as hl

    df = pd.DataFrame({
        "Ticker": ["CASHY", "LOW", "GOOD", "NEW", "GOOD2", "GOOD3"],
        "Value_Rank": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        "Grade_Moat": [4.0, 2.5, 3.0, np.nan, 4.0, 3.5],
        "Grade_Mgmt": [4.0, 4.0, 3.0, np.nan, 3.0, 3.5],
        "EPV_Per_Share_Yield": [6.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # CASHY: cheap on EV, not above 6% on its share price
    })  # fmt: skip
    out = hl.cheapest_list(df, 2)
    assert out["Ticker"].tolist() == ["GOOD", "NEW", "GOOD2"]  # CASHY and LOW skipped, NEW waits for its grade
    assert out["Grade_Pass"].tolist() == [True, None, True]
    assert grading.passes_cheapest(np.nan, 3.0) is None and grading.passes_cheapest(3.0, 2.75) is False
