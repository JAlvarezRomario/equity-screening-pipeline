"""The checks of the numbers (Nov 8, 2026): the earnings check and the book check, a second Gemini bar after the moat
and management grade. Synthetic data, no network."""

import json

import pandas as pd

from conftest import make_raw
from fq import forward, grading
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.highlights import Highlights, published_lists
from fq.report.html import build_site
from fq.report.links import slug
from test_grading import _graded, _grading_row


def _ranked(cfg):
    return apply_dual_ranking(calculate_metrics(make_raw(cfg), cfg), cfg)


def _answer(verdict="pass", tag=None, topics=None):
    topics = topics or [{"name": "Cycle", "status": "ok", "line": "Margins as through the last cycle"},
                        {"name": "One-offs and accounting", "status": "caution", "line": "A plant sale gain"},
                        {"name": "AI and robotics", "status": "ok", "line": "Physical service"}]  # fmt: skip
    block = {"topics": topics, "why": "- First fact.\n- **Second** fact.", "headline": "Recent years look normal.",
             "tag": tag, "verdict": verdict}  # fmt: skip
    return f"Some prose first.\n\n```json\n{json.dumps(block)}\n```\n"


def _checked(cfg, verdicts: dict[tuple[str, str], bool]):
    rows = [{"Ticker": t, "Kind": k, "Pass": v, "Headline": "Peak margins carry the record." if not v else "Fine.",
             "Tag": None if v else ("peak earnings" if k == grading.EARNINGS else "book overstated"),
             "Topics": json.dumps([{"name": "Cycle", "status": "ok" if v else "fail", "line": "A line"}]),
             "Why": "- A reason.", "Checked_At": pd.Timestamp("2026-11-08"), "Check_Hash": grading.check_hash(k)}
            for (t, k), v in verdicts.items()]  # fmt: skip
    grading.save_checks(pd.DataFrame(rows), cfg.data_dir / "checks.parquet")


def test_the_verdict_is_read_from_the_last_json_block_and_tidied():
    out = grading.parse_check(_answer(), grading.EARNINGS)
    assert out["Pass"] is True and out["Tag"] is None and out["Headline"] == "Recent years look normal."
    topics = json.loads(out["Topics"])
    assert [t["name"] for t in topics] == ["Cycle", "One-offs and accounting", "AI and robotics"]
    assert topics[1]["status"] == "caution" and out["Why"].startswith("- First fact.")
    failed = grading.parse_check(_answer("fail", "AI risk"), grading.EARNINGS)
    assert failed["Pass"] is False and failed["Tag"] == "AI risk"
    # A tag outside the vocabulary falls back to the first topic under a caution (here One-offs); a pass carries none.
    assert grading.parse_check(_answer("fail", "weird"), grading.EARNINGS)["Tag"] == "one-offs"
    assert grading.parse_check(_answer("pass", "AI risk"), grading.EARNINGS)["Tag"] is None
    # No readable verdict: None, so it is asked again and never counts as a pass.
    assert grading.parse_check("No JSON here.", grading.EARNINGS) is None
    assert grading.parse_check('```json\n{"verdict": "maybe"}\n```', grading.EARNINGS) is None


def test_a_failing_topic_or_cautions_on_cycle_and_one_offs_fail_whatever_the_model_wrote():
    # User's rule, Nov 8, 2026 (after Haidilao passed on two cautions): two flattering forces at once fail.
    two = [{"name": "Cycle", "status": "caution", "line": "Reopening spike"},
           {"name": "One-offs and accounting", "status": "caution", "line": "Capex holiday"},
           {"name": "AI and robotics", "status": "ok", "line": "Social dining"}]  # fmt: skip
    out = grading.parse_check(_answer("pass", topics=two), grading.EARNINGS)
    assert out["Pass"] is False and out["Tag"] == "peak earnings"
    one_off = [{"name": "Cycle", "status": "ok", "line": "Normal"},
               {"name": "One-offs and accounting", "status": "fail", "line": "A gain carries it"}]  # fmt: skip
    out = grading.parse_check(_answer("pass", topics=one_off), grading.EARNINGS)
    assert out["Pass"] is False and out["Tag"] == "one-offs"
    # The pill names the failing topic, not the model's tag (Haidilao: a failing cycle tagged "cash catch").
    cycle = [{"name": "Cycle", "status": "fail", "line": "Rebound and capex freeze"}]
    assert grading.parse_check(_answer("fail", "cash catch", topics=cycle), grading.EARNINGS)["Tag"] == "peak earnings"
    # Redeia: two cautions, no cash question, the model's "cash catch" refused; the first caution names it.
    two_tagged = grading.parse_check(_answer("fail", "cash catch", topics=two), grading.EARNINGS)
    assert two_tagged["Pass"] is False and two_tagged["Tag"] == "peak earnings"
    asked = [*two[:1], {"name": "One-offs and accounting", "status": "ok", "line": "x"},
             {"name": "Net cash", "status": "caution", "line": "Hoarded"}]  # fmt: skip
    assert grading.parse_check(_answer("fail", "cash catch", topics=asked), grading.EARNINGS)["Tag"] == "cash catch"
    book = [{"name": "Assets carried high", "status": "caution", "line": "x"},
            {"name": "Net assets vs book", "status": "ok", "line": "y"}]  # fmt: skip
    assert grading.parse_check(_answer("pass", topics=book), grading.BOOK)["Pass"] is True  # one caution: no fail
    assert "more than about 20% below the average of the years shown" in grading.check_prompt(
        grading.EARNINGS, "A", "A", "F", pd.Timestamp("2026-11-08")
    )


def test_accounting_notes_keep_their_terms_and_lose_only_the_stock_and_our_boilerplate():
    # Review of Nov 8, 2026: "EV" and "FCF yield" as accounting terms stay; the suspect-figures flag is the earnings
    # check's most useful note.
    kept = grading.price_free([
        "Holds client money (payroll balances): that cash is not subtracted from EV or invested capital.",
        "Data suspect (earning power more than twice the reported FCF yield (add-backs worth checking)): ranked, but "
        "left out of the top lists until checked.",
        "Distress: Altman Z'' 0.90 with thin interest cover; scored, but left out of the top lists and the mood list "
        "(cheap may mean failing).",
        "Normalised FCF yield 9.1% against a price of 12.4."])  # fmt: skip
    assert kept == ["Holds client money (payroll balances): that cash is not subtracted from EV or invested capital.",
                    "Data suspect (earning power more than twice the reported FCF yield (add-backs worth checking)).",
                    "Distress: Altman Z'' 0.90 with thin interest cover."]  # fmt: skip


def test_the_cash_question_is_asked_only_from_twenty_percent_net_cash():
    assert grading.asks_about_cash(pd.Series({"Net_Cash_to_Cap": 20.0}))
    assert not grading.asks_about_cash(pd.Series({"Net_Cash_to_Cap": 19.9}))
    assert not grading.asks_about_cash(pd.Series({"Net_Cash_to_Cap": 40.0, "Financial_Mode": True}))
    today = pd.Timestamp("2026-11-08")
    with_cash = grading.check_prompt(grading.EARNINGS, "Acme", "ACM", "FIGS", today, cash=True)
    without = grading.check_prompt(grading.EARNINGS, "Acme", "ACM", "FIGS", today, cash=False)
    assert "Net cash" in with_cash and "20% of its market value" in with_cash
    assert "Net cash" not in without and "market value" not in without
    for prompt in (with_cash, without, grading.check_prompt(grading.BOOK, "Acme", "ACM", "FIGS", today)):
        assert "{{" not in prompt and "FIGS" in prompt and "Acme (ACM)" in prompt
        # Reasoning before the verdict: the JSON asks for the topics and reasons first, the verdict last.
        block = prompt.split("```json")[1]
        assert block.index('"topics"') < block.index('"why"') < block.index('"headline"') < block.index('"verdict"')


def test_the_check_reads_the_figures_and_flags_never_a_price_or_a_verdict_of_ours():
    row = _grading_row(**{"Current Assets_Y0": 400e6, "Tangible Book Value_Y0": 250e6, "Net PPE_Y0": 300e6})
    notes = ["Cheap (Value 99, Quality 40): the market prices in a fall.",
             "Cyclical (a cyclical industry): a few years may not cover a cycle, so Value uses a mid-cycle margin; "
             "yields can still look cheap near a peak.",
             "Dilution: shares increased 6.0% in Dec-24.",
             "Peak margin: the latest operating margin is 2.1x its weaker years'."]  # fmt: skip
    md = grading.check_input(row, notes, grading.EARNINGS)
    assert "### Data notes and warning flags" in md and "Dilution: shares increased 6.0% in Dec-24." in md
    assert "Peak margin: the latest operating margin is 2.1x its weaker years'." in md
    assert "Cyclical (a cyclical industry)" in md and "balance sheet" not in md
    for leak in ("Cheap", "Value 99", "Quality", "yields", "prices", "87.65", "13.3", "Momentum", "Rank", "Score"):
        assert leak not in md, leak
    book = grading.check_input(row, notes, grading.BOOK)
    assert (
        "The balance sheet" in book and "| Tangible book value | 250 |" in book and "| Current assets | 400 |" in book
    )


def test_the_earnings_check_sees_what_sits_behind_the_clean_figures_and_the_book_check_retained_earnings():
    # Consultant review, Nov 8, 2026: one-offs still inside, a capex holiday, working capital coming back.
    row = _grading_row(**{"Net Income_Y0": 140e6, "Total Unusual Items_Y0": 25e6, "Depreciation And Amortization_Y0": 60e6,
                          "Capital Expenditure_Y0": -20e6, "Change In Working Capital_Y0": 45e6,
                          "Retained Earnings_Y0": 700e6, "Common Stock Equity_Y0": 800e6})  # fmt: skip
    md = grading.check_input(row, [], grading.EARNINGS)
    assert "### Behind the earnings" in md and "| Net income, as reported | 140 |" in md
    assert "| Unusual items, pre-tax (taken out of the clean figures) | 25 |" in md
    assert "| Change in working capital (cash out is negative) | 45 |" in md
    assert "Retained earnings" not in md
    book = grading.check_input(row, [], grading.BOOK)
    assert "| Retained earnings | 700 |" in book and "Behind the earnings" not in book
    today = pd.Timestamp("2026-11-08")
    assert "capex holiday" in grading.check_prompt(grading.EARNINGS, "A", "A", "F", today)
    cash = grading.check_prompt(grading.EARNINGS, "A", "A", "F", today, cash=True)
    assert "fail only on a real catch" in cash and "**caution**" in cash
    assert "eat the book" in grading.check_prompt(grading.BOOK, "A", "A", "F", today)


def test_due_when_new_unread_another_prompt_a_new_report_or_a_year_old():
    now = pd.Timestamp("2026-11-08")
    ranked = pd.DataFrame({"Ticker": ["A", "B", "C", "D", "E"], "PeriodEnd_Y0": ["2025-12-31"] * 4 + ["2026-06-30"]})
    h = {k: grading.check_hash(k) for k in grading.CHECK_KINDS}
    stored = pd.DataFrame({
        "Ticker": ["A", "B", "C", "E"], "Kind": ["earnings"] * 4, "Pass": [True, None, True, False],
        "Checked_At": [now, now, now - pd.Timedelta(days=400), now], "Y0_End": [pd.Timestamp("2025-12-31")] * 4,
        "Check_Hash": [h["earnings"]] * 4})  # fmt: skip
    cands = [("A", "earnings"), ("A", "book"), ("B", "earnings"), ("C", "earnings"), ("D", "earnings"),
             ("E", "earnings")]  # fmt: skip
    assert grading.checks_due(cands, ranked, stored, now) == [("A", "book"), ("B", "earnings"), ("C", "earnings"),
                                                             ("D", "earnings"), ("E", "earnings")]  # fmt: skip
    other = stored.assign(Check_Hash="old", Y0_End=pd.Timestamp("2026-06-30"))
    assert grading.checks_due([("A", "earnings")], ranked, other, now) == [("A", "earnings")]


def test_a_failed_check_shades_the_row_but_keeps_the_grade_and_a_missing_one_holds_its_place(cfg):
    df = _ranked(cfg)
    df["EPV_Per_Share_Yield"] = 10.0  # over the cheap lists' share-price bar
    _graded(cfg, df, {}, {})  # every company clears the moat and management bar
    lists = published_lists(df, cfg)
    cheap = lists.query("List == 'Cheap and good'")["Ticker"].tolist()
    assert len(cheap) >= 2 and lists["Check_Pass"].isna().all()  # nothing checked yet: nobody shaded
    low = cheap[0]
    _checked(cfg, {(low, grading.EARNINGS): False, (cheap[1], grading.EARNINGS): True})
    lists = published_lists(df, cfg)
    mine = lists[lists["Ticker"] == low]
    assert mine["Grade_Pass"].eq(True).all() and mine["Check_Pass"].eq(False).all()  # two bars, the grade untouched
    assert lists.loc[lists["Ticker"] == cheap[1], "Check_Pass"].eq(True).all()
    html = build_site(df, cfg).read_text(encoding="utf-8")
    panel = html.split('id="hl-top-1"')[1].split("</section>")[0]
    main, fold = panel.split('class="explain below-bar"')
    assert f">{low}</a>" not in main and f">{low}</a>" in fold and "✗ peak earnings" in fold
    assert "or the check of their numbers" in fold and "not checked" in main  # the others wait, in place
    assert "moat 3.0 · mgmt 3.0" in fold  # the grade pill stays


def test_balance_sheet_plays_are_judged_on_their_book_check(cfg):
    df = _ranked(cfg)
    h = Highlights.compute(df, cfg)
    on_book = h.shown("Below book")
    out = __import__("fq.report.highlights", fromlist=["with_check"]).with_check(
        on_book.assign(Check_Book_Pass=False, Check_Earnings_Pass=True), "Below book"
    )
    assert out["Check_Kind"].eq("book").all() and out["Check_Pass"].eq(False).all()
    assert grading.check_kind("Below book") == "book" and grading.check_kind("Cheap and good") == "earnings"


def test_the_check_pool_holds_the_listed_past_their_bar_and_every_holding(cfg, monkeypatch):
    from fq import pipeline

    df = _ranked(cfg)
    df["EPV_Per_Share_Yield"] = 10.0
    held = "HELD"
    df = pd.concat([df, df.iloc[[0]].assign(Ticker=held)], ignore_index=True)
    _graded(cfg, df, {df["Ticker"].iloc[1]: 1.0}, {df["Ticker"].iloc[1]: 1.0})
    seen = {}
    monkeypatch.setattr(pipeline, "portfolio", lambda: frozenset({held, "NOT.RANKED"}))
    monkeypatch.setattr(grading, "checks_due", lambda candidates, *a: seen.setdefault("c", candidates) and [])
    h = Highlights.compute(df, cfg)
    pipeline.check_companies(cfg, df, h, dry_run=True)
    pub = h.published()
    passed = set(pub.loc[pub["Grade_Pass"].eq(True), "Ticker"])
    assert {t for t, _ in seen["c"]} == passed | {held}
    assert all(k in grading.CHECK_KINDS for _, k in seen["c"]) and (held, grading.EARNINGS) in seen["c"]


def test_the_check_card_sits_before_what_a_share_earns(cfg):
    df = _ranked(cfg)
    _graded(cfg, df, {}, {})
    df["EPV_Per_Share_Yield"] = 10.0  # over the cheap lists' share-price bar, so some companies are listed
    df.loc[df.index[-2:], "Traded_Value_USD"] = 1.0  # two too illiquid for any list (`highlights.listable`)
    listed = list(dict.fromkeys(published_lists(df, cfg)["Ticker"]))
    unlisted, other = [t for t in df["Ticker"] if t not in set(listed)][:2]
    ticker = listed[0]
    _checked(cfg, {(ticker, grading.EARNINGS): False, (unlisted, grading.EARNINGS): True})
    index = build_site(df, cfg)
    page = lambda t: (index.parent / "companies" / f"{slug(t)}.html").read_text(encoding="utf-8")  # noqa: E731
    mine = page(ticker)
    assert mine.index("The numbers, checked") < mine.index("What a share earns")
    assert "✗ Peak earnings" in mine and "Peak margins carry the record." in mine and "A reason." in mine
    assert "check-low" in mine and 'aria-label="fails"' in mine and "without the price" in mine
    assert "check-card" not in page(other)  # no check made, no card
    assert "check-card" not in page(unlisted)  # checked once, off the lists now: kept, not shown (as the grade card)


def test_the_forward_test_freezes_the_checks_with_the_grades_and_gates_failed_members():
    cohort = pd.DataFrame({"List": ["Cheap and good", "Below book", "Below book", forward.PORTFOLIO_LIST],
                           "Ticker": ["A", "A", "B", "B"]})  # fmt: skip
    gate = pd.DataFrame({"Ticker": ["A", "A", "B"], "Kind": ["earnings", "book", "book"], "Pass": [False, True, False],
                         "Checked_At": [pd.Timestamp("2026-11-30")] * 3})  # fmt: skip
    out = forward.apply_check_gate(cohort, gate)
    # A fails its earnings check, not its book one: out of Cheap and good, kept on Below book; B kept in the portfolio.
    assert out[["List", "Ticker"]].values.tolist() == [["Below book", "A"], [forward.PORTFOLIO_LIST, "B"]]
    assert out.attrs["checked_out"] == 2


def test_the_list_holds_the_core_lists_members_past_both_bars_by_share_price_yield(cfg):
    # User's choice, Nov 9, 2026: the companies that passed every test, shown first.
    df = _ranked(cfg)
    df["EPV_Per_Share_Yield"] = [7.0 + i for i in range(len(df))]  # over the cheap lists' bar, a distinct order
    _graded(cfg, df, {}, {})
    h = Highlights.compute(df, cfg)
    cheap = h.shown("Cheap and good")["Ticker"].tolist()
    assert len(cheap) >= 3 and h.shown("The List").empty  # nothing checked yet: nobody in
    passed, failed = cheap[:2], cheap[2]
    _checked(cfg, {**{(t, grading.EARNINGS): True for t in passed}, (failed, grading.EARNINGS): False})
    h = Highlights.compute(df, cfg)
    the_list = h.shown("The List")
    assert set(the_list["Ticker"]) >= set(passed) and failed not in set(the_list["Ticker"])
    yields = the_list["EPV_Per_Share_Yield"].tolist()
    assert yields == sorted(yields, reverse=True)  # owner yield on the share price, highest first
    assert the_list["The_List_From"].str.contains("Cheap and good").all() or len(set(the_list["Ticker"]) - set(cheap))
    assert not any(t == "The List" for t, _ in [(n, k) for n, k in h.check_pool()])  # asks no check of its own
    assert grading.check_kind("The List") is None
    html = build_site(df, cfg).read_text(encoding="utf-8")
    tabs = html.split('role="tablist"')[1].split('<section class="tab-panel"')[0]  # every row of tabs
    assert tabs.index('data-panel="hl-the-list" class="tab-main"') < tabs.index('data-panel="hl-top-1"')
    panel = html.split('id="hl-the-list"')[1].split("</section>")[0]
    assert "pill-src" in panel and "passed every test" in panel


def test_an_ai_caution_fails_only_in_an_ai_fragile_industry():
    # User's rule, Nov 9, 2026 (Artner, staffing, passed on an AI caution): there the company must earn an "ok".
    topics = [{"name": "Cycle", "status": "ok", "line": "Structural shortage"},
              {"name": "One-offs and accounting", "status": "caution", "line": "A small acquisition"},
              {"name": "AI and robotics", "status": "caution", "line": "AI coding threatens billable hours"}]  # fmt: skip
    assert grading.parse_check(_answer("pass", topics=topics), grading.EARNINGS)["Pass"] is True
    out = grading.parse_check(_answer("pass", topics=topics), grading.EARNINGS, fragile=True)
    assert out["Pass"] is False and out["Tag"] == "AI risk"
    safe = [*topics[:2], {"name": "AI and robotics", "status": "ok", "line": "Physical work"}]
    assert grading.parse_check(_answer("pass", topics=safe), grading.EARNINGS, fragile=True)["Pass"] is True
    assert grading.ai_fragile(pd.Series({"industry": "Staffing & Employment Services"}))
    assert not grading.ai_fragile(pd.Series({"industry": "Tools & Accessories"}))
