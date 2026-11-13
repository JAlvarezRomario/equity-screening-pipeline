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


def _book_answer(verdict="pass", topics=None, base=120.0, low=90.0, high=150.0, bridge=None):
    topics = topics or [{"name": "Assets carried high", "status": "ok", "line": "Receivables are blue-chip"},
                        {"name": "Assets carried low", "status": "none", "line": "No old land"},
                        {"name": "Net assets vs book", "status": "ok", "line": "Covers the book"}]  # fmt: skip
    bridge = bridge if bridge is not None else [{"item": "Land at market", "amount": 30, "note": "Bought in 1960"},
                                                {"item": "Old stock", "amount": -10, "note": "Slow movers"}]  # fmt: skip
    block = {"topics": topics, "bridge": bridge, "nav": {"low": low, "base": base, "high": high},
             "confidence": "medium", "why": "- A fact.", "headline": "The book holds.", "tag": None,
             "verdict": verdict}  # fmt: skip
    return f"Prose.\n\n```json\n{json.dumps(block)}\n```\n"


def test_the_book_holds_when_the_base_net_asset_estimate_covers_the_book_whatever_the_model_wrote():
    # User's rule, Nov 12, 2026: base NAV >= book decides; a high-asset fail alone no longer fails, the bridge sizes it.
    base = {"Book": 100.0, "Tangible_Book": 80.0}
    out = grading.parse_check(_book_answer("fail", base=110), grading.BOOK, baseline=base)
    assert out["Pass"] is True and out["NAV_Base"] == 110 and out["Book"] == 100 and out["Confidence"] == "medium"
    assert json.loads(out["Bridge"])[0] == {"item": "Land at market", "amount": 30, "note": "Bought in 1960"}
    high_fail = [{"name": "Assets carried high", "status": "fail", "line": "Stock carried at cost"}]
    assert grading.parse_check(_book_answer(topics=high_fail, base=105), grading.BOOK, baseline=base)["Pass"] is True
    below = grading.parse_check(_book_answer("pass", base=89), grading.BOOK, baseline=base)
    assert below["Pass"] is False and below["Tag"] == "book overstated"
    # Within `BOOK_TOLERANCE` (user's choice: 5% on Nov 12, 2026, 10% on Nov 13 after the full re-check).
    assert grading.parse_check(_book_answer("fail", base=91), grading.BOOK, baseline=base)["Pass"] is True
    assert "at least about 90% of the book" in grading.check_prompt(
        grading.BOOK, "A", "A", "F", pd.Timestamp("2026-11-12")
    )
    # A new tolerance re-reads the stored answers, no model call (`recheck_book_rule`).
    stored = pd.DataFrame([{"Ticker": "A", "Kind": "book", "Pass": False, "Tag": "book overstated", "NAV_Base": 92.0,
                            "Book": 100.0, "Tangible_Book": 80.0, "Report": _book_answer("pass", base=92)},
                           {"Ticker": "B", "Kind": "earnings", "Pass": True, "Tag": None, "NAV_Base": None,
                            "Book": None, "Tangible_Book": None, "Report": "untouched"}])  # fmt: skip
    again = grading.recheck_book_rule(stored)
    assert bool(again.loc[0, "Pass"]) and pd.isna(again.loc[0, "Tag"]) and again.loc[1, "Report"] == "untouched"
    locked = [{"name": "Reaches shareholders", "status": "fail", "line": "Parent borrows the cash"}]
    out = grading.parse_check(_book_answer("pass", topics=locked, base=200), grading.BOOK, baseline=base)
    assert out["Pass"] is False and out["Tag"] == "value locked"
    eats = [{"name": "Eats the book", "status": "fail", "line": "Losses run on"}]
    assert grading.parse_check(_book_answer(topics=eats), grading.BOOK, baseline=base)["Tag"] == "book eroding"
    # No base estimate: not made, asked again (never a pass); "none" is a status of its own.
    assert grading.parse_check(_book_answer(base="n/a"), grading.BOOK, baseline=base) is None
    topics = json.loads(grading.parse_check(_book_answer(), grading.BOOK, baseline=base)["Topics"])
    assert topics[1] == {"name": "Assets carried low", "status": "none", "line": "No old land"}
    # Figures written as text still read; a low above the base is held to it.
    assert grading.parse_check(_book_answer(base="$1,250m", low=1300), grading.BOOK, baseline=base)["NAV_Low"] == 1250


def test_the_earnings_check_fails_when_its_normal_year_is_well_below_the_years_shown():
    def answer(margin, oe):
        block = json.loads(_answer().split("```json\n")[1].split("\n```")[0])
        block["normal"] = {"operating_margin_pct": margin, "owner_earnings": oe}
        return f"```json\n{json.dumps(block)}\n```"

    base = {"Avg_Margin": 14.0, "Avg_Owner_Earnings": 13.0}
    fine = grading.parse_check(answer(13.0, 12.0), grading.EARNINGS, baseline=base)
    assert fine["Pass"] is True and fine["Normal_Margin"] == 13.0 and fine["Avg_Owner_Earnings"] == 13.0
    peak = grading.parse_check(answer(10.0, 12.0), grading.EARNINGS, baseline=base)  # 10 < 0.8 x 14
    assert peak["Pass"] is False and peak["Tag"] == "peak earnings"
    assert grading.parse_check(answer(14.0, 10.0), grading.EARNINGS, baseline=base)["Pass"] is False
    assert grading.parse_check(_answer(), grading.EARNINGS, baseline=base)["Pass"] is True  # no normal year: unmeasured
    row = _grading_row()
    got = grading.check_baseline(row, grading.EARNINGS)
    assert set(got) == {"Avg_Margin", "Avg_Owner_Earnings"}


def test_the_cash_question_sees_what_the_owners_were_paid():
    # Ishihara (Nov 12, 2026): a clean pile being returned had a caution; the payout is now in front of the model.
    row = _grading_row(**{"FCF_Reported_Y0": 20e6, "FCF_Reported_Y1": 20e6, "Cash Dividends Paid_Y0": -5e6,
                          "Cash Dividends Paid_Y1": -5e6, "Repurchase Of Capital Stock_Y0": -10e6})  # fmt: skip
    line = grading.payout_line(row, __import__("fq.settings", fromlist=["Settings"]).Settings())
    assert line.startswith("Dividends plus buybacks took 50% of free cash flow") and "(20 of 40 million)" in line
    with_cash = grading.check_input(row, [], grading.EARNINGS, cash=True)
    assert "### What the owners were paid" in with_cash
    assert "What the owners were paid" not in grading.check_input(row, [], grading.EARNINGS)
    prompt = grading.check_prompt(grading.EARNINGS, "A", "A", "F", pd.Timestamp("2026-11-12"), cash=True)
    assert "A large pile that is being returned is ok" in prompt


def test_the_book_card_shows_the_estimate_against_both_books_the_bridge_and_the_price_multiple():
    from fq.report import cards

    row = pd.Series({"Check_Book_NAV_Low": 90.0, "Check_Book_NAV_Base": 110.0, "Check_Book_NAV_High": 140.0,
                     "Check_Book_Book": 100.0, "Check_Book_Tangible_Book": 80.0, "Check_Book_Confidence": "low",
                     "Check_Book_Bridge": json.dumps([{"item": "Land", "amount": 30, "note": "Old cost"},
                                                      {"item": "Stock", "amount": -20, "note": "Slow"}]),
                     "BV_to_Price": 200.0})  # fmt: skip # price is half the book
    nav = cards.nav_figures(row)
    # User's choice, Nov 12, 2026: low = a fire sale, base = a going concern (the verdict), high = a buyer.
    assert nav["book"] == "0.90x if closed, 1.10x as a going concern, 1.40x to a buyer (book)"
    assert nav["tangible"] == "1.12x if closed, 1.38x as a going concern, 1.75x to a buyer (tangible book)"
    assert [b["amount"] for b in nav["bridge"]] == ["100", "+30", "-20", "110"]
    assert nav["price"].startswith("At today's price: about 0.45x the going-concern estimate and 0.56x the fire-sale")
    closed = row.copy()
    closed["Check_Book_NAV_Low"] = -5.0
    assert "more than the fire-sale value, which is nothing" in cards.nav_figures(closed)["price"]
    assert cards.nav_figures(pd.Series({"Check_Book_NAV_Base": None})) is None  # a check made before: no figures
    earn = pd.Series({"Check_Earnings_Normal_Margin": 14.0, "Check_Earnings_Avg_Margin": 14.2,
                      "Check_Earnings_Normal_Owner_Earnings": 13.0, "Check_Earnings_Avg_Owner_Earnings": 14.0})  # fmt: skip
    assert cards.normal_year(earn) == ("A normal year: about 14.0% operating margin against 14.2% and $13M owner "
                                       "earnings against $14M on average in the years shown.")  # fmt: skip
    assert cards.STATUS_MARKS["none"] == ("–", "nothing found")


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
    answer = _book_answer("pass", book, base=120)
    assert grading.parse_check(answer, grading.BOOK, baseline={"Book": 100})["Pass"] is True  # one caution: no fail
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
    # An alias hash stays current (the book checks re-read under the 10% rule, Nov 13, 2026).
    alias = stored.assign(Check_Hash=next(iter(grading.CHECK_HASH_ALIASES)))
    assert grading.checks_due([("A", "earnings")], ranked, alias, now) == []


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
    good = {t: 3.5 for t in df["Ticker"]}  # over The List's floor of 3.25
    _graded(cfg, df, good, good)
    h = Highlights.compute(df, cfg)
    cheap = h.shown("Cheap and good")["Ticker"].tolist()
    assert len(cheap) >= 3 and h.shown("The List").empty  # nothing checked yet: nobody in
    passed, failed = cheap[:2], cheap[2]
    _checked(cfg, {**{(t, grading.EARNINGS): True for t in passed}, (failed, grading.EARNINGS): False})
    h = Highlights.compute(df, cfg)
    the_list = h.shown("The List")
    assert set(the_list["Ticker"]) >= set(passed) and failed not in set(the_list["Ticker"])
    assert the_list["The_List_Order"].tolist() == list(range(1, len(the_list) + 1))
    assert the_list["The_List_From"].str.contains("Cheap and good").all() or len(set(the_list["Ticker"]) - set(cheap))
    assert not any(t == "The List" for t, _ in [(n, k) for n, k in h.check_pool()])  # asks no check of its own
    assert grading.check_kind("The List") is None
    html = build_site(df, cfg).read_text(encoding="utf-8")
    tabs = html.split('role="tablist"')[1].split('<section class="tab-panel"')[0]  # every row of tabs
    assert tabs.index('data-panel="hl-the-list" class="tab-main"') < tabs.index('data-panel="hl-top-1"')
    panel = html.split('id="hl-the-list"')[1].split("</section>")[0]
    assert "pill-src" in panel and "passed every test" in panel


def test_the_list_leaves_out_banks_insurers_and_reits(monkeypatch):
    # User's choice, Nov 9, 2026: no banks, insurers or REITs; fee businesses (Visa, Moody's) stay.
    import fq.report.highlights as hl

    monkeypatch.setattr(hl, "_screened", lambda tab: lambda rows, n, hurdle: rows.assign(Grade_Pass=True,
                                                                                         Check_Pass=True,
                                                                                         Grade_Moat=3.5,
                                                                                         Grade_Mgmt=3.5))  # fmt: skip
    df = pd.DataFrame({"Ticker": ["OP", "BK", "IN", "RE", "V"], "EPV_Per_Share_Yield": [7.0, 9.0, 8.0, 7.5, 6.5],
                       "Business_Model": ["operating", "bank", "insurer", "operating", "client_cash"],
                       "Rank_Group": [None, "financial", "financial", "real_estate", None]})  # fmt: skip
    assert hl.the_list(df, 200)["Ticker"].tolist() == ["OP", "V"]


def _sources(monkeypatch, frames):
    """`highlights._screened` returning each source tab's given rows (`frames`: name -> frame), none for the rest."""
    import fq.report.highlights as hl

    def screened(tab):
        rows = frames.get(tab.name)
        return lambda df, n, hurdle: df.iloc[0:0] if rows is None else rows

    monkeypatch.setattr(hl, "_screened", screened)
    return hl


def _row(ticker, yield_, moat=3.5, mgmt=3.5, grade=True, check=True, **more):
    return {"Ticker": ticker, "EPV_Per_Share_Yield": yield_, "Grade_Moat": moat, "Grade_Mgmt": mgmt,
            "Grade_Pass": grade, "Check_Pass": check, "Business_Model": "operating", "Rank_Group": "ranked", **more}  # fmt: skip


def test_the_list_holds_one_grade_floor_and_a_failed_check_anywhere_keeps_a_company_out(monkeypatch):
    # Nov 9, 2026 review (user's approval): "either at least 4.25" lets weak management in on Income; a check failed on
    # one source is not hidden by a pass on another.
    comp = pd.DataFrame([_row("WEAK", 9.0, moat=4.5, mgmt=1.5), _row("OK", 8.0), _row("MIXED", 7.0)])
    book = pd.DataFrame([_row("MIXED", 7.0, check=False, Check_Earnings_Pass=True)])
    hl = _sources(monkeypatch, {"Compounders": comp, "Below book": book})
    assert hl.the_list(pd.DataFrame(), 20)["Ticker"].tolist() == ["OK"]


def test_the_list_asks_a_balance_sheet_play_alone_for_its_earnings_check_and_leaves_out_holdings(monkeypatch):
    book = pd.DataFrame([_row("WAIT", 9.0), _row("EARN", 8.0, Check_Earnings_Pass=True),
                         _row("HOLD", 7.0, Check_Earnings_Pass=True, Rank_Group="holding"),
                         _row("BOTH", 6.0)])  # fmt: skip
    cheap = pd.DataFrame([_row("BOTH", 6.0)])  # its earnings were checked on Cheap and good
    hl = _sources(monkeypatch, {"Below book": book, "Cheap and good": cheap})
    out = hl.the_list(pd.DataFrame(), 20)
    assert out["Ticker"].tolist() == ["EARN", "BOTH"]
    assert out.set_index("Ticker").loc["BOTH", "The_List_From"] == "Cheap and good · Balance sheet plays"


def test_the_list_takes_each_lists_best_in_turn_the_share_price_yield_breaking_ties(monkeypatch):
    # User's choice, Nov 9, 2026: each idea gets its best few; one yield for all buried the compounders.
    cheap = pd.DataFrame([_row("C1", 20.0, Rank=1), _row("C2", 18.0, Rank=2), _row("C3", 16.0, Rank=3)])
    comp = pd.DataFrame([_row("K2", 7.0, Rank=9), _row("K1", 8.0, Rank=5)])  # out of order: its own order decides
    book = pd.DataFrame([_row("FAIL", 30.0, check=False), _row("I1", 9.0, Check_Earnings_Pass=True)])
    hl = _sources(monkeypatch, {"Cheap and good": cheap, "Compounders": comp, "Below book": book})
    out = hl.the_list(pd.DataFrame(), 20)
    # Turn 1: C1, I1 (Balance sheet plays' first that qualifies), K1; turn 2: C2, K2; turn 3: C3.
    assert out["Ticker"].tolist() == ["C1", "I1", "K1", "C2", "K2", "C3"]
    assert out["The_List_Order"].tolist() == [1, 2, 3, 4, 5, 6]


def test_the_list_shows_the_best_twenty_five_from_its_five_sources_with_both_grades_above_3(monkeypatch):
    # Nov 11, 2026 (user's choice): Boring businesses and Income feed it again; the raw scores still do not.
    rows = pd.DataFrame([_row(f"T{i}", 40.0 - i) for i in range(30)])
    rows = pd.concat([rows, pd.DataFrame([_row("LOW", 99.0, moat=3.25, mgmt=3.0)])], ignore_index=True)
    boring = pd.DataFrame([_row("BORING", 98.0)])
    raw = pd.DataFrame([_row("RAW", 97.0)])
    hl = _sources(monkeypatch, {"Compounders": rows, "Shortlist": boring, "Income": boring, "Cheapest": raw})
    assert hl.THE_LIST_N == 25
    assert hl.THE_LIST_SOURCES == ("Cheap and good", "Shortlist", "Compounders", "Income", "Below book")
    out = hl.the_list(pd.DataFrame(), hl.THE_LIST_N)
    assert len(out) == 25 and out["Ticker"].iloc[0] == "BORING"  # each list's first, the higher yield first
    assert out.set_index("Ticker").loc["BORING", "The_List_From"] == "Boring businesses · Income"
    assert out["The_List_From"].iloc[1:].eq("Compounders").all()
    assert not {"LOW", "RAW"} & set(out["Ticker"])


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


def test_the_score_lists_walk_down_until_ten_pass_every_test(monkeypatch):
    # User's rule, Nov 9, 2026: at least 10 of Cheap and good, Cheapest and Best businesses pass grade and check.
    from fq.report import highlights as hl

    n = 30
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "Rank": range(1, n + 1),
                       "Value_Rank": range(1, n + 1), "EPV_Per_Share_Yield": 9.0,
                       "Grade_Moat": 3.5, "Grade_Mgmt": 3.5,
                       "Check_Earnings_Pass": [False] * 15 + [True] * 15})  # fmt: skip
    monkeypatch.setattr(hl, "top_n", lambda d, col, k: d.sort_values(col).head(k))
    cheap = hl.cheap_and_good_list(df, 20)
    assert len(cheap) == 25 and cheap["Check_Earnings_Pass"].sum() == 10  # 15 fail first: 5 more down, 10 pass
    walk = hl.cheapest_list(df, 20)
    assert len(walk) == 25 and walk["Check_Earnings_Pass"].sum() == 10 and walk["Grade_Pass"].eq(True).all()
    unchecked = df.assign(Check_Earnings_Pass=[False] * 15 + [None] * 15)
    assert len(hl.cheap_and_good_list(unchecked, 20)) == 25  # not checked yet counts as a pass: the walk waits
    none_pass = df.assign(Check_Earnings_Pass=False)
    assert len(hl.cheap_and_good_list(none_pass, 20)) == n  # all the way down, never past the last row
    assert len(hl.cheap_and_good_list(pd.concat([df.assign(Check_Earnings_Pass=False)] * 3, ignore_index=True)
                                      .assign(Rank=range(1, 3 * n + 1)), 20)) == hl.WALK_MAX_ROWS  # fmt: skip


def test_the_list_leaves_out_cash_hoards_under_weak_management(monkeypatch):
    # User's choice, Nov 9, 2026: net cash of at least 25% of the market cap needs management graded at least 4, in
    # any country.
    rows = pd.DataFrame([_row("OK", 9.0), _row("HOARD", 8.0, Net_Cash_to_Cap=40.0),
                         _row("HOARD4", 7.0, mgmt=4.0, Net_Cash_to_Cap=40.0),
                         _row("SOME", 6.5, Net_Cash_to_Cap=10.0)])  # fmt: skip
    hl = _sources(monkeypatch, {"Compounders": rows})
    assert hl.the_list(pd.DataFrame(), 20)["Ticker"].tolist() == ["OK", "HOARD4", "SOME"]


def test_lists_keep_fallen_shares_but_need_a_record(cfg, monkeypatch):
    # User's choices, Nov 10, 2026: the 6-month price floor went (a fallen share stays, with a pill); a list needs
    # 4 annual returns and 5 years of share counts (a blank history passes), unless the ticker is confirmed.
    from fq.report import screens as sc
    from fq.report.highlights import listable
    from fq.report.pills import short_record as pills_short
    from fq.report.pills import tags

    old = '{"0": 10.0, "5": 10.0, "10": 10.0}'  # year numbers, not calendar years: only the span is read
    new = '{"8": 10.0, "10": 10.0}'
    roic = {f"ROIC_Y{i}": [15.0, 15.0, 15.0, 15.0, 15.0] for i in range(4)}
    roic["ROIC_Y2"] = [15.0, None, 15.0, 15.0, 15.0]  # SHORT has 3 of 4
    df = pd.DataFrame({"Ticker": ["FELL", "SHORT", "NEW", "BLANK", "THIN"], "Momentum_6M": [-25.0, 1.0, 1.0, None, 5.0],
                       "Shares_Hist": [old, old, new, None, old], "Traded_Value_USD": [1e9, 1e9, 1e9, 1e9, 1.0],
                       **roic})  # fmt: skip
    assert listable(df, cfg)["Ticker"].tolist() == ["FELL", "BLANK"]
    words = {t["word"] for t in tags(df.iloc[0])}
    assert "down 25% in 6m" in words and "short record" not in words
    assert "short record" in {t["word"] for t in tags(df.iloc[2])}
    stored = df.assign(Record_Years=sc.roic_years(df), Listed_Years=sc.listed_years(df))  # as `highlights.prepare`
    stored = stored.assign(Has_Record=sc.has_record(stored))
    assert listable(stored, cfg)["Ticker"].tolist() == ["FELL", "BLANK"]
    for i in range(len(df)):
        assert pills_short(df.iloc[i]) == pills_short(stored.iloc[i])
    paid = df.assign(DPS_Hist=[None, None, old, None, None])  # dividends since long before the share counts start
    assert listable(paid, cfg)["Ticker"].tolist() == ["FELL", "NEW", "BLANK"]
    monkeypatch.setattr(sc, "_record_confirmed", lambda: frozenset({"NEW"}))  # an old business under a new ticker
    assert listable(df, cfg)["Ticker"].tolist() == ["FELL", "NEW", "BLANK"]


def test_the_list_names_each_list_and_the_place_that_earned_the_seat(monkeypatch):
    cheap = pd.DataFrame([_row("A", 9.0, Rank=1), _row("B", 8.0, Rank=2)])
    book = pd.DataFrame([_row("B", 8.0, Check_Earnings_Pass=True)])
    hl = _sources(monkeypatch, {"Cheap and good": cheap, "Below book": book})
    why = hl.the_list(pd.DataFrame(), 20).set_index("Ticker")["The_List_Why"]
    assert why["A"] == "Cheap and good #1" and why["B"] == "Cheap and good #2 · Balance sheet plays #1"


def test_the_list_asks_every_member_the_bond_plus_country_risk_grades_above_3_no_ai_risk_nothing_breaking(monkeypatch):
    # User's choices, Nov 10, 2026: the raw-score lists no longer feed it; every member, whatever its source, earns at
    # least the 10-year Treasury plus its country's sovereign spread on the share price (no 4% seat for Compounders),
    # has both grades above 3, is past the AI screen, and nothing is breaking now; a fallen share is no reason.
    cheap = pd.DataFrame([_row("OK", 9.0), _row("THIN", 4.4), _row("BLANK", None), _row("THREE", 8.0, mgmt=3.0),
                          _row("SOFT", 8.5, industry="Software - Application"), _row("FELL", 8.2, Momentum_6M=-25.0),
                          _row("RISKY", 5.0, Sovereign_Spread=1.0), _row("BREAKS", 7.5, EPS_Rev_90d=-20.0),
                          _row("SALES", 7.2, Revenue_Change_Recent=-12.0)]).assign(Treasury_Yield=4.5)  # fmt: skip
    comp = pd.DataFrame([_row("GROWS", 4.2), _row("BOND", 4.8)]).assign(Treasury_Yield=4.5)
    raw = pd.DataFrame([_row("QUALITY", 12.0), _row("VALUE", 11.0)])
    sources = {"Cheap and good": cheap, "Compounders": comp, "Best businesses": raw, "Cheapest": raw}
    hl = _sources(monkeypatch, sources)
    assert hl.the_list(pd.DataFrame(), 20)["Ticker"].tolist() == ["OK", "BOND", "FELL"]
    hl = _sources(monkeypatch, {"Cheap and good": cheap.assign(Treasury_Yield=7.5)})
    assert hl.the_list(pd.DataFrame(), 20)["Ticker"].tolist() == ["OK", "FELL"]  # the bond moves the floor


def test_the_lists_mix_sets_its_shares_against_the_markets():
    from fq.report.html import list_mix

    countries = ["Japan"] * 4 + ["United States"] * 3 + ["Hong Kong"] * 2 + ["Greece"]
    currency = {"Japan": "JPY", "United States": "USD", "Hong Kong": "HKD", "Greece": "EUR"}
    rows = pd.DataFrame({"country": countries, "currency": [currency[c] for c in countries],
                         "Market Cap": [0.5] * 4 + [50.0] * 3 + [1.0] * 2 + [0.2]})  # fmt: skip
    market = pd.DataFrame({"country": ["United States", "Japan"], "currency": ["USD", "JPY"],
                           "Market Cap": [90.0, 10.0]})  # fmt: skip
    mix = {m["label"]: m["text"] for m in list_mix(rows, market)}
    assert mix["Countries"] == ("Japan 40% (market 10%); United States 30% (market 90%); Hong Kong 20% (market 0%); "
                                "other 10% (market 0%)")  # fmt: skip
    assert mix["Size"].startswith("Small 60%") and mix["Currencies"].startswith("JPY 40% (market 10%)")


def test_a_list_fills_past_its_failures_and_keeps_them_folded():
    # User's choice, Nov 9, 2026: one rule for every list (Insurers showed 17 shaded of 20 while 16 more qualified).
    import fq.report.highlights as hl

    verdicts = [False, True, None, False, True, True, True]
    out = pd.DataFrame({"Ticker": list("ABCDEFG"), "Grade_Pass": verdicts, "Check_Pass": [True] * 7})
    assert hl._fill(out, 3)["Ticker"].tolist() == ["A", "B", "C", "D", "E"]  # 3 that do not fail, the 2 fails kept
    assert hl._fill(out, 10)["Ticker"].tolist() == list("ABCDEFG")  # fewer qualify: all shown
    many = pd.DataFrame({"Ticker": [f"T{i}" for i in range(100)], "Grade_Pass": False, "Check_Pass": True})
    assert len(hl._fill(many, 20)) == hl.WALK_MAX_ROWS and len(hl._fill(many, 100)) == 100  # the count is uncapped


def test_walks_count_their_eligible_and_the_retired_lists_are_gone():
    # User's choices: a walk's count is its pool (Nov 9, 2026); Temporary trouble and Low beta retired before the
    # forward test's freeze (Nov 11, 2026).
    import fq.report.highlights as hl

    assert all(t.eligible is not None for t in hl.TABS if t.ranked)
    assert not {"Temporary trouble", "Low beta"} & set(hl.TAB_ORDER)
    assert len(hl.TABS) == 13


def _deep(ticker, bv=250.0, nav=225.0, model="operating", book=True, mgmt=3.5, **more):
    return {"Ticker": ticker, "BV_to_Price": bv, "NAV_to_Price": nav, "Effective_NAV_to_Price": nav,
            "Business_Model": model, "Rank_Group": "ranked", "Check_Book_Pass": book, "Grade_Moat": 2.0,
            "Grade_Mgmt": mgmt, "industry": "Conglomerates", **more}  # fmt: skip


def test_deep_discount_is_under_two_thirds_of_both_books_and_never_a_lender():
    # User's rule, Nov 11, 2026 (Swire Pacific B at 0.40x book and 0.44x tangible book had no book check).
    from fq.report import screens as sc

    df = pd.DataFrame([_deep("BOTH"), _deep("TANGIBLE", nav=110.0), _deep("BANK", model="bank"),
                       _deep("EDGE", bv=152.0, nav=152.0)])  # fmt: skip
    assert sc.deep_discount(df).tolist() == [True, False, False, True]


def test_deep_discount_list_walks_the_discount_on_management_with_the_book_holding():
    # User's choices, Nov 11, 2026: a list of its own, largest discount first, management at least 3.5, moat not asked.
    import fq.report.highlights as hl

    df = pd.DataFrame([_deep("OK", nav=200.0), _deep("DEEPEST", nav=400.0), _deep("WEAK", nav=300.0, mgmt=3.25),
                       _deep("WAIT", nav=250.0, mgmt=float("nan")), _deep("FAILED", nav=500.0, book=False),
                       _deep("UNCHECKED", nav=450.0, book=None), _deep("REIT", nav=600.0, Rank_Group="real_estate")])  # fmt: skip
    out = hl.deep_discount_walk(df, 20)
    assert out["Ticker"].tolist() == ["DEEPEST", "WAIT", "OK"]
    assert out["Grade_Pass"].tolist() == [True, None, True]
    assert len(hl._eligible_deep(df)) == 4  # the book checked and holding: DEEPEST, WEAK, WAIT, OK
    assert hl.TAB["Deep discount"].ranked and hl.label("Deep discount") == "Deep discount"


def test_every_deep_discount_gets_a_book_check_its_card_and_a_pill_when_it_holds(monkeypatch):
    import fq.report.highlights as hl
    from fq.report.cards import check_cards
    from fq.report.pills import tags

    frame = pd.DataFrame([_deep("SWIRE", book=None), _deep("PLAIN", bv=120.0, nav=110.0, book=None)])
    frame = frame.assign(Deep_Discount=[True, False], Dividend_WHT=0.0)
    h = hl.Highlights(frame, frame.iloc[0:0], None, 6.0, {})
    monkeypatch.setattr(h, "published", lambda: pd.DataFrame(columns=["Ticker", "List", "Grade_Pass", "Check_Pass"]))
    assert ("SWIRE", "book") in h.check_pool() and ("PLAIN", "book") not in h.check_pool()
    row = frame.iloc[0].copy()
    row["Grade_Listed"], row["Check_Book_Pass"], row["Check_Book_Topics"] = False, True, "[]"
    assert [c["kind"] for c in check_cards(row)] == ["book"]
    assert "deep discount to book" in {t["word"] for t in tags(row)}
    row["Check_Book_Pass"] = False
    assert "deep discount to book" not in {t["word"] for t in tags(row)}
