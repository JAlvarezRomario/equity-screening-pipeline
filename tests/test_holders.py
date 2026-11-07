"""The largest holders (`fetch.top_holders`) and their card on the company page (`cards.holders_card`)."""

import json

import pandas as pd

from conftest import make_raw
from fq.fetch import parse_splits, top_holders
from fq.grading import grade_input
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.cards import holders_card
from fq.report.html import build_site
from fq.report.links import slug


def _institutions():
    return pd.DataFrame({
        "Date Reported": pd.to_datetime(["2026-06-30", "2026-06-30"]),
        "Holder": ["General Atlantic, L.P.", "Blackrock Inc."],
        "pctHeld": [0.1582, 0.0172], "Shares": [46_656_695, 5_080_957],
        "Value": [6.0e10, 6.5e9],  # pence for a London line: never read
        "pctChange": [0.0, 0.2093],
    })  # fmt: skip


def _funds():
    return pd.DataFrame({"Date Reported": pd.to_datetime(["2026-09-30"]), "Holder": ["iShares Russell 2000 ETF"],
                         "pctHeld": [0.0082], "Shares": [2_427_796], "Value": [1.0], "pctChange": [0.0274]})  # fmt: skip


def _roster():
    return pd.DataFrame({
        "Name": ["Coles (Simon)", "Coles (Simon)", "Aberforth Partners LLP", "Kanovich (Sebastian)", "Big (Basis)"],
        "Position": ["Chief Executive Officer", "Chief Executive Officer", "", "Director", ""],
        "Most Recent Transaction": ["Bought", "Bought", "Annual Return", "Sale", ""],
        "Latest Transaction Date": pd.to_datetime(["2026-10-22", "2026-10-22", "2026-06-10", "2026-08-05", "2026-01-01"]),
        "Shares Owned Directly": [1_000_000, 1_000_000, 3_000_000, 0, 500_000_000],
        "Position Direct Date": pd.to_datetime(["2026-10-22", "2026-10-22", "2026-06-10", "2026-08-05", "2026-01-01"]),
    })  # fmt: skip


def test_top_holders_merges_the_three_tables_largest_first():
    out = top_holders(_institutions(), _funds(), _roster(), shares_now=100_000_000)
    names = [h["name"] for h in out]
    assert names == ["General Atlantic, L.P.", "Aberforth Partners LLP", "Blackrock Inc.", "Coles (Simon)",
                     "iShares Russell 2000 ETF"]  # fmt: skip
    by = {h["name"]: h for h in out}
    assert by["General Atlantic, L.P."]["type"] == "Institution" and by["General Atlantic, L.P."]["pct"] == 15.82
    assert by["Blackrock Inc."]["change"] == 20.9 and "Value" not in by["Blackrock Inc."]
    assert by["iShares Russell 2000 ETF"]["type"] == "Fund"
    # A roster person with a position is an insider (once, though Yahoo repeats the row); an organisation without
    # one is a major holder; a sold-out insider and a count above every share are left out.
    assert by["Coles (Simon)"]["type"] == "Insider" and by["Coles (Simon)"]["pct"] == 1.0
    assert by["Coles (Simon)"]["latest"] == "Bought" and by["Coles (Simon)"]["latest_date"] == "2026-10-22"
    assert by["Aberforth Partners LLP"]["type"] == "Major holder"
    assert "Kanovich (Sebastian)" not in by and "Big (Basis)" not in by


def test_insider_holdings_follow_later_splits_only():
    splits = parse_splits("2026-07-01 x2; 2026-12-01 x5")
    out = top_holders(pd.DataFrame(), pd.DataFrame(), _roster().iloc[[0, 2]], shares_now=100_000_000, splits=splits)
    by = {h["name"]: h for h in out}
    assert by["Coles (Simon)"]["pct"] == 5.0  # reported after the first split: x5
    assert by["Aberforth Partners LLP"]["pct"] == 30.0  # before both: x10


def test_no_tables_no_holders():
    assert top_holders(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), shares_now=1e6) == []
    assert top_holders(None, None, _roster(), shares_now=None) == []  # no share count: no insider %


def test_card_says_never_fetched_none_and_rows():
    assert holders_card(pd.Series({"Holders": None, "Holders_At": None})) is None
    empty = holders_card(pd.Series({"Holders": "[]", "Holders_At": "2026-11-01"}))
    assert empty["rows"] == []
    full = holders_card(pd.Series({
        "Holders": json.dumps(top_holders(_institutions(), _funds(), _roster(), shares_now=100_000_000)),
        "Holders_At": "2026-11-01", "heldPercentInstitutions": 0.42, "Institutions_Count": 53.0,
    }))  # fmt: skip
    assert full["line"] == "Institutions hold 42.0% of the shares (53 institutions)"
    rows = {r["name"]: r for r in full["rows"]}
    assert rows["Blackrock Inc."]["change"] == "+20.9%" and rows["Blackrock Inc."]["cls"] == "pos"
    assert rows["Coles (Simon)"]["change"] == "Bought, 2026-10-22"
    assert rows["iShares Russell 2000 ETF"]["pct"] == "0.82%"
    assert not full["funds_only"] and not full["shared"]
    funds = holders_card(pd.Series({"Holders": json.dumps(top_holders(None, _funds(), None, shares_now=1e8)),
                                    "Holders_At": "2026-11-01"}))  # fmt: skip
    assert funds["funds_only"]
    new = [{"name": "X", "type": "Institution", "pct": 1.0, "change": 5000.0}]
    assert holders_card(pd.Series({"Holders": json.dumps(new), "Holders_At": "2026-11-01"}))["rows"][0]["change"] == (
        "New position"
    )


def test_company_page_shows_the_card_escaped_and_the_grader_never_reads_it(cfg):
    df = apply_dual_ranking(calculate_metrics(make_raw(cfg), cfg), cfg)
    holders = [{"name": "<b>Evil</b> Holdings", "type": "Institution", "pct": 12.5, "change": None,
                "position": None, "latest": None, "latest_date": None, "date": "2026-06-30"}]  # fmt: skip
    first, second = df["Ticker"].iloc[0], df["Ticker"].iloc[1]
    df["Holders"], df["Holders_At"] = None, None
    df.loc[df["Ticker"] == first, ["Holders", "Holders_At"]] = [json.dumps(holders), "2026-11-01"]
    df.loc[df["Ticker"] == second, ["Holders", "Holders_At"]] = ["[]", "2026-11-01"]
    index = build_site(df, cfg)
    pages = index.parent / "companies"
    page = (pages / f"{slug(first)}.html").read_text(encoding="utf-8")
    assert "Who owns it" in page and "&lt;b&gt;Evil&lt;/b&gt; Holdings" in page and "12.5%" in page
    row = df[df["Ticker"] == first].iloc[0]
    assert "Evil" not in grade_input(row) and "Who owns it" not in grade_input(row)
    assert "Yahoo has no holder records" in (pages / f"{slug(second)}.html").read_text(encoding="utf-8")
    third = df["Ticker"].iloc[2]
    assert "Who owns it" not in (pages / f"{slug(third)}.html").read_text(encoding="utf-8")


def test_roster_organisations_are_major_holders_and_small_insiders_stay():
    roster = pd.DataFrame({
        "Name": ["Fidelity International", "Wiles (Nicholas)", "SMITH JOHN"],
        "Position": ["", "", "Director"],
        "Shares Owned Directly": [5_000_000, 10_000, 20_000],
        "Position Direct Date": pd.to_datetime(["2026-06-10", "2026-10-22", "2026-10-22"]),
    })  # fmt: skip
    many = pd.DataFrame({"Holder": [f"Fund {i}" for i in range(12)], "pctHeld": [0.02] * 12})
    out = top_holders(many, pd.DataFrame(), roster, shares_now=100_000_000, n=10)
    by = {h["name"]: h for h in out}
    assert by["Fidelity International"]["type"] == "Major holder"
    assert by["Wiles (Nicholas)"]["type"] == "Insider" and by["SMITH JOHN"]["type"] == "Insider"
    assert len(out) == 12 and [h["type"] for h in out[-2:]] == ["Insider", "Insider"]  # beyond the ten largest
