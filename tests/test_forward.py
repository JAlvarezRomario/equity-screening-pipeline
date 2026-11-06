"""Forward test of the frozen lists (no network: synthetic prices and exchange rates)."""

import numpy as np
import pandas as pd
import pytest

from fq import forward
from fq.report.forward_page import build_forward, build_placeholder

START = pd.Timestamp("2024-01-01")


def _prices(days, columns, growth):
    """Daily closes rising by `growth[c]` per business day."""
    steps = np.arange(len(days))
    return pd.DataFrame({c: 100 * (1 + growth[c]) ** steps for c in columns}, index=days)


def _cohort():
    published = pd.DataFrame({"List": ["Cheap", "Cheap", "Good"], "Ticker": ["A", "B", "C"]})
    ranked = pd.DataFrame({"Ticker": ["A", "B", "C", "D"], "Rank": [2, 1, 3, np.nan], "shortName": list("abcd"),
                           "currency": ["USD", "GBp", "USD", "USD"], "Value_Score": 50.0, "Quality_Score": 60.0})  # fmt: skip
    return forward.freeze_cohort(published, ranked, {"C", "D"}, START, START)


def test_cohort_holds_lists_and_portfolio():
    cohort = _cohort()
    lists = cohort.groupby("List", sort=False)["Ticker"].apply(list).to_dict()
    assert lists["Cheap"] == ["A", "B"]  # the order shown
    assert lists[forward.PORTFOLIO_LIST] == ["C", "D"]  # held, ranked first, an unranked holding after
    assert set(lists) == {"Cheap", "Good", forward.PORTFOLIO_LIST}
    assert cohort.loc[cohort["Ticker"] == "B", "Currency"].iloc[0] == "GBp"


def test_cohort_is_never_rewritten(tmp_path):
    path = tmp_path / "cohort.parquet"
    forward.save_cohort(_cohort(), path)
    with pytest.raises(FileExistsError):
        forward.save_cohort(_cohort(), path)


def test_month_ends_only_once_over():
    ends = forward.month_ends(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31"))
    assert ends == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]  # March 31 is not over yet
    assert forward.month_ends(pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-15")) == []


def test_buy_and_hold_equal_weight():
    """+10% and -10% held without rebalancing average to 0, whatever the path."""
    ledger = pd.DataFrame({"Date": [pd.Timestamp("2024-01-31")] * 2, "Ticker": ["A", "B"], "Return": [0.10, -0.10],
                           "Stale": False})  # fmt: skip
    cohort = pd.DataFrame({"List": ["L", "L"], "Ticker": ["A", "B"]})
    values = forward.series_values(cohort, ledger, START)
    assert values.loc[START, "L"] == 0
    assert values["L"].iloc[-1] == pytest.approx(0.0)
    # Two months: A +10% then +10%, B -10% then 0: (1.21 + 0.9) / 2 - 1, not the mean of monthly means.
    second = pd.DataFrame({"Date": [pd.Timestamp("2024-02-29")] * 2, "Ticker": ["A", "B"], "Return": [0.10, 0.0],
                           "Stale": False})  # fmt: skip
    values = forward.series_values(cohort, pd.concat([ledger, second]), START)
    assert values["L"].iloc[-1] == pytest.approx(((1.21 + 0.9) / 2 - 1) * 100)


def test_delisted_member_held_at_last_value():
    days = pd.bdate_range("2023-12-20", "2024-03-05")
    prices = _prices(days, ["A", "B"], {"A": 0.001, "B": 0.002})
    prices.loc[prices.index > pd.Timestamp("2024-01-31"), "B"] = np.nan  # B stops trading after January
    marks = [START, pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    out = forward.period_returns(prices, marks).set_index(["Date", "Ticker"])
    assert out.loc[(marks[1], "B"), "Return"] > 0 and not out.loc[(marks[1], "B"), "Stale"]
    assert out.loc[(marks[2], "B"), "Return"] == 0 and out.loc[(marks[2], "B"), "Stale"]
    assert not out.loc[(marks[2], "A"), "Stale"]


def test_stored_marks_survive_back_adjustment():
    """A later download is back-adjusted for a dividend (every earlier close scaled): stored returns stay, and the
    new step's ratio is the same either way."""
    days = pd.bdate_range("2023-12-20", "2024-03-05")
    prices = _prices(days, ["A"], {"A": 0.001})
    first = forward.period_returns(prices, [START, pd.Timestamp("2024-01-31")])
    adjusted = prices.copy()
    adjusted.loc[adjusted.index < pd.Timestamp("2024-02-15")] *= 0.98
    second = forward.period_returns(adjusted, [START, pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")])
    ledger = forward.extend_ledger(first, second)
    jan = ledger.loc[ledger["Date"] == pd.Timestamp("2024-01-31"), "Return"]
    assert len(jan) == 1 and jan.iloc[0] == pytest.approx(first["Return"].iloc[0])


def test_usd_conversion_of_minor_units():
    days = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"P.L": [200.0, 200.0, 200.0], "X": [10.0, 10.0, 10.0]}, index=days)
    gbp = pd.Series([1.25, 1.30, 1.30], index=days)
    usd, missing = forward.to_usd(prices, {"P.L": "GBp", "X": "XXX"}, lambda c: gbp if c == "GBP" else None)
    assert usd["P.L"].tolist() == pytest.approx([2.5, 2.6, 2.6])  # 200 pence = 2 pounds, at each day's rate
    assert missing == ["X"]
    _, missing = forward.to_usd(prices, {"P.L": np.nan}, lambda c: gbp)
    assert missing == ["P.L", "X"]  # an unknown currency is reported, never taken for dollars


def _fake_fetch(growth):
    def fetch(tickers, since):
        days = pd.bdate_range(since, "2024-12-31")
        return _prices(days, tickers, {t: growth.get(t, 0.0) for t in tickers})

    return fetch


def test_step_freezes_then_backward_then_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(forward, "START", pd.Timestamp("2024-03-31"))
    published = pd.DataFrame({"List": ["Cheap", "Cheap"], "Ticker": ["A", "B"]})
    ranked = pd.DataFrame({"Ticker": ["A", "B"], "Rank": [1, 2], "currency": "USD"})
    fetch, usd = _fake_fetch({"A": 0.002, "SPY": 0.001, "VT": 0.0005}), lambda c: None
    folder = tmp_path / "forward"
    (folder / "preview").mkdir(parents=True)  # the former phantom preview goes when the cohort is frozen

    cohort, ledger = forward.step(folder, published, ranked, set(), pd.Timestamp("2024-03-29"), fetch, usd)
    assert (folder / "cohort.parquet").exists() and not (folder / "preview").exists()
    assert cohort["Start"].iloc[0] == forward.START and forward.back_start(forward.START) == pd.Timestamp("2023-09-30")
    # Backward: six months back, each month-end over so far; the pivot's own close waits until it is over.
    assert sorted(ledger["Date"].unique())[0] == pd.Timestamp("2023-10-31")
    assert ledger["Date"].max() == pd.Timestamp("2024-02-29")
    html = build_forward(tmp_path / "site", cohort, ledger).read_text(encoding="utf-8")
    assert "Not started yet" in html and "29 March 2024" in html
    assert "Backward" not in html and "Hindsight" not in html  # not shown (Oct 31, 2026: misleading)

    changed = pd.DataFrame({"List": ["Cheap"], "Ticker": ["Z"]})  # the live list moved on: the cohort does not
    cohort, ledger = forward.step(folder, changed, ranked, set(), pd.Timestamp("2024-05-10"), fetch, usd)
    assert set(cohort.loc[cohort["List"] == "Cheap", "Ticker"]) == {"A", "B"}
    back, fwd = forward.phases(ledger, forward.START)
    assert back["Date"].max() == forward.START and back["Date"].nunique() == 6
    assert sorted(fwd["Date"].unique()) == [pd.Timestamp("2024-04-30")]
    values = forward.series_values(cohort, fwd, forward.START)
    assert values.iloc[0].eq(0).all() and values["Cheap"].iloc[-1] > values[forward.BENCHMARKS["VT"]].iloc[-1] > 0
    members = forward.member_table(cohort, ledger).set_index(["List", "Ticker"])
    assert members.loc[("Cheap", "A"), "Return_Back"] > members.loc[("Cheap", "A"), "Return_Fwd"] > 0

    page = build_forward(tmp_path / "site", cohort, ledger, live=changed, copy_static=True)
    html = page.read_text(encoding="utf-8")
    assert "left the list" in html and "Not started yet" not in html
    assert "forward_chart" in html and "backward_chart" not in html
    assert "Not investment, financial, tax or legal advice" in html
    assert forward.step(tmp_path / "empty", published, ranked, set(), pd.Timestamp("2024-05-10"), fetch, usd,
                        offline=True) is None  # fmt: skip


def test_placeholder_names_the_start_and_the_first_result(tmp_path):
    html = build_placeholder(tmp_path / "site").read_text(encoding="utf-8")
    start = forward.last_close(forward.START)
    first = forward.last_close(forward.START + pd.offsets.MonthEnd(1))
    assert f"{start:%d %B %Y}" in html and f"{first:%d %B %Y}" in html and f"{forward.FROZEN_ON:%d %B %Y}" in html
    assert "Starting soon" in html and "Back to the lists" in html


def test_summary_against_benchmarks():
    ledger = pd.DataFrame({"Date": [pd.Timestamp("2024-01-31")] * 4, "Ticker": ["A", "B", "SPY", "VT"],
                           "Return": [0.10, 0.0, 0.03, 0.02], "Stale": [False, True, False, False]})  # fmt: skip
    cohort = pd.DataFrame({"List": ["L", "L"], "Ticker": ["A", "B"], "Start": START})
    values = forward.series_values(cohort, ledger, START)
    table = forward.summary(values, cohort, ledger).set_index("Series")
    assert table.loc["L", "Return"] == pytest.approx(5.0)
    assert table.loc["L", "vs VT"] == pytest.approx(3.0) and table.loc["L", "vs SPY"] == pytest.approx(2.0)
    assert table.loc["L", "Members_Ahead"] == pytest.approx(50.0) and table.loc["L", "Stale"] == 1
    assert np.isnan(table.loc["L", "Annualised"])  # under a year


def test_grade_gate_is_frozen_and_drops_failing_list_members_only(tmp_path):
    cohort = pd.DataFrame({"List": ["Cheap and good", "Cheap and good", "Cheapest", forward.PORTFOLIO_LIST],
                           "Ticker": ["A", "B", "C", "B"]})  # fmt: skip
    grades = pd.DataFrame({"Ticker": ["A", "B"], "Moat": [4.0, 2.0], "Management": [4.0, 3.0],
                           "Passes": [True, False], "Graded_At": pd.to_datetime(["2026-10-31"] * 2)})  # fmt: skip
    path = forward.freeze_gate(grades, tmp_path / forward.GATE_FILE)
    with pytest.raises(FileExistsError):
        forward.freeze_gate(grades, path)
    gated = forward.apply_gate(cohort, pd.read_parquet(path))
    assert list(zip(gated["List"], gated["Ticker"], strict=True)) == [("Cheap and good", "A"), ("Cheapest", "C"),
                                                         (forward.PORTFOLIO_LIST, "B")]  # fmt: skip
    assert gated.attrs["gated_out"] == 1  # C is ungraded and stays; B stays in the portfolio


MONTHS = [
    pd.Timestamp("2024-03-31"),
    pd.Timestamp("2024-04-30"),
    pd.Timestamp("2024-05-31"),
    pd.Timestamp("2024-06-30"),
]


def _setup(tmp_path, monkeypatch, tickers=("A", "B")):
    """A stored cohort whose pivot is March (no backward months)."""
    monkeypatch.setattr(forward, "BACK_MONTHS", 0)
    folder = tmp_path / "forward"
    published = pd.DataFrame({"List": "L", "Ticker": list(tickers)})
    ranked = pd.DataFrame({"Ticker": list(tickers), "Rank": range(1, len(tickers) + 1), "currency": "USD"})
    forward.save_cohort(
        forward.freeze_cohort(published, ranked, set(), MONTHS[0], MONTHS[0]), folder / "cohort.parquet"
    )
    return folder


def _stored(folder, rows):
    """Ledger rows (ticker, return, stale) at April, written as before the guards."""
    frame = pd.DataFrame(rows, columns=["Ticker", "Return", "Stale"]).assign(Date=MONTHS[1])
    frame.to_parquet(folder / "ledger.parquet", index=False)


def _flat(since, tickers):
    days = pd.bdate_range(since, "2024-07-15")
    return pd.DataFrame({t: np.full(len(days), 100.0) for t in tickers}, index=days)


def test_failed_download_is_not_stored_and_is_retried(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    _stored(folder, [(t, 0.01, False) for t in ("A", "B", "SPY", "VT")])

    def broken(tickers, since):
        raise ConnectionError("Yahoo down")

    def no_spy(tickers, since):
        return _flat(since, tickers).assign(SPY=np.nan)

    for fetch in (broken, no_spy):
        assert set(forward.update(folder, pd.Timestamp("2024-06-02"), fetch, lambda c: None)["Date"]) == {MONTHS[1]}
        assert set(pd.read_parquet(folder / "ledger.parquet")["Date"]) == {MONTHS[1]}

    ledger = forward.update(folder, pd.Timestamp("2024-07-02"), lambda t, s: _flat(s, t), lambda c: None)
    assert set(ledger["Date"]) == {MONTHS[1], MONTHS[2], MONTHS[3]}  # the next run adds both months


def test_member_without_a_price_catches_up_without_rewriting(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, tickers=("A", "B", "C", "D", "E", "F"))
    _stored(folder, [(t, 0.0, t == "A") for t in ("A", "B", "C", "D", "E", "F", "SPY", "VT")])  # A's April price failed

    def fetch(tickers, since):
        days = pd.bdate_range(since, "2024-07-15")
        return _flat(since, tickers).assign(A=np.where(days <= MONTHS[0], 100.0, 130.0),  # rose while missing
                                            F=np.where(days <= MONTHS[1], 100.0, np.nan))  # delisted after April  # fmt: skip

    ledger = forward.update(folder, pd.Timestamp("2024-06-02"), fetch, lambda c: None)
    row = ledger[ledger["Date"] == MONTHS[2]].set_index("Ticker")
    assert row.at["A", "Return"] == pytest.approx(0.30) and not row.at["A", "Stale"]
    assert ledger[(ledger["Date"] == MONTHS[1]) & (ledger["Ticker"] == "A")]["Return"].iloc[0] == 0  # not rewritten
    assert row.at["F", "Return"] == 0 and row.at["F", "Stale"]
    assert forward.growth(ledger, MONTHS[0]).iloc[-1]["A"] == pytest.approx(1.30)  # counted once


def test_price_breaks_leave_out_a_change_of_unit_and_keep_a_collapse():
    days = pd.bdate_range("2024-04-01", periods=4)
    prices = pd.DataFrame(
        {"UNIT": [200.0, 200.0, 2.0, 2.2], "FALL": [100.0, 100.0, 20.0, 20.0], "OK": 100.0}, index=days
    )
    clean, breaks = forward.price_breaks(prices)
    assert clean["UNIT"].iloc[-1] / clean["UNIT"].iloc[0] == pytest.approx(1.1)  # only the real 10%
    assert clean["FALL"].iloc[-1] / clean["FALL"].iloc[0] == pytest.approx(0.2)  # a 5x fall may be real: it stays
    assert set(breaks["Ticker"]) == {"UNIT", "FALL"}


def test_break_marks_its_month_for_checking(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    _stored(folder, [(t, 0.0, False) for t in ("A", "B", "SPY", "VT")])

    def fetch(tickers, since):
        days = pd.bdate_range(since, "2024-07-15")
        return _flat(since, tickers).assign(A=np.where(days < pd.Timestamp("2024-05-10"), 200.0, 2.0))

    ledger = forward.update(folder, pd.Timestamp("2024-06-02"), fetch, lambda c: None)
    row = ledger[ledger["Date"] == MONTHS[2]].set_index("Ticker")
    assert row.at["A", "Return"] == pytest.approx(0.0) and row.at["A", "Check"] and not row.at["B", "Check"]
    assert not ledger[ledger["Date"] == MONTHS[1]]["Check"].any()  # older rows read as False


def test_split_recorded_late_is_carried_once_into_the_next_month(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    # April was stored as -50%: a 2:1 split Yahoo had not recorded yet. Now it has: the history is flat.
    _stored(folder, [("A", -0.5, False), ("B", 0.0, False), ("SPY", 0.0, False), ("VT", 0.0, False)])
    restated = lambda t, s: _flat(s, t)  # noqa: E731

    ledger = forward.update(folder, pd.Timestamp("2024-06-02"), restated, lambda c: None)
    assert ledger[(ledger["Date"] == MONTHS[1]) & (ledger["Ticker"] == "A")]["Return"].iloc[0] == -0.5  # still stored
    row = ledger[ledger["Date"] == MONTHS[2]].set_index("Ticker")
    assert row.at["A", "Return"] == pytest.approx(1.0) and row.at["A", "Check"] and not row.at["B", "Check"]
    assert forward.growth(ledger, MONTHS[0]).iloc[-1]["A"] == pytest.approx(1.0)  # the true return: flat

    ledger = forward.update(folder, pd.Timestamp("2024-07-02"), restated, lambda c: None)
    assert ledger[ledger["Date"] == MONTHS[3]].set_index("Ticker").at["A", "Return"] == pytest.approx(0.0)  # only once
    assert forward.growth(ledger, MONTHS[0]).iloc[-1]["A"] == pytest.approx(1.0)


def test_back_adjustment_for_a_dividend_triggers_no_correction(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    _stored(folder, [(t, 0.0, False) for t in ("A", "B", "SPY", "VT")])

    def fetch(tickers, since):  # a dividend in May scales every close before it, April's two marks alike
        days = pd.bdate_range(since, "2024-07-15")
        return _flat(since, tickers).assign(A=np.where(days < pd.Timestamp("2024-05-15"), 98.0, 100.0))

    ledger = forward.update(folder, pd.Timestamp("2024-06-02"), fetch, lambda c: None)
    row = ledger[ledger["Date"] == MONTHS[2]].set_index("Ticker")
    assert row.at["A", "Return"] == pytest.approx(100 / 98 - 1) and not row.at["A", "Check"]


def test_late_list_is_added_once_outside_the_cohort_and_backfilled(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    _stored(folder, [(t, 0.01, False) for t in ("A", "B", "SPY", "VT")])
    monkeypatch.setattr(forward, "LATE_LISTS", {"Repeaters": pd.Timestamp("2024-04-07")})
    frozen = (folder / "cohort.parquet").read_bytes()
    ranked = pd.DataFrame({"Ticker": ["A", "NEW"], "Rank": [1, 2], "currency": "USD"})
    tabs = pd.DataFrame({"List": ["Repeaters", "Repeaters", "L"], "Ticker": ["NEW", "A", "B"]})
    fetch, today = (lambda t, s: _flat(s, t)), pd.Timestamp("2024-05-02")

    cohort, ledger = forward.step(folder, tabs, ranked, set(), today, fetch, lambda c: None, late={"Repeaters": tabs})
    assert (folder / "cohort.parquet").read_bytes() == frozen
    assert cohort[cohort["List"] == "Repeaters"]["Ticker"].tolist() == ["NEW", "A"]  # its own order, nothing else
    assert (cohort["Start"] == MONTHS[0]).all()
    assert cohort.loc[cohort["List"] == "Repeaters", "Published_On"].iloc[0] == pd.Timestamp("2024-04-07")
    assert ledger[ledger["Ticker"] == "NEW"]["Date"].tolist() == [MONTHS[1]]  # backfilled
    assert ledger[ledger["Ticker"] == "A"]["Return"].iloc[0] == 0.01  # stored rows kept

    other = tabs.assign(Ticker=["X", "Y", "Z"])  # a later list never replaces it
    cohort, _ = forward.step(folder, tabs, ranked, set(), today, fetch, lambda c: None, late={"Repeaters": other})
    assert "X" not in set(cohort["Ticker"])
    gate = pd.DataFrame({"Ticker": ["NEW"], "Moat": [1.0], "Management": [1.0], "Passes": [False],
                         "Graded_At": pd.to_datetime(["2024-04-07"])})  # fmt: skip
    assert "NEW" in set(forward.apply_gate(cohort, gate)["Ticker"])  # the gate does not apply to it

    # A list born later still: its own file, the first addition left as it was.
    first = (folder / forward.ADDED_FILE).read_bytes()
    monkeypatch.setattr(forward, "LATE_LISTS", {"Repeaters": pd.Timestamp("2024-04-07"),
                                                "Fast": pd.Timestamp("2024-04-14")})  # fmt: skip
    fast = pd.DataFrame({"List": ["Fast"], "Ticker": ["B"]})
    cohort, _ = forward.step(folder, tabs, ranked, set(), today, fetch, lambda c: None, late={"Fast": fast})
    assert (folder / forward.ADDED_FILE).read_bytes() == first
    assert (folder / "cohort_added_2024-04-14.parquet").exists()
    assert cohort[cohort["List"] == "Fast"]["Ticker"].tolist() == ["B"]
    assert cohort[cohort["List"] == "Repeaters"]["Ticker"].tolist() == ["NEW", "A"]


def test_a_list_with_a_pivot_of_its_own_is_bought_at_its_close_and_measured_apart(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    _stored(folder, [(t, 0.0, False) for t in ("A", "B", "SPY", "VT")])
    monkeypatch.setattr(forward, "LATE_LISTS", {"Energy": pd.Timestamp("2024-05-01")})
    monkeypatch.setattr(forward, "LATE_PIVOTS", {"Energy": MONTHS[1]})  # April's close, a month after the others
    frozen = (folder / "cohort.parquet").read_bytes()
    ranked = pd.DataFrame({"Ticker": ["A", "NEW"], "Rank": [1, 2], "currency": "USD"})
    tabs = pd.DataFrame({"List": ["Energy"], "Ticker": ["NEW"]})

    def fetch(tickers, since):  # NEW doubles in April, before its pivot, then gains 10% in May
        days = pd.bdate_range(since, "2024-07-15")
        new = np.select([days <= MONTHS[0], days <= MONTHS[1]], [50.0, 100.0], 110.0)
        return _flat(since, tickers).assign(**({"NEW": new} if "NEW" in tickers else {}))

    cohort, ledger = forward.step(folder, tabs, ranked, set(), pd.Timestamp("2024-06-02"), fetch, lambda c: None,
                                  late={"Energy": tabs})  # fmt: skip
    assert (folder / "cohort.parquet").read_bytes() == frozen
    assert cohort.loc[cohort["List"] == "Energy", "Start"].eq(MONTHS[1]).all()
    assert cohort.loc[cohort["List"] == "L", "Start"].eq(MONTHS[0]).all()
    assert [when for when, _ in forward.pivot_groups(cohort)] == [MONTHS[0], MONTHS[1]]
    members = forward.member_table(cohort, ledger).set_index(["List", "Ticker"])
    assert members.loc[("Energy", "NEW"), "Return_Fwd"] == pytest.approx(10.0)  # April's doubling is not its own
    html = build_forward(tmp_path / "site", cohort, ledger).read_text(encoding="utf-8")
    assert "Forward: since 30 April 2024 (Energy)" in html and "a start of its own" in html
    assert html.count('id="forward_chart') == 2  # one chart per pivot


def test_dividend_recorded_late_is_carried_once_into_the_next_month(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    # April was stored flat; Yahoo then recorded a 2% dividend gone ex in late April: April's start is scaled down.
    _stored(folder, [(t, 0.0, False) for t in ("A", "B", "SPY", "VT")])

    def fetch(tickers, since):
        days = pd.bdate_range(since, "2024-07-15")
        return _flat(since, tickers).assign(A=np.where(days < pd.Timestamp("2024-04-25"), 98.0, 100.0))

    ledger = forward.update(folder, pd.Timestamp("2024-06-02"), fetch, lambda c: None)
    row = ledger[ledger["Date"] == MONTHS[2]].set_index("Ticker")
    assert row.at["A", "Return"] == pytest.approx(100 / 98 - 1) and row.at["A", "Check"]
    assert forward.growth(ledger, MONTHS[0]).iloc[-1]["A"] == pytest.approx(100 / 98)  # the dividend, counted once
    ledger = forward.update(folder, pd.Timestamp("2024-07-02"), fetch, lambda c: None)
    assert ledger[ledger["Date"] == MONTHS[3]].set_index("Ticker").at["A", "Return"] == pytest.approx(0.0)


def _both(total, price):
    """A fetch result like `fetch.total_and_price_history`."""
    return pd.concat({"Total": total, "Price": price}, axis=1)


def test_price_return_is_stored_and_the_dividends_split_out(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch)
    _stored(folder, [(t, 0.0, False) for t in ("A", "B", "SPY", "VT")])

    def fetch(tickers, since):  # A pays 3% in May (total up 3%, price flat); B's Yahoo dividend is 40%: a unit error
        days = pd.bdate_range(since, "2024-07-15")
        total = _flat(since, tickers).assign(A=np.where(days <= MONTHS[1], 100 / 1.03, 100.0),
                                             B=np.where(days <= MONTHS[1], 100 / 1.4, 100.0))  # fmt: skip
        return _both(total, _flat(since, tickers))

    ledger = forward.update(folder, pd.Timestamp("2024-06-02"), fetch, lambda c: None)
    assert ledger[ledger["Date"] == MONTHS[1]]["Price_Return"].isna().all()  # stored before: unknown, kept
    row = ledger[ledger["Date"] == MONTHS[2]].set_index("Ticker")
    assert row.at["A", "Return"] == pytest.approx(0.03) and row.at["A", "Price_Return"] == pytest.approx(0.0)
    assert not row.at["A", "Check"] and row.at["B", "Check"]  # 3% is a dividend, 40% is to check

    cohort = forward.load_cohort(folder)
    fwd = ledger[ledger["Date"] > MONTHS[1]]
    table = forward.summary(forward.series_values(cohort, fwd, MONTHS[1]), cohort, fwd).set_index("Series")
    assert table.at["L", "Dividends"] == pytest.approx((3 + 40) / 2)  # equal weight: the mean of A's and B's
    assert "Dividends" not in forward.summary(forward.series_values(cohort, ledger, MONTHS[0]), cohort, ledger)
