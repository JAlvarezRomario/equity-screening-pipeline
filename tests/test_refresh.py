"""Auto refresh: which cached companies get fresh fundamentals and which only new prices."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq import refresh
from fq.universe import Selection

NOW = pd.Timestamp("2026-10-19 09:00")


def cached(cfg, fetched="2026-10-10 08:00", quarter="2026-06-30", annual="2025-12-31", **extra) -> pd.DataFrame:
    """One cached company with its newest quarter / annual period ends and fetch time."""
    raw = make_raw(cfg, n=1).assign(Ticker="AAA", Fetched_At=fetched, last_quarter=quarter, **extra)
    raw[f"PeriodEnd_{cfg.latest}"] = annual
    raw["PeriodEnd_TTM"] = quarter
    return raw


def why(cfg, raw) -> str | None:
    return refresh.plan(raw, list(raw["Ticker"]), cfg, NOW).get("AAA")


def test_uncached_ticker_is_new(cfg):
    assert refresh.plan(None, ["X"], cfg, NOW) == {"X": "new"}
    assert refresh.plan(cached(cfg), ["AAA", "X"], cfg, NOW) == {"X": "new"}


def test_recent_fetch_with_nothing_due_needs_prices_only(cfg):
    # Next quarter (Sep 30) is expected on Yahoo around Nov 14: not yet.
    assert why(cfg, cached(cfg)) is None


def test_quarter_due_after_last_fetch(cfg):
    # Quarter to Mar 31 expected around Aug 14; last fetch was in July.
    assert why(cfg, cached(cfg, fetched="2026-07-20 08:00", quarter="2026-03-31")) == "quarter due"


def test_due_date_before_last_fetch_does_not_retrigger(cfg):
    # Half-yearly reporter: Aug 14 came and went, but we already looked again on Sep 1.
    assert why(cfg, cached(cfg, fetched="2026-09-01 08:00", quarter="2026-03-31")) is None


def test_annual_due(cfg):
    raw = cached(cfg, fetched="2026-09-01 08:00", quarter="2026-03-31", annual="2025-06-30")
    raw["last_quarter"] = raw["PeriodEnd_TTM"] = "N/A"  # annual-only reporter
    assert why(cfg, raw) == "annual due"  # FY to Jun 30 expected around Sep 28


def test_earnings_call_passed(cfg):
    call = pd.Timestamp("2026-10-15").timestamp()
    assert why(cfg, cached(cfg, earningsTimestamp=call)) == "earnings passed"


def test_earnings_call_too_recent_waits_for_yahoo(cfg):
    call = pd.Timestamp("2026-10-18").timestamp()  # within earnings_settle_days
    assert why(cfg, cached(cfg, earningsTimestamp=call)) is None


def test_old_or_unknown_fetch_time(cfg):
    assert why(cfg, cached(cfg, fetched="2026-06-01 08:00", quarter="2026-03-31", annual="2025-12-31")) in {
        "quarter due",
        "old",
    }
    raw = cached(cfg, fetched="2026-07-01 08:00", quarter="2026-03-31")
    raw["last_quarter"] = raw["PeriodEnd_TTM"] = None
    raw[f"PeriodEnd_{cfg.latest}"] = None
    assert why(cfg, raw.assign(Fetched_At="2026-06-01 08:00")) == "old"
    assert why(cfg, raw.assign(Fetched_At=None)) == "old"


@pytest.mark.parametrize("concept", ["Total Revenue", "Operating Cash Flow"])
@pytest.mark.parametrize(("fetched", "expected"), [("2026-10-16 08:00", None), ("2026-10-05 08:00", "incomplete")])
def test_incomplete_data_retried_after_a_week(cfg, fetched, expected, concept):
    raw = cached(cfg, fetched=fetched)
    raw[f"{concept}_TTM"] = raw[f"{concept}_{cfg.latest}"] = np.nan
    assert why(cfg, raw) == expected


@pytest.mark.parametrize(
    ("mode", "fetched", "priced"),
    [
        ("auto", ["NEW", "DUE"], ["OK"]),
        ("full", ["OK", "DUE", "NEW"], None),
        ("prices", ["NEW"], ["OK", "DUE"]),
        ("new", ["NEW"], None),
    ],
)
def test_fetch_modes(cfg, monkeypatch, mode, fetched, priced):
    from fq import fetch, pipeline

    raw = pd.concat(
        [
            cached(cfg).assign(Ticker="OK"),
            cached(cfg, fetched="2026-07-20 08:00", quarter="2026-03-31").assign(Ticker="DUE"),
        ],
        ignore_index=True,
    )
    fetch.save_raw(raw, cfg.raw_path)
    calls = {}

    def fake_fetch(todo, cfg, existing):
        calls["fetched"] = todo
        return existing

    def fake_prices(df, todo, cfg):
        calls["priced"] = todo
        return df

    plan = refresh.plan
    monkeypatch.setattr(refresh, "plan", lambda existing, tickers, cfg, now: plan(existing, tickers, cfg, NOW))
    monkeypatch.setattr(fetch, "fetch_universe", fake_fetch)
    monkeypatch.setattr(fetch, "refresh_prices", fake_prices)
    monkeypatch.setattr(fetch, "save_raw", lambda df, path: None)
    pipeline.fetch_data(Selection(label="test", tickers=["OK", "DUE", "NEW"]), cfg, mode)
    assert sorted(calls["fetched"]) == sorted(fetched)
    assert calls.get("priced") == priced


def test_fetched_with_other_fields_is_due(cfg):
    assert why(cfg, cached(cfg, Fields_Version="0ld")) == "new fields"
    assert why(cfg, cached(cfg).drop(columns="Fields_Version")) == "new fields"  # before fields were tracked


def test_select_spends_the_budget_by_priority():
    run = {"NEW1": "new", "NEW2": "new", "OLD": "old", "Q": "quarter due"}
    bench = {"B_EARN": "earnings passed", "B_FIELDS": "new fields", "B_NEW": "new", "Q": "old"}
    now, later = refresh.select(run, bench, cap=5)
    assert now == ["NEW1", "NEW2", "Q", "OLD", "B_NEW"]  # run first, each by priority; Q counted once
    assert later == ["B_EARN", "B_FIELDS"]
    assert refresh.select(run, {}, cap=1) == (["NEW1", "NEW2"], ["Q", "OLD"])  # own new tickers always
    assert refresh.select(run, bench, cap=0)[1] == []  # no cap


def test_auto_run_refreshes_due_benchmark_companies_within_the_cap(cfg, monkeypatch):
    from dataclasses import replace

    from fq import fetch, pipeline

    raw = pd.concat(
        [
            cached(cfg).assign(Ticker="OK"),
            cached(cfg, fetched="2026-07-20 08:00", quarter="2026-03-31").assign(Ticker="B_DUE"),
            cached(cfg, Fields_Version="0ld").assign(Ticker="B_FIELDS"),
        ],
        ignore_index=True,
    )
    fetch.save_raw(raw, cfg.raw_path)
    calls = {}
    plan = refresh.plan
    monkeypatch.setattr(refresh, "plan", lambda existing, tickers, cfg, now: plan(existing, tickers, cfg, NOW))
    monkeypatch.setattr(
        fetch, "fetch_universe", lambda todo, cfg, existing: calls.setdefault("fetched", todo) and existing
    )
    monkeypatch.setattr(fetch, "refresh_prices", lambda df, todo, cfg: calls.setdefault("priced", todo) and df)
    monkeypatch.setattr(fetch, "save_raw", lambda df, path: None)
    bench = Selection("benchmark", ["OK", "B_DUE", "B_FIELDS", "B_NEW"])
    pipeline.fetch_data(Selection("portfolio", ["OK", "MINE"]), replace(cfg, max_full_fetches=3), "auto", bench)
    assert calls["fetched"] == ["MINE", "B_NEW", "B_DUE"]  # B_FIELDS waits for a later run
    assert calls["priced"] == ["OK", "B_FIELDS"]


def test_save_raw_keeps_the_old_cache_when_a_write_fails(cfg, tmp_path, monkeypatch):
    """A run stopped mid-write must not corrupt the cache: the write goes to a temporary file first."""
    from fq import fetch

    path = tmp_path / "raw.parquet"
    fetch.save_raw(make_raw(cfg, n=2), path)

    def broken(self, target, **kwargs):
        open(target, "wb").write(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)
    with pytest.raises(KeyboardInterrupt):
        fetch.save_raw(make_raw(cfg, n=3), path)
    monkeypatch.undo()
    assert len(fetch.load_raw(path)) == 2
    assert list(tmp_path.iterdir()) == [path]
