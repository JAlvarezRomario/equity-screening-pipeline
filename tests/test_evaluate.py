"""Forward tracking of score snapshots (no network: synthetic prices)."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.evaluate import (
    MARKET,
    evaluate,
    evaluate_snapshot,
    forward_returns,
    load_raw_archives,
    load_snapshots,
    method_files,
    method_fingerprint,
    newey_west_t,
    save_raw_archive,
    save_snapshot,
    stability,
    to_usd,
)
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.html import build_site
from fq.report.tables import tracking_tables


def _snap(n=50):
    """Companies ranked 1..n, Value_Score falling with the rank."""
    tickers = [f"T{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Rank": np.arange(1, n + 1),
            "Value_Rank": np.arange(1, n + 1),
            "Quality_Rank": np.arange(1, n + 1),
            "Value_Score": np.linspace(100, 2, n),
            "Quality_Score": 50.0,
            "Pct_EPV_Yield": np.linspace(100, 2, n),
        }
    )


def _prices(tickers, start, yearly):
    """Daily prices compounding at `yearly` % per ticker."""
    days = pd.bdate_range(start, periods=400)
    t = np.arange(len(days))[:, None] / 252
    return pd.DataFrame(100 * (1 + np.asarray(yearly) / 100) ** t, index=days, columns=tickers)


def test_forward_returns_and_future_horizons():
    prices = _prices(["A", "B"], "2024-01-01", [10.0, -10.0])
    fwd = forward_returns(prices, pd.Timestamp("2024-01-01"), 12)
    assert fwd["A"] == pytest.approx(10.0, abs=0.6) and fwd["B"] == pytest.approx(-10.0, abs=0.6)
    assert forward_returns(prices, pd.Timestamp("2024-09-01"), 12) is None  # end date after the last price
    assert forward_returns(prices, pd.Timestamp("2023-01-01"), 3) is None  # before the history


def test_returns_are_converted_to_usd():
    prices = _prices(["US", "JP", "UK", "XX"], "2024-01-01", [0.0, 0.0, 0.0, 0.0])
    yen = pd.Series(np.linspace(0.010, 0.008, len(prices)), index=prices.index)  # yen fell 20%
    pound = pd.Series(1.25, index=prices.index[::5])  # sparse quotes are carried forward
    ccy = {"US": "USD", "JP": "JPY", "UK": "GBp", "XX": "XXX"}
    usd = to_usd(prices, ccy, {"JPY": yen, "GBP": pound, "XXX": None})
    fwd = forward_returns(usd, prices.index[0], 12)
    assert fwd["US"] == pytest.approx(0) and fwd["UK"] == pytest.approx(0) and fwd["XX"] == pytest.approx(0)
    assert fwd["JP"] < -10  # flat in yen, a loss in USD
    assert usd["UK"].iloc[-1] == pytest.approx(100 * 1.25 / 100)  # pence -> pounds -> USD


def test_better_ranks_earning_more_show_positive_excess_and_ic():
    snap = _snap()
    fwd = pd.Series(np.linspace(30, -20, 50), index=snap["Ticker"])  # return falls with the rank
    rows, ic = evaluate_snapshot(snap, fwd, top=10)
    by = {r["List"]: r for r in rows}
    assert by["Top 10 combined"]["Excess"] > 15 and by["Top 10 combined"]["N"] == 10
    assert by["Value quintile 1"]["Return"] > by["Value quintile 5"]["Return"]
    assert ic["Value_Score"] == pytest.approx(1.0) and ic["EPV_Yield"] == pytest.approx(1.0)
    assert np.isnan(ic["Quality_Score"])  # a constant score says nothing


def test_lists_are_also_measured_against_the_market():
    snap = _snap()
    tickers = list(snap["Ticker"])
    prices = _prices([*tickers, MARKET], "2024-01-01", [*np.linspace(30, -20, 50), 8.0])
    lists, _ = evaluate({pd.Timestamp("2024-01-02"): snap}, prices, horizons=(12,), top=10)
    top = lists.set_index("List").loc["Top 10 combined"]
    assert top["Excess_Market"] == pytest.approx(top["Return"] - 8.0, abs=0.6)
    assert MARKET not in set(lists["List"])  # the market is a yardstick, never a company of the snapshot
    no_market = evaluate({pd.Timestamp("2024-01-02"): snap}, prices[tickers], horizons=(12,), top=10)[0]
    assert no_market["Excess_Market"].isna().all()
    tables = tracking_tables(no_market, pd.DataFrame(columns=["Method", "Months", "Measure", "Snapshots", "IC"]))
    assert [t[0].split(" - ")[0] for t in tables] == ["Excess return over the median company (%)"]  # no empty one


def test_newey_west_t_discounts_overlapping_snapshots():
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 1, 400)).rolling(12).mean().dropna() + 0.1  # overlapping, autocorrelated
    plain = noise.mean() / (noise.std(ddof=0) / np.sqrt(len(noise)))
    assert newey_west_t(noise, 0) == pytest.approx(plain)
    assert abs(newey_west_t(noise, 11)) < abs(plain) / 2
    assert np.isnan(newey_west_t(pd.Series([0.1, 0.2]), 1))  # too few snapshots


def test_ic_t_is_reported_per_measure(cfg):
    rng = np.random.default_rng(1)
    snaps = {}
    for w in range(6):  # each week's scores a little different, so the ICs vary
        snap = _snap()
        snap["Value_Score"] += rng.normal(0, 30, len(snap))
        snaps[pd.Timestamp("2024-01-02") + pd.DateOffset(weeks=w)] = snap
    prices = _prices([f"T{i}" for i in range(50)], "2024-01-01", np.linspace(30, -20, 50) + rng.normal(0, 5, 50))
    _, ic = evaluate(snaps, prices, horizons=(3,), top=10)
    row = ic.set_index("Measure").loc["Value_Score"]
    assert row["Snapshots"] == 6 and row["IC_t"] > 0
    assert " t +" in tracking_tables(pd.DataFrame(), ic)[0][2][0][1]


def test_rank_stability_between_consecutive_snapshots():
    a, b = _snap(), _snap()
    b["Rank"] = b["Rank"].to_numpy()[::-1]  # everything turned over: a data break, or a new method
    b["Method"] = a["Method"] = "m1"
    c = a.assign(Method="m2")
    out = stability({pd.Timestamp("2024-01-02"): a, pd.Timestamp("2024-02-01"): a, pd.Timestamp("2024-03-01"): b,
                     pd.Timestamp("2024-04-01"): c})  # fmt: skip
    assert list(out["Rank"]) == pytest.approx([1.0, -1.0])  # m2 has no earlier snapshot of its own
    assert (out["N"] == 50).all()


def test_mood_list_is_compared_with_cheap_companies_it_leaves_out():
    snap = _snap().assign(Dislocation=False, Data_Suspect=False)  # Value_Score falls from 100
    snap.loc[[0, 2], "Dislocation"] = True
    snap.loc[1, "Data_Suspect"] = True
    fwd = pd.Series(0.0, index=snap["Ticker"])
    rows, _ = evaluate_snapshot(snap, fwd, top=5)
    by = {r["List"]: r for r in rows}
    cheap = int((snap["Value_Score"] >= 80).sum())
    assert by["Cheap (Value >= 80)"]["N"] == cheap - 1  # suspect left out
    assert by["Cheap, explained"]["N"] == cheap - 3 and by["Dislocation"]["N"] == 2


def test_cheap_cyclicals_at_peak_margins_are_followed_apart():
    snap = _snap().assign(Dislocation=False, Data_Suspect=False, Peak_Margin=False)
    snap.loc[[0, 1], "Peak_Margin"] = True
    fwd = pd.Series(0.0, index=snap["Ticker"])
    fwd[["T0", "T1"]] = 1000.0
    by = {r["List"]: r for r in evaluate_snapshot(snap, fwd, top=5)[0]}
    assert by["Cheap, peak margins"]["N"] == 2
    assert by["Top 5 combined"]["Return"] == 0.0  # out of the top lists
    old = {r["List"] for r in evaluate_snapshot(snap.drop(columns="Peak_Margin"), fwd, top=5)[0]}
    assert "Cheap, peak margins" not in old  # snapshots saved before the flag


def test_cheap_distressed_companies_are_followed_apart():
    snap = _snap().assign(Dislocation=False, Data_Suspect=False, Distress=False)
    snap.loc[[0, 2], "Distress"] = True
    fwd = pd.Series(0.0, index=snap["Ticker"])
    fwd[["T0", "T2"]] = -90.0
    by = {r["List"]: r for r in evaluate_snapshot(snap, fwd, top=5)[0]}
    assert by["Cheap, distressed"]["N"] == 2 and by["Cheap, distressed"]["Return"] == -90.0
    assert by["Top 5 combined"]["Return"] == 0.0  # out of the top lists


def test_flagged_companies_stay_out_of_the_top_lists():
    snap = _snap().assign(Data_Suspect=False)
    snap.loc[0, "Data_Suspect"] = True
    fwd = pd.Series(0.0, index=snap["Ticker"])
    fwd["T0"] = 1000.0
    rows, _ = evaluate_snapshot(snap, fwd, top=5)
    assert {r["List"]: r for r in rows}["Top 5 combined"]["Return"] == 0.0


def test_snapshots_round_trip_and_evaluate_skips_young_ones(cfg):
    ranked = apply_dual_ranking(calculate_metrics(make_raw(cfg, n=30), cfg), cfg)
    folder = cfg.snapshot_dir
    save_snapshot(ranked, folder, "test", pd.Timestamp("2024-01-02"))
    save_snapshot(ranked, folder, "test", pd.Timestamp("2024-01-02 18:00"))  # same day: replaced
    save_snapshot(ranked, folder, "test", pd.Timestamp("2024-12-01"))
    save_snapshot(ranked, folder, "other", pd.Timestamp("2024-01-02"))
    snaps = load_snapshots(folder, "test")
    assert list(snaps) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-01")]
    assert {"Ticker", "Value_Score", "Pct_EPV_Yield"} <= set(snaps[pd.Timestamp("2024-01-02")].columns)

    tickers = list(ranked["Ticker"])
    prices = _prices(tickers, "2024-01-01", np.linspace(20, -10, len(tickers)))
    lists, ic = evaluate(snaps, prices, horizons=(3, 12), top=5)
    counts = lists.groupby("Months")["Snapshots"].max()
    assert counts[3] == 2 and counts[12] == 1  # the December snapshot is too young for 12 months
    assert {"Value_Score", "ROIC_Norm"} <= set(ic["Measure"])

    lists.to_parquet(cfg.evaluation_paths[0])
    ic.to_parquet(cfg.evaluation_paths[1])
    tables = tracking_tables(lists, ic)
    assert tables[0][1] == ["List", "3 months", "12 months"]
    page = build_site(ranked, cfg).read_text(encoding="utf-8")
    assert "Track record" not in page  # user's choice: shown by `fq evaluate`, not on the published site


def test_methods_are_recorded_and_kept_apart(cfg):
    method = method_fingerprint()
    assert len(method) == 8 and method == method_fingerprint()  # stable between calls
    names = {f.name for f in method_files()}
    assert {"valuation.py", "capital.py", "ranking.py", "model_overrides.txt"} <= names  # the scoring code is in
    ranked = apply_dual_ranking(calculate_metrics(make_raw(cfg, n=30), cfg), cfg)
    folder = cfg.snapshot_dir
    save_snapshot(ranked, folder, "test", pd.Timestamp("2024-01-02"))
    save_snapshot(ranked, folder, "test", pd.Timestamp("2024-02-01"), method="oldoldol")
    snaps = load_snapshots(folder, "test")
    assert snaps[pd.Timestamp("2024-01-02")]["Method"].iloc[0] == method
    tickers = list(ranked["Ticker"])
    prices = _prices(tickers, "2024-01-01", np.linspace(20, -10, len(tickers)))
    lists, ic = evaluate(snaps, prices, horizons=(3,), top=5)
    assert set(lists["Method"]) == {method, "oldoldol"}
    assert (lists.groupby("Method")["Snapshots"].max() == 1).all()  # never pooled
    tables = tracking_tables(lists, ic, method)
    assert method in tables[0][0] and "not today's" not in tables[0][0]
    assert "not today's" in tracking_tables(lists, ic, "newnewne")[0][0]


def test_raw_archive_rescored_with_todays_method(cfg):
    from fq.pipeline import rescore as _rescore

    raw = make_raw(cfg, n=30)
    day = pd.Timestamp("2024-01-02")
    save_raw_archive(raw, cfg.snapshot_dir, day, benchmark=[])
    assert list(load_raw_archives(cfg.snapshot_dir)) == [day]
    ranked = apply_dual_ranking(calculate_metrics(raw, cfg), cfg)
    save_snapshot(ranked, cfg.snapshot_dir, "test", day, method="oldoldol")
    stored = load_snapshots(cfg.snapshot_dir, "test")
    test_cfg = cfg.__class__(data_dir=cfg.data_dir, output_dir=cfg.output_dir, label="test")
    again = _rescore(test_cfg, stored, until=pd.Timestamp("2025-01-01"))
    assert again[day]["Method"].iloc[0] == method_fingerprint()
    by = lambda s: s.set_index("Ticker")["Value_Score"].sort_index()  # noqa: E731
    pd.testing.assert_series_equal(by(again[day]), by(stored[day]))  # same code, same scores
    assert _rescore(test_cfg, stored, until=pd.Timestamp("2023-12-31")) == {}  # too young to evaluate


def test_unchanged_raw_archive_is_a_hard_link_and_every_date_stays(cfg):
    raw = make_raw(cfg, n=5)
    days = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    first = save_raw_archive(raw, cfg.snapshot_dir, days[0], benchmark=[])
    second = save_raw_archive(raw, cfg.snapshot_dir, days[1], benchmark=[])
    assert list(load_raw_archives(cfg.snapshot_dir)) == days[:2]
    assert second.stat().st_ino == first.stat().st_ino  # the same file, no second copy
    changed = raw.assign(currentPrice=raw["currentPrice"] * 1.1)
    third = save_raw_archive(changed, cfg.snapshot_dir, days[2], benchmark=[])
    assert third.stat().st_ino != second.stat().st_ino
    # Rewriting a linked day with new data replaces its own entry and never alters the day it was linked to.
    save_raw_archive(changed, cfg.snapshot_dir, days[1], benchmark=[])
    pd.testing.assert_frame_equal(pd.read_parquet(first).drop(columns="In_Benchmark"), raw)
    assert not list((cfg.snapshot_dir / "raw").glob("*.tmp"))


def test_clean_cache_keeps_what_cannot_be_downloaded_again(tmp_path):
    from fq.pipeline import clean

    data, out = tmp_path / "data", tmp_path / "output"
    for d in ("forward", "forward_backup_x", "snapshots/raw", "evaluation"):
        (data / d).mkdir(parents=True)
    for f in ("grades.parquet", "grade_gate_v1_backup.parquet", "raw.parquet", "ranked_benchmark.parquet", "fx.json"):
        (data / f).write_text("x")
    (out / "benchmark").mkdir(parents=True)
    removed, kept = clean(out, data, cache=False)
    assert removed == [out] and kept == [] and (data / "raw.parquet").exists()
    removed, kept = clean(out, data, cache=True)
    assert {p.name for p in data.iterdir()} == {
        "forward", "forward_backup_x", "snapshots", "grades.parquet", "grade_gate_v1_backup.parquet"
    }  # fmt: skip
    assert {p.name for p in removed} == {"evaluation", "raw.parquet", "ranked_benchmark.parquet", "fx.json"}


def test_published_tabs_are_followed_when_their_method_matches(cfg):
    from fq.evaluate import load_published, save_published

    snap = _snap().assign(Method="m1")
    tickers = list(snap["Ticker"])
    prices = _prices(tickers, "2024-01-01", np.linspace(30, -20, 50))
    day = pd.Timestamp("2024-01-02")
    tabs = pd.DataFrame({"Ticker": ["T0", "T1", "T49"], "List": ["Net cash", "Net cash", "Cannibals"]})
    save_published(tabs, cfg.snapshot_dir, "test", day + pd.Timedelta(hours=9), method="m1")
    published = load_published(cfg.snapshot_dir, "test")
    assert list(published) == [day]
    lists, _ = evaluate({day: snap}, prices, horizons=(12,), top=10, published=published)
    by = lists.set_index("List")
    assert by.loc["Tab: Net cash", "Avg_N"] == 2 and by.loc["Tab: Net cash", "Excess"] > 0
    assert by.loc["Tab: Cannibals", "Excess"] < 0
    other = evaluate({day: snap.assign(Method="m2")}, prices, horizons=(12,), top=10, published=published)[0]
    assert not other["List"].str.startswith("Tab: ").any()  # a rescored snapshot is not what the site showed
