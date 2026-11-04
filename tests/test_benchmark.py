"""Scoring a group against a benchmark (S&P 500) while ranking and reporting only the group."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking, criterion_percentile
from fq.report import index_charts as charts
from fq.report.format import benchmark_note, summary_columns
from fq.report.html import build_site
from fq.report.links import slug
from fq.report.tiles import HEADLINE_FACTS


@pytest.mark.parametrize("direction", ["top", "bottom"])
def test_reference_member_scores_as_in_the_reference_alone(direction):
    rng = np.random.default_rng(1)
    ref = pd.Series(np.r_[rng.normal(10, 5, 60), [10.0, 10.0, 500.0]])  # ties and an outlier
    alone = criterion_percentile(ref, direction, 0.01)
    placed = criterion_percentile(ref.iloc[[0, 60, 62]], direction, 0.01, reference=ref)
    assert np.allclose(placed.to_numpy(), alone.iloc[[0, 60, 62]].to_numpy())


def test_outsider_is_placed_in_the_reference_distribution():
    ref = pd.Series(np.arange(1.0, 101.0))
    pct = criterion_percentile(pd.Series([1000.0, -5.0, 50.5, np.nan]), "top", 0.01, reference=ref)
    assert pct[0] == pytest.approx(criterion_percentile(ref, "top", 0.01).max())  # ties with the winsorized top
    assert pct[1] < 0.02 and pct[2] == pytest.approx(0.505, abs=1e-9) and pd.isna(pct[3])


def _bench_and_group(cfg, n=30, group=(0, 3, 7)):
    metrics = calculate_metrics(make_raw(cfg, n=n, seed=3), cfg)
    bench = apply_dual_ranking(metrics, cfg)
    return metrics, bench, metrics.iloc[list(group)].reset_index(drop=True)


def test_group_scores_match_the_benchmark_and_ranks_stay_in_the_group(cfg):
    _, bench, group = _bench_and_group(cfg)
    ranked = apply_dual_ranking(group, cfg, reference=bench, benchmark="sp500").set_index("Ticker")
    ref = bench.set_index("Ticker")
    for t in ranked.index:
        assert ranked.at[t, "Value_Score"] == pytest.approx(ref.at[t, "Value_Score"])
        assert ranked.at[t, "Quality_Score"] == pytest.approx(ref.at[t, "Quality_Score"])
        assert ranked.at[t, "Value_Rank_Bench"] == ref.at[t, "Value_Rank"]
    assert sorted(ranked["Rank"]) == [1, 2, 3] and sorted(ranked["Value_Rank"].dropna()) == [1, 2, 3]
    assert (ranked["Benchmark"] == "sp500").all() and (
        ranked["Benchmark_N"] == bench["Value_Score"].notna().sum()
    ).all()
    for column, *_ in (*cfg.value_criteria, *cfg.quality_criteria):
        assert ranked[f"Pct_{column}"].dropna().between(0, 100).all()  # NaN where the value is missing


def test_group_is_ordered_as_in_the_benchmark(cfg):
    _, bench, group = _bench_and_group(cfg, group=(1, 4, 9, 12, 20))
    ranked = apply_dual_ranking(group, cfg, reference=bench, benchmark="sp500")
    in_bench = bench[bench["Ticker"].isin(group["Ticker"])].sort_values("Rank")["Ticker"].tolist()
    assert ranked.sort_values("Rank")["Ticker"].tolist() == in_bench


def test_report_shows_only_the_group_with_benchmark_context(cfg):
    _, bench, group = _bench_and_group(cfg)
    ranked = apply_dual_ranking(group, cfg, reference=bench, benchmark="sp500")
    assert "S&P 500 benchmark (30 companies)" in benchmark_note(ranked)
    assert "Value_Rank_Bench" in [c[1] for c in summary_columns(ranked)]
    fig, meta = charts.quadrant_figure(ranked, bench=bench)
    cloud = fig.data[0]
    assert cloud.name == "Benchmark" and cloud.customdata is None and len(cloud.x) == 30 - 3
    assert all(fig.data[m["trace"]].customdata is not None for m in meta)  # toggles and clicks skip the cloud

    index = build_site(ranked, cfg, bench=bench)
    pages = list((index.parent / "companies").glob("*.html"))
    assert sorted(p.stem for p in pages) == sorted(slug(t) for t in group["Ticker"])
    page = (index.parent / "companies" / f"{slug(group.loc[0, 'Ticker'])}.html").read_text(encoding="utf-8")
    assert 'id="score_breakdown"' not in page
    headline = page.split('class="headline"')[1].split("</ul>")[0]
    known = [c for _, c, *_ in HEADLINE_FACTS if pd.notna(pd.to_numeric(group.loc[0].get(c), errors="coerce"))]
    assert headline.count('class="chip') >= len(known) and " B$" in headline
    assert "N/A" not in headline  # a missing figure is left out, not shown as N/A
    assert "<dt>Analysts</dt>" not in page  # in the headline, not repeated in the key figures

    text = index.read_text(encoding="utf-8")
    order = [text.index(s) for s in ("Turning over the Rocks", "<h3>Cheap and good</h3>", "<h3>Operating companies</h3>", "<h3>Cheapest</h3>", "The map: Value vs Quality",
                                     "<h2>Full ranking</h2>")]  # fmt: skip
    assert order == sorted(order)


def test_quadrant_sp500_colors_and_area_sizing():
    df = pd.DataFrame(
        [
            {
                "Ticker": "AAPL",  # S&P 500 member
                "shortName": "Apple Inc.",
                "country": "United States",
                "Value_Score": 60.0,
                "Quality_Score": 85.0,
                "Market Cap": 3000.0,
                "FCFy_Norm": 4.5,
                "ROIC_Norm": 25.0,
            },
            {
                "Ticker": "SMALL.MC",  # Rest of universe (Spanish small cap)
                "shortName": "Small Spanish Corp",
                "country": "Spain",
                "Value_Score": 75.0,
                "Quality_Score": 55.0,
                "Market Cap": 0.10,
                "FCFy_Norm": 8.0,
                "ROIC_Norm": 12.0,
            },
        ]
    )
    fig, meta = charts.quadrant_figure(df)
    assert fig is not None
    trace_names = [t.name for t in fig.data if t.name in ("S&P 500", "Rest of universe")]
    assert "S&P 500" in trace_names
    assert "Rest of universe" in trace_names

    sp_trace = next(t for t in fig.data if t.name == "S&P 500")
    rest_trace = next(t for t in fig.data if t.name == "Rest of universe")

    assert sp_trace.marker.color == charts.SP500_COLOR
    assert rest_trace.marker.color == charts.REST_COLOR
    assert float(sp_trace.marker.size[0]) > float(rest_trace.marker.size[0])
    assert float(sp_trace.marker.size[0]) == charts.MAX_DOT_PX  # AAPL is the largest company
    assert float(rest_trace.marker.size[0]) == charts.MIN_DOT_PX  # SMALL.MC hits the floor
    assert sp_trace.hoverinfo == rest_trace.hoverinfo == "none"  # static dots, no tooltip
    assert charts.dot_diameter(100.0, 400.0) == charts.dot_diameter(400.0, 400.0) / 2  # area ~ cap


def test_rank_falls_back_when_benchmark_is_not_cached(cfg, monkeypatch):
    from fq import fetch, pipeline
    from fq.universe import Selection

    raw = make_raw(cfg, n=6)
    fetch.save_raw(raw, cfg.raw_path)
    cfg.bench_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.bench_path.write_bytes(b"stale")
    monkeypatch.setattr(pipeline, "portfolio", lambda: frozenset())
    group = Selection("custom", list(raw["Ticker"][:3]))
    bench = Selection("sp500", [*raw["Ticker"], *[f"X{i}" for i in range(20)]])  # mostly uncached
    pipeline.rank_run(group, cfg, bench)
    ranked = pd.read_parquet(cfg.ranked_path)
    assert "Benchmark" not in ranked.columns and not cfg.bench_path.exists()

    pipeline.rank_run(group, cfg, Selection("sp500", list(raw["Ticker"])))  # fully cached
    ranked = pd.read_parquet(cfg.ranked_path)
    assert len(ranked) == 3 and (ranked["Benchmark"] == "sp500").all() and len(pd.read_parquet(cfg.bench_path)) == 6


def test_default_benchmark(monkeypatch):
    from fq import pipeline
    from fq.universe import Selection

    monkeypatch.setattr(pipeline, "resolve_selection", lambda token: Selection(token, ["A", "B", "C"]))
    assert pipeline.benchmark_selection(Selection("portfolio", ["A"]), None).label == "benchmark"
    assert pipeline.benchmark_selection(Selection("custom", ["A"]), None).label == "benchmark"
    assert pipeline.benchmark_selection(Selection("benchmark", ["A", "B", "C"]), None) is None
    assert pipeline.benchmark_selection(Selection("portfolio", ["A"]), "none") is None
    assert (
        pipeline.benchmark_selection(Selection("custom", ["A", "B", "C", "D"]), "sp500") is None
    )  # benchmark inside the run


def test_sector_blend_keeps_benchmark_invariance(cfg):
    metrics = calculate_metrics(make_raw(cfg, n=40, seed=5), cfg)
    metrics["sector"] = ["Technology"] * 20 + ["Energy"] * 16 + ["Utilities"] * 4  # Utilities too small
    bench = apply_dual_ranking(metrics, cfg)
    group = metrics.iloc[[0, 25, 38]].reset_index(drop=True)
    ranked = apply_dual_ranking(group, cfg, reference=bench, benchmark="sp500").set_index("Ticker")
    ref = bench.set_index("Ticker")
    for t in ranked.index:
        assert ranked.at[t, "Value_Score"] == pytest.approx(ref.at[t, "Value_Score"])

    market = apply_dual_ranking(metrics, replace(cfg, sector_blend=0.0)).set_index("Ticker")
    small = metrics.loc[36, "Ticker"]  # Utilities (4 < 15): market percentile only
    assert ref.at[small, "Value_Score"] == pytest.approx(market.at[small, "Value_Score"])
    assert ref["Quality_Score"].equals(market.loc[ref.index, "Quality_Score"])  # Quality is market-wide


def test_cheapest_in_a_dear_sector_is_lifted(cfg):
    from fq.ranking import sector_percentile

    values = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    sectors = pd.Series(["A", "A", "A", "B", "B", "B"])
    within = sector_percentile(values, "top", 0.0, sectors, values, sectors, min_members=3)
    market = criterion_percentile(values, "top", 0.0)
    assert within[2] == within[5] == pytest.approx(1.0) and market[2] == pytest.approx(0.5)
    assert sector_percentile(values, "top", 0.0, sectors, values, sectors, min_members=4).isna().all()


def test_histograms_bin_only_the_axis_and_count_every_company():
    """One FCF yield of -262,119% made Plotly draw 131,449 bins (Oct 31, 2026): only the axis is binned, the title
    still counts every company and says how many lie beyond it."""
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(5)], "FCFy_Norm": [3.0, 5.0, 7.0, 49.9, -262119.6]})
    (title, div), *_ = charts.histogram_divs(df)
    assert "-262119" not in div and '"start":-10' in div and '"end":50' in div and "49.9" in div
    assert "n = 5" in div and "1 beyond the axis" in div
    assert "bdata" not in div  # values as short numbers, not 8-byte binary
