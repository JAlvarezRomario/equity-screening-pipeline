"""The run, step by step, as plain functions (no command line here): fetch, rank, report, grade, holders, forward
test, evaluation. `fq.cli` parses options and calls these; a script, a notebook or a test can call them directly.

Functions take a `Settings` (and a `Selection` where they work on a run's tickers) and read and write the cache under
`cfg.data_dir`. A problem the user can fix (a missing cache, a bad set name) raises `PipelineError`.
"""

import fnmatch
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd

from fq import evaluate as evaluate_mod
from fq import fetch as fetch_mod
from fq import forward, fx, grading, health, history, refresh
from fq.filters import clean_and_filter
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report.forward_page import build_forward, build_placeholder
from fq.report.highlights import Highlights
from fq.report.html import build_site
from fq.settings import BENCHMARK_MIN_CACHED, DEFAULT_BENCHMARK, SELF_BENCHMARKED, Settings
from fq.universe import Selection, resolve_selection

log = logging.getLogger("fq")
Mode = Literal["auto", "full", "prices", "new"]


class PipelineError(Exception):
    """Something the user can fix (a missing cache or a set that does not exist); the command line shows it."""


# What `clean --cache` never deletes (Nov 3, 2026): records that cannot be downloaded again. The forward test is
# dated as it happened, the grades cost real money, and the score snapshots and raw archives are what `fq
# evaluate` and `--rescore` follow forward. Delete them by hand if ever needed.
PROTECTED_DATA = ("forward", "forward_backup_*", "snapshots", "grades*.parquet", "grade_gate*.parquet")


def clean(output_dir: Path, data_dir: Path, cache: bool = False) -> tuple[list[Path], list[Path]]:
    """Delete the generated site (`output_dir`) and, with `cache`, everything in `data_dir` that can be rebuilt
    (downloads, rankings, rates), keeping `PROTECTED_DATA`. Returns (removed, kept)."""
    removed, kept = [], []

    def remove(path: Path) -> None:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        removed.append(path)

    if output_dir.exists():
        remove(output_dir)
    if cache and data_dir.exists():
        for path in sorted(data_dir.iterdir()):
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in PROTECTED_DATA):
                kept.append(path)
            else:
                remove(path)
    return removed, kept


def portfolio() -> frozenset[str]:
    """Holdings are never dropped for their industry; empty when portfolio.txt does not exist."""
    try:
        return frozenset(resolve_selection("portfolio").tickers)
    except ValueError:
        return frozenset()


def benchmark_selection(sel: Selection, token: str | None) -> Selection | None:
    """The set the run is scored against, or None to rank the run on its own."""
    if token is None:
        token = "none" if sel.label in SELF_BENCHMARKED else DEFAULT_BENCHMARK
    if token.strip().lower() == "none":
        return None
    try:
        bench = resolve_selection(token)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    return None if set(bench.tickers) <= set(sel.tickers) else bench


def fetch_data(sel: Selection, cfg: Settings, mode: Mode = "auto", bench: Selection | None = None) -> None:
    """Fetch fundamentals for the tickers that need them and refresh prices of the rest.

    `full`: every ticker; `new`: only uncached tickers, no prices; `prices`: uncached tickers plus
    prices; `auto`: uncached tickers and those `refresh.plan` finds due (new report, new fields, old,
    incomplete), the benchmark's included, within `cfg.max_full_fetches` by priority (`refresh.select`).
    Cached run and `bench` tickers not fetched get new prices (auto / prices), once a week at most.
    """
    existing = fetch_mod.load_raw(cfg.raw_path) if cfg.raw_path.exists() else None
    if existing is not None:
        try:
            health.check_schema(existing)
        except health.CacheError as exc:
            log.warning("Discarding old cache: %s", exc)
            existing = None
    cached = set(existing["Ticker"]) if existing is not None else set()
    run = set(sel.tickers)
    extra = [t for t in (bench.tickers if bench else []) if t not in run]
    if mode == "full":
        todo = list(sel.tickers)
    elif mode == "auto":
        now = pd.Timestamp.now()
        run_due = refresh.plan(existing, sel.tickers, cfg, now)
        bench_due = refresh.plan(existing, extra, cfg, now) if extra else {}
        todo, deferred = refresh.select(run_due, bench_due, cfg.max_full_fetches)
        due = {**bench_due, **run_due}
        counts = Counter(due[t] for t in todo)
        detail = ", ".join(f"{counts[r]} {r}" for r in refresh.REASONS if counts[r])
        n_bench = sum(t in bench_due for t in todo)
        log.info("Auto refresh: %d full%s%s", len(todo), f" ({detail})" if detail else "",
                 f", {n_bench} of them for the {bench.label} benchmark" if n_bench else "")  # fmt: skip
        if deferred:
            log.info("%d due companies wait for a later run (max %d full fetches per run; `--full` fetches all)",
                     len(deferred), cfg.max_full_fetches)  # fmt: skip
        for why in refresh.REASONS:
            if counts[why]:
                log.debug("  %s: %s", why, ", ".join(t for t in todo if due[t] == why))
    else:
        todo = [t for t in sel.tickers if t not in cached]
        log.info("%d new tickers to fetch (%d cached)", len(todo), len(sel.tickers) - len(todo))
    chunk_size = 100
    df = existing
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i : i + chunk_size]
        df = fetch_mod.fetch_universe(chunk, cfg, df)
        fetch_mod.save_raw(df, cfg.raw_path)
    if df is None:
        df = pd.DataFrame()
    if mode in ("auto", "prices"):
        fetched = set(todo)
        stale = [t for t in [*sel.tickers, *extra] if t in cached and t not in fetched]
        df = fetch_mod.refresh_prices(df, list(dict.fromkeys(stale)), cfg)
        fetch_mod.save_raw(df, cfg.raw_path)
    log.info("Cache holds %d companies (%s)", len(df), cfg.raw_path)


def rank_benchmark(
    bench: Selection | None, raw: pd.DataFrame, metrics: pd.DataFrame, cfg: Settings
) -> pd.DataFrame | None:
    """The benchmark ranked on its own, or None if unusable. Filtered exactly as its own run (portfolio
    holdings kept), so a company scores the same here as in the benchmark's report."""
    if bench is None:
        return None
    present = [t for t in bench.tickers if t in set(metrics["Ticker"])]
    if len(present) < BENCHMARK_MIN_CACHED * len(bench.tickers):
        log.warning("Benchmark '%s': only %d of %d tickers cached; ranking the run on its own "
                    "(run `fq run -u %s` once to use it)", bench.label, len(present), len(bench.tickers),
                    bench.label)  # fmt: skip
        return None
    age = health.cache_age_days(raw[raw["Ticker"].isin(present)])
    if age is not None and age > cfg.max_cache_age_days:
        log.warning("Benchmark '%s' fundamentals are up to %d days old; `fq run -u %s` refreshes them",
                    bench.label, age, bench.label)  # fmt: skip
    kept, _ = clean_and_filter(metrics, present, protected=portfolio(), verbose=False)
    ranked = apply_dual_ranking(kept, cfg)
    log.info("Scoring against benchmark '%s' (%d companies)", bench.label, int(ranked["Value_Score"].notna().sum()))
    return ranked


def rank_run(sel: Selection, cfg: Settings, bench: Selection | None = None) -> None:
    if not cfg.raw_path.exists():
        raise PipelineError(f"{cfg.raw_path} not found; run `fq fetch` first.")
    raw = fetch_mod.load_raw(cfg.raw_path)
    try:
        health.check_schema(raw)
    except health.CacheError as exc:
        raise PipelineError(str(exc)) from exc
    health.report(raw, cfg)
    metrics = calculate_metrics(raw, cfg)
    kept, removed = clean_and_filter(metrics, sel.tickers, protected=portfolio())
    missing = sorted(set(sel.tickers) - set(metrics["Ticker"]))
    if missing:
        log.warning("Not fetched / no data: %s", ", ".join(missing))
    reference = rank_benchmark(bench, raw, metrics, cfg)
    ranked = apply_dual_ranking(kept, cfg, reference, bench.label if reference is not None else None)
    fetch_mod.save_raw(ranked, cfg.ranked_path)
    now = pd.Timestamp.now()
    snap = evaluate_mod.save_snapshot(ranked, cfg.snapshot_dir, cfg.label, now)
    archive = evaluate_mod.save_raw_archive(raw, cfg.snapshot_dir, now, resolve_selection(DEFAULT_BENCHMARK).tickers)
    log.debug("Snapshot saved to %s (method %s), raw cache archived to %s", snap,
              evaluate_mod.method_fingerprint(), archive)  # fmt: skip
    if reference is not None:
        fetch_mod.save_raw(reference, cfg.bench_path)
    else:
        cfg.bench_path.unlink(missing_ok=True)  # never draw a cloud left over from an earlier run
    requested = set(sel.tickers)
    dropped = [t for t, _ in removed if t in requested]
    log.info("Ranked %d of %d requested tickers (%d filtered out)", len(ranked), len(sel.tickers), len(dropped))


def gate_grades(cfg: Settings) -> pd.DataFrame:
    """Today's grades with the gate's verdict, for the forward test's frozen gate."""
    grades = grading.load(cfg.data_dir / "grades.parquet")
    return grades.assign(
        Passes=[grading.passes(m, g) for m, g in zip(grades["Moat"], grades["Management"], strict=True)]
    )


def forward_step(cfg: Settings, ranked: pd.DataFrame, h: Highlights, offline: bool = False):
    """The backward and forward test (`fq.forward`) of the benchmark run: (cohort, ledger, today's lists), or None for
    other runs, when nothing is stored offline, or when it fails (the report goes on without it).

    The cohort is frozen only from the lists published on `forward.FROZEN_ON` (the last report of that day): before
    they exist, nothing is frozen and None is returned (the site shows a placeholder)."""
    if cfg.label != DEFAULT_BENCHMARK:
        return None
    try:
        live = h.published()
        published, published_on = live, None
        saved = {d: p for d, p in evaluate_mod.load_published(cfg.snapshot_dir, cfg.label).items()
                 if d.normalize() == forward.FROZEN_ON}  # fmt: skip
        if saved:
            published_on = max(saved)
            published = saved[published_on][["Ticker", "List"]]
            published_on = published_on.normalize()
        elif not (cfg.data_dir / "forward" / "cohort.parquet").exists():
            return None  # never frozen from any other day's lists
        tabs = evaluate_mod.load_published(cfg.snapshot_dir, cfg.label)
        late = {name: frame for name, when in forward.LATE_LISTS.items() for d, frame in tabs.items()
                if d.normalize() == when and name in set(frame["List"])}  # fmt: skip
        result = forward.step(cfg.data_dir / "forward", published, ranked, portfolio(), pd.Timestamp.now(),
                              fetch_mod.total_and_price_history, fx._usd_history, method=evaluate_mod.method_fingerprint(),
                              published_on=published_on, offline=offline, grades=gate_grades(cfg), late=late)  # fmt: skip
    except Exception as exc:  # never lose the report to the forward test
        log.warning("Forward test skipped: %s", exc, exc_info=True)
        return None
    return None if result is None else (*result, live)


def grade_companies(
    cfg: Settings,
    ranked: pd.DataFrame,
    h: Highlights,
    max_grades: int = 0,
    dry_run: bool = False,
    resynthesize_only: bool = False,
    new_only: bool = False,
) -> int:
    """Grade the companies the highlight lists show and every Repeaters candidate (`Highlights.grade_pool`), and
    the holdings whose grade is missing or due
    (`grading.due`), at most `max_grades` (0 = no cap), from their rows (`grading.grade_input`, no price); then rewrite
    every synthesis written with an older synthesis prompt (`grading.resynthesize`, the cheap model only).
    `resynthesize_only` skips the grading; `new_only` grades only companies never graded and rewrites no synthesis
    (no re-grade is spent). Returns how many grades or syntheses changed; never fails the report (an error is logged)."""
    path = cfg.data_dir / "grades.parquet"
    now = pd.Timestamp.now()
    held = [t for t in sorted(portfolio()) if t in set(ranked["Ticker"])]  # every holding is graded too
    candidates = list(dict.fromkeys([*h.grade_pool(), *held]))
    stored = grading.load(path)
    todo = [] if resynthesize_only else grading.due(candidates, ranked, stored, now)
    if new_only:
        known = set(stored["Ticker"]) if not stored.empty else set()
        todo = [t for t in todo if t not in known]
    if max_grades:
        todo = todo[:max_grades]
    log.info("Moat and management grades: %d listed or held, %d due, est. $%.2f", len(candidates), len(todo),
             grading.estimate_usd(len(todo), now))  # fmt: skip
    changed = 0
    if todo and not dry_run:
        try:
            new = grading.run(todo, ranked, path)
            log.info("Graded %d companies for $%.2f", len(new), new["Cost_USD"].sum() if len(new) else 0.0)
            changed += len(new)
        except Exception as exc:  # never lose the report to the grading
            log.warning("Grading skipped: %s", exc)
    stale = [] if new_only else grading.synthesis_due(grading.load(path))
    log.info("Syntheses written with an older prompt: %d (synthesis model only, about $0.01-0.02 each)", len(stale))
    if stale and not dry_run:
        try:
            changed += grading.resynthesize(stale, ranked, path)
        except Exception as exc:
            log.warning("Synthesis rewrite skipped: %s", exc)
    return changed


def write_report(
    cfg: Settings,
    open_after: bool = False,
    index_only: bool = False,
    companies_only: bool = False,
    tickers: set[str] | None = None,
    offline: bool = False,
) -> None:
    if not cfg.ranked_path.exists():
        raise PipelineError(f"{cfg.ranked_path} not found; run `fq rank` first.")
    ranked = pd.read_parquet(cfg.ranked_path)
    bench = pd.read_parquet(cfg.bench_path) if cfg.bench_path.exists() else None
    h = Highlights.compute(ranked, cfg, bench)  # every list once: the page, the forward test, the grading
    fw = forward_step(cfg, ranked, h, offline) if not companies_only and tickers is None else None
    soon = fw is None and cfg.label == DEFAULT_BENCHMARK and not companies_only and tickers is None
    forward_link = "Forward test" if fw is not None or soon else None
    index = build_site(
        ranked,
        cfg,
        bench=bench,
        held=portfolio(),
        index_only=index_only,
        companies_only=companies_only,
        tickers=tickers,
        forward=forward_link,
        highlights=h,
    )
    # The benchmark's listed companies are graded from the pages just written; a new grade re-renders its page and
    # the index (unchanged pages are skipped by their fingerprint).
    grade = (cfg.label == DEFAULT_BENCHMARK and not offline and not companies_only and tickers is None
             and bool(os.environ.get("GEMINI_API_KEY")))  # fmt: skip
    if grade and grade_companies(cfg, ranked, h, max_grades=grading.MAX_PER_RUN):
        h = Highlights.compute(ranked, cfg, bench)  # the new grades move rows below the bar and into Repeaters
        index = build_site(ranked, cfg, bench=bench, held=portfolio(),
                           forward=forward_link, highlights=h)  # fmt: skip
    if fw is not None:
        cohort, ledger, live = fw
        log.info("Forward test written to %s", build_forward(cfg.site_dir, cohort, ledger, live=live))
    elif soon:
        log.info("Forward test not started: placeholder written to %s", build_placeholder(cfg.site_dir))
    if not companies_only and tickers is None:  # the tabs as shown, for `fq evaluate`
        tabs = evaluate_mod.save_published(h.published(), cfg.snapshot_dir, cfg.label,
                                           pd.Timestamp.now())  # fmt: skip
        log.debug("Published lists saved to %s", tabs)
    print(f"\nOpen {index.resolve()}")
    if open_after and hasattr(os, "startfile"):
        os.startfile(index.resolve())  # noqa: S606 - Windows: open in the default browser


HOLDERS_FRESH_DAYS = 30  # `fq holders` skips companies whose holders are younger than this (a stopped run resumes)


def holder_targets(sel: Selection, cfg: Settings, raw: pd.DataFrame, every: bool) -> list[str]:
    """The companies `fq holders` fetches: every cached ticker of the run; for the benchmark (unless `every`) only those
    the site's lists can show, the best fifth of the combined ranking (`screens.in_best_fraction`), the latest
    published tabs (Cheapest and Best businesses are not cut to it) and the holdings."""
    cached = set(raw["Ticker"])
    tickers = [t for t in sel.tickers if t in cached]
    if every or sel.label != DEFAULT_BENCHMARK or not cfg.ranked_path.exists():
        return tickers
    from fq.report.screens import in_best_fraction

    ranked = pd.read_parquet(cfg.ranked_path, columns=["Ticker", "Rank"])
    wanted = set(ranked.loc[in_best_fraction(ranked), "Ticker"]) | set(portfolio())
    tabs = evaluate_mod.load_published(cfg.snapshot_dir, cfg.label)
    if tabs:
        wanted |= set(tabs[max(tabs)]["Ticker"])
    return [t for t in tickers if t in wanted]


def rescore(cfg: Settings, snapshots: dict[pd.Timestamp, pd.DataFrame], until: pd.Timestamp) -> dict:
    """Today's method on every raw cache archived up to `until`: {date: snapshot}.

    The companies are those of the stored snapshot of that date (the benchmark run: that day's members, also
    without one), scored against the benchmark it used (`Benchmark`; older snapshots: the default). Archives
    from another cache schema cannot be read by today's code and are skipped.
    """
    method = evaluate_mod.method_fingerprint()
    out = {}
    for day, path in evaluate_mod.load_raw_archives(cfg.snapshot_dir).items():
        stored = snapshots.get(day)
        if day > until or (stored is None and cfg.label != DEFAULT_BENCHMARK):
            continue
        raw = fetch_mod.load_raw(path)
        try:
            health.check_schema(raw)
        except health.CacheError as exc:
            log.warning("Archive %s skipped: %s", path.name, exc)
            continue
        members = raw.loc[raw.pop("In_Benchmark").astype(bool), "Ticker"].tolist()
        tickers = list(stored["Ticker"]) if stored is not None else members
        token = None if cfg.label in SELF_BENCHMARKED else DEFAULT_BENCHMARK
        if stored is not None and "Benchmark" in stored.columns:
            token = stored["Benchmark"].dropna().iloc[0] if stored["Benchmark"].notna().any() else None
        bench = None
        if token is not None:
            bench_tickers = members if token == DEFAULT_BENCHMARK else resolve_selection(token).tickers
            bench = Selection(token, bench_tickers) if bench_tickers else None
        metrics = calculate_metrics(raw, cfg)
        kept, _ = clean_and_filter(metrics, tickers, protected=portfolio(), verbose=False)
        reference = rank_benchmark(bench, raw, metrics, cfg)
        ranked = apply_dual_ranking(kept, cfg, reference, token if reference is not None else None)
        out[day] = evaluate_mod.snapshot(ranked, method)
        log.info("Rescored %s with method %s (%d companies)", f"{day:%Y-%m-%d}", method, len(ranked))
    return out


def quality_history(cfg: Settings) -> pd.DataFrame | None:
    """Did Quality foresee what happened to the fundamentals? The benchmark scored as it stood `history.LAGS` years
    ago (filtered as its own run) against what its statements did since; saved to `cfg.history_path`."""
    if not cfg.raw_path.exists():
        return None
    raw = fetch_mod.load_raw(cfg.raw_path)
    try:
        health.check_schema(raw)
    except health.CacheError as exc:
        log.warning("Quality history skipped: %s", exc)
        return None
    tickers = [t for t in resolve_selection(DEFAULT_BENCHMARK).tickers if t in set(raw["Ticker"])]
    raw = raw[raw["Ticker"].isin(tickers)].reset_index(drop=True)
    now = calculate_metrics(raw, cfg)
    tables = []
    for lag in history.LAGS:
        lagged = calculate_metrics(history.lag_raw(raw, lag, cfg), cfg)
        kept, _ = clean_and_filter(lagged, tickers, protected=portfolio(), verbose=False)
        ranked = apply_dual_ranking(kept, cfg)
        tables.append(history.quality_ic(ranked, history.outcomes(now, lagged, lag, cfg), lag, cfg))
    out = pd.concat(tables, ignore_index=True)
    out.to_parquet(cfg.history_path, index=False)
    wide = out.pivot_table(index="Measure", columns=["Outcome", "Lag"], values="IC", sort=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.2f}".format):
        print(f"\nQuality scored {', '.join(map(str, history.LAGS))} years back vs what its statements did "
                   f"since (rank correlation, {DEFAULT_BENCHMARK}, N ~{int(out['N'].median())}):\n{wide}")  # fmt: skip
    return out


def evaluate_run(cfg: Settings, rescore: bool = False) -> None:
    snapshots = evaluate_mod.load_snapshots(cfg.snapshot_dir, cfg.label)
    oldest_due = pd.Timestamp.now() - pd.DateOffset(months=min(evaluate_mod.HORIZONS_MONTHS))
    if rescore:  # today's method replaces the stored scores on every archived date
        snapshots = dict(sorted((snapshots | rescore(cfg, snapshots, oldest_due)).items()))
    quality_history(cfg)  # instant: on the history in the cache, no snapshots or prices needed
    due = [d for d in snapshots if d + pd.DateOffset(months=min(evaluate_mod.HORIZONS_MONTHS)) <= pd.Timestamp.now()]
    if not due:
        first = f"; the first was taken {min(snapshots):%Y-%m-%d}" if snapshots else ""
        log.info("No '%s' snapshot is %d months old yet%s. Every `fq rank` saves one.", cfg.label,
                 min(evaluate_mod.HORIZONS_MONTHS), first)  # fmt: skip
        return
    tickers = sorted({t for d in due for t in snapshots[d]["Ticker"]})
    log.info("Evaluating %d snapshots (%d tickers)", len(due), len(tickers))
    # The market (`evaluate.MARKET`, USD) comes in the same batch.
    prices = fetch_mod.total_return_history([*tickers, evaluate_mod.MARKET], min(due) - pd.Timedelta(days=7))
    # Quote currency per ticker: the snapshots keep it; older ones fall back to the shared cache.
    currencies = {}
    if cfg.raw_path.exists():
        raw = fetch_mod.load_raw(cfg.raw_path)
        if "currency" in raw.columns:
            currencies = dict(zip(raw["Ticker"], raw["currency"], strict=True))
    for d in due:
        if "currency" in snapshots[d].columns:
            currencies |= dict(zip(snapshots[d]["Ticker"], snapshots[d]["currency"], strict=True))
    needed = {"GBP" if c == "GBp" else c for t, c in currencies.items() if t in tickers and c and c != "USD"}
    prices = evaluate_mod.to_usd(prices, currencies, {c: fx._usd_history(c) for c in needed})
    published = evaluate_mod.load_published(cfg.snapshot_dir, cfg.label)
    lists, ic = evaluate_mod.evaluate({d: snapshots[d] for d in due}, prices, published=published)
    lists_path, ic_path = cfg.evaluation_paths
    lists.to_parquet(lists_path, index=False)
    ic.to_parquet(ic_path, index=False)
    stable = evaluate_mod.stability(snapshots)
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(f"\nForward returns vs the median company and vs {evaluate_mod.MARKET} (%, USD, dividends "
                   f"included):\n{lists}")  # fmt: skip
        print(f"\nInformation coefficient (rank correlation with the forward return; IC_t Newey-West):\n{ic}")
        print(f"\nRank stability between consecutive snapshots (Spearman):\n{stable}")
    for r in stable[stable["Rank"] < evaluate_mod.STABLE_RANKS].itertuples():
        log.warning("Ranks of %s moved a lot since %s (Spearman %.2f, %d companies): a data break?",
                    f"{r.Snapshot:%Y-%m-%d}", f"{r.Previous:%Y-%m-%d}", r.Rank, r.N)  # fmt: skip


def load_benchmark_run(cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The benchmark run as `rank_run` last saved it: (ranked, the benchmark it was scored against or None)."""
    if not cfg.ranked_path.exists():
        raise PipelineError(f"{cfg.ranked_path} not found; run `fq run -u benchmark` first.")
    ranked = pd.read_parquet(cfg.ranked_path)
    bench = pd.read_parquet(cfg.bench_path) if cfg.bench_path.exists() else None
    return ranked, bench


def grade_only(cfg: Settings, max_grades: int = 0, dry_run: bool = False, resynthesize_only: bool = False,
               report: bool = True, new_only: bool = False) -> None:  # fmt: skip
    """`fq grade`: grades (or syntheses) alone, then the site again from the cache when something changed."""
    ranked, bench = load_benchmark_run(cfg)
    h = Highlights.compute(ranked, cfg, bench)
    if grade_companies(cfg, ranked, h, max_grades=max_grades, dry_run=dry_run, resynthesize_only=resynthesize_only,
                       new_only=new_only) and report:  # fmt: skip
        write_report(cfg, offline=True)


def forward_only(cfg: Settings, offline: bool = False) -> None:
    """`fq forward`: the backward and forward test alone (freeze once, add the month-ends passed, write forward.html)."""
    ranked, bench = load_benchmark_run(cfg)
    fw = forward_step(cfg, ranked, Highlights.compute(ranked, cfg, bench), offline)
    if fw is None:
        print(f"Not started yet: placeholder written to {build_placeholder(cfg.site_dir).resolve()}")
        return
    cohort, ledger, live = fw
    print(f"\nOpen {build_forward(cfg.site_dir, cohort, ledger, live=live).resolve()}")


def fetch_holders_run(sel: Selection, cfg: Settings, every: bool = False, max_n: int = 0, force: bool = False) -> int:
    """`fq holders`: the largest holders alone into the cache (`holder_targets`, none younger than
    `HOLDERS_FRESH_DAYS` unless `force`, at most `max_n` when above 0). Returns how many companies were fetched."""
    if not cfg.raw_path.exists():
        raise PipelineError(f"{cfg.raw_path} not found; run `fq fetch` first.")
    raw = fetch_mod.load_raw(cfg.raw_path)
    todo = holder_targets(sel, cfg, raw, every)
    if not force and "Holders_At" in raw.columns:
        at = pd.to_datetime(raw.set_index("Ticker")["Holders_At"], errors="coerce")
        fresh = set(at[at >= pd.Timestamp.now().normalize() - pd.Timedelta(days=HOLDERS_FRESH_DAYS)].index)
        todo = [t for t in todo if t not in fresh]
    if max_n > 0:
        todo = todo[:max_n]
    log.info("Holders to fetch: %d companies", len(todo))
    if todo:
        fetch_mod.fetch_holders(raw, todo, cfg, save_to=cfg.raw_path)
    return len(todo)
