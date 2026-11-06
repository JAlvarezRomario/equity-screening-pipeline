"""Command line interface: `fq fetch | rank | report | run | grade | holders | forward | evaluate | clean`.

Options and prompts only; the work is in `fq.pipeline`."""

import functools
import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from fq import pipeline
from fq.pipeline import Mode, PipelineError
from fq.settings import DEFAULT_BENCHMARK, SELF_BENCHMARKED, Settings
from fq.universe import Selection, resolve_selection

app = typer.Typer(help="Fundamental value/quality screener with an HTML report.", no_args_is_help=True)
log = logging.getLogger("fq")

UniverseOpt = Annotated[
    str | None,
    typer.Option(
        "--universe", "-u",
        help="'portfolio', 'benchmark' (the global set), 'test', a ticker file, or comma-separated tickers "
        "(mixable: 'portfolio,KO'). Omit to choose from a menu.",
    ),
]  # fmt: skip
DataDirOpt = Annotated[Path, typer.Option(help="Where cached data is stored.")]
OutDirOpt = Annotated[Path, typer.Option(help="Base folder for reports (one sub-folder per run).")]
FullOpt = Annotated[
    bool, typer.Option("--full", help="Re-fetch fundamentals of every ticker (default: only where needed).")
]
OnlyNewOpt = Annotated[bool, typer.Option(help="Fetch only tickers missing from the cache (no price refresh).")]
PricesOnlyOpt = Annotated[
    bool, typer.Option(help="Reuse saved fundamentals, download only prices (new tickers are fetched fully).")
]
MaxFetchOpt = Annotated[
    int, typer.Option(help="Most full fetches in an auto run, by priority (0 = no cap); the rest wait for later runs.")
]
BenchmarkOpt = Annotated[
    str | None,
    typer.Option(
        help=f"Score the run against this set (any --universe token, or 'none'); ranks stay within the run. "
        f"Default: {DEFAULT_BENCHMARK}, none for {'/'.join(sorted(SELF_BENCHMARKED))} runs.",
    ),
]

MENU = """
What do you want to run?
  1) A list of tickers (you type them now)
  2) My portfolio ({n_portfolio})
  3) The global benchmark ({n_bench} tickers)
"""


def _usage(fn):
    """Show a `PipelineError` as a usage error (no traceback)."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PipelineError as exc:
            raise typer.BadParameter(str(exc)) from exc

    return wrapped


def _setup_logging(logs_dir: Path = Path("logs")) -> None:
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(logs_dir / "fq.log", encoding="utf-8")],
        force=True,
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    for noisy in ("httpx", "google_genai", "google_genai.models"):  # one line per Gemini request
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _choose(universe: str | None) -> Selection:
    """Resolve --universe, or ask interactively when it was not given."""
    if universe is None:
        try:
            n_portfolio = f"{len(resolve_selection('portfolio').tickers)} tickers in portfolio.txt"
        except ValueError:
            n_portfolio = "portfolio.txt not found"
        typer.echo(MENU.format(n_portfolio=n_portfolio, n_bench=len(resolve_selection("benchmark").tickers)))
        choice = ""
        while choice not in {"1", "2", "3"}:
            choice = typer.prompt("Choice (1-3)").strip()
        universe = {
            "1": lambda: typer.prompt("Tickers (comma or space separated, e.g. AAPL MSFT SAP.DE)").replace(" ", ","),
            "2": lambda: "portfolio",
            "3": lambda: "benchmark",
        }[choice]()
    try:
        return resolve_selection(universe)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _cfg(sel: Selection, data_dir: Path, output_dir: Path) -> Settings:
    return Settings(data_dir=data_dir, output_dir=output_dir, label=sel.label)


def _mode(full: bool, only_new: bool, prices_only: bool) -> Mode:
    if full + only_new + prices_only > 1:
        raise typer.BadParameter("Use only one of --full, --only-new and --prices-only.")
    return "full" if full else "new" if only_new else "prices" if prices_only else "auto"


def _choose_refresh() -> tuple[Mode | None, bool]:
    """Menu: how much to download (Enter = auto) -> (mode, None = no download; download only, no ranking)."""
    typer.echo("\nWhat should be downloaded?\n"
               "  1) Auto: prices, plus full data where a new report is due or data is old (default)\n"
               "  2) Everything (fresh fundamentals + prices, slow)\n"
               "  3) Prices only (reuse saved fundamentals, fast)\n"
               "  4) Download only: every new or due company (no cap) and prices; no ranking or pages\n"
               "  5) Nothing: rank and write the pages from the cache (unchanged pages are left as they are)\n")  # fmt: skip
    choice = None
    while choice not in {"1", "2", "3", "4", "5"}:
        choice = typer.prompt("Choice (1-5)", default="1").strip()
    return {"1": ("auto", False), "2": ("full", False), "3": ("prices", False), "4": ("auto", True),
            "5": (None, False)}[choice]  # fmt: skip


@app.command()
@_usage
def fetch(
    universe: UniverseOpt = None,
    full: FullOpt = False,
    only_new: OnlyNewOpt = False,
    prices_only: PricesOnlyOpt = False,
    benchmark: BenchmarkOpt = None,
    max_fetch: MaxFetchOpt = Settings.max_full_fetches,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """Download from Yahoo Finance into the local cache (default: fundamentals only where needed)."""
    _setup_logging()
    mode = _mode(full, only_new, prices_only)
    sel = _choose(universe)
    cfg = replace(_cfg(sel, data_dir, output_dir), max_full_fetches=max_fetch)
    pipeline.fetch_data(sel, cfg, mode, pipeline.benchmark_selection(sel, benchmark))


@app.command()
@_usage
def rank(
    universe: UniverseOpt = None,
    benchmark: BenchmarkOpt = None,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """Compute metrics, filter and rank from the cache (works offline)."""
    _setup_logging()
    sel = _choose(universe)
    pipeline.rank_run(sel, _cfg(sel, data_dir, output_dir), pipeline.benchmark_selection(sel, benchmark))


@app.command()
@_usage
def report(
    universe: UniverseOpt = None,
    index_only: Annotated[
        bool,
        typer.Option(
            "--index-only",
            "-i",
            help="Only rebuild the front page (index.html, quadrant, rankings), skipping company pages.",
        ),
    ] = False,
    companies_only: Annotated[
        bool, typer.Option("--companies-only", help="Only rebuild individual company pages.")
    ] = False,
    ticker: Annotated[
        str | None, typer.Option("--ticker", "-t", help="Rebuild only a specific company page (e.g. --ticker AAPL).")
    ] = None,
    open_: Annotated[bool, typer.Option("--open/--no-open", help="Open the report in the browser when done.")] = False,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """Write the HTML site from the ranked data."""
    _setup_logging()
    if index_only and companies_only:
        raise typer.BadParameter("Cannot use both --index-only and --companies-only.")
    sel = _choose(universe)
    tickers = {ticker.upper()} if ticker else None
    pipeline.write_report(
        _cfg(sel, data_dir, output_dir),
        open_after=open_,
        index_only=index_only,
        companies_only=companies_only,
        tickers=tickers,
    )


@app.command()
@_usage
def run(
    universe: UniverseOpt = None,
    full: FullOpt = False,
    only_new: OnlyNewOpt = False,
    offline: Annotated[bool, typer.Option(help="Skip fetching; reuse the cache.")] = False,
    prices_only: PricesOnlyOpt = False,
    benchmark: BenchmarkOpt = None,
    max_fetch: MaxFetchOpt = Settings.max_full_fetches,
    open_: Annotated[bool, typer.Option("--open/--no-open", help="Open the report in the browser when done.")] = False,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """fetch -> rank -> report in one go (asks what to run if --universe is omitted)."""
    _setup_logging()
    interactive = universe is None
    mode = _mode(full, only_new, prices_only)
    sel = _choose(universe)
    cfg = replace(_cfg(sel, data_dir, output_dir), max_full_fetches=max_fetch)
    download_only = False
    if interactive and not (offline or full or prices_only or only_new):
        chosen, download_only = _choose_refresh()
        offline, mode = chosen is None, chosen or mode
        if download_only:
            cfg = replace(cfg, max_full_fetches=0)
    bench = pipeline.benchmark_selection(sel, benchmark)
    log.info("Run '%s': %d tickers (%s)%s", sel.label, len(sel.tickers), "offline" if offline else mode,
             f", benchmark {bench.label}" if bench else "")  # fmt: skip
    if not offline:
        pipeline.fetch_data(sel, cfg, mode, bench)
    if download_only:
        typer.echo("\nDownloaded. To rank and write the pages: run.bat, the same set, then 5.")
        return
    pipeline.rank_run(sel, cfg, bench)
    pipeline.write_report(cfg, open_, offline=offline)


@app.command("grade")
@_usage
def grade_cmd(
    max_grades: Annotated[int, typer.Option(help="Most companies graded in this run (0 = no cap).")] = 0,
    dry_run: Annotated[bool, typer.Option(help="Only count the companies due and estimate the cost.")] = False,
    resynthesize: Annotated[
        bool, typer.Option(help="Only rewrite syntheses made with an older synthesis prompt (no new grades).")
    ] = False,
    report: Annotated[bool, typer.Option(help="Rebuild the site afterwards (from the cache, no download).")] = True,
    new_only: Annotated[
        bool, typer.Option(help="Only companies never graded: no re-grade and no synthesis rewrite is spent.")
    ] = False,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """Moat and management grades (Gemini) for the companies the benchmark's highlight lists show, where missing or
    due (a new annual report, or a year old); `fq run -u benchmark` does this too, within `max_grades`."""
    _setup_logging()
    cfg = Settings(data_dir=data_dir, output_dir=output_dir, label=DEFAULT_BENCHMARK)
    pipeline.grade_only(cfg, max_grades, dry_run, resynthesize, report, new_only)


@app.command("holders")
@_usage
def holders_cmd(
    universe: UniverseOpt = DEFAULT_BENCHMARK,
    every: Annotated[
        bool, typer.Option("--all", help="Every cached company of the run, not only those the lists can show.")
    ] = False,
    max_n: Annotated[int, typer.Option("--max", help="Most companies fetched in this run (0 = no cap).")] = 0,
    force: Annotated[
        bool, typer.Option(help=f"Fetch again holders younger than {pipeline.HOLDERS_FRESH_DAYS} days.")
    ] = False,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """The largest holders (institutions, funds, insiders) alone, one light request per company, written into the
    cache without touching anything else; every full fetch stores them too. Then `fq run -u benchmark --offline`."""
    _setup_logging()
    sel = _choose(universe)
    n = pipeline.fetch_holders_run(sel, _cfg(sel, data_dir, output_dir), every, max_n, force)
    typer.echo(f"Holders fetched for {n} companies; now `fq run -u {universe} --offline` to show them.")


@app.command("forward")
@_usage
def forward_cmd(
    offline: Annotated[bool, typer.Option(help="Download nothing; rebuild the page from what is stored.")] = False,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """The backward and forward test alone: freeze the lists (once, from the tabs of `forward.FROZEN_ON`), add the month-ends passed, write
    forward.html (`fq run -u benchmark` does this too)."""
    _setup_logging()
    cfg = Settings(data_dir=data_dir, output_dir=output_dir, label=DEFAULT_BENCHMARK)
    pipeline.forward_only(cfg, offline)


@app.command()
@_usage
def evaluate(
    universe: UniverseOpt = None,
    rescore: Annotated[
        bool, typer.Option(help="Score every archived raw cache again with today's method before evaluating.")
    ] = False,
    data_dir: DataDirOpt = Path("data"),
    output_dir: OutDirOpt = Path("output"),
) -> None:
    """Follow saved score snapshots forward: did the top lists beat the median company?"""
    _setup_logging()
    sel = _choose(universe)
    pipeline.evaluate_run(_cfg(sel, data_dir, output_dir), rescore)


@app.command()
@_usage
def clean(
    output_dir: OutDirOpt = Path("output"),
    cache: Annotated[
        bool,
        typer.Option(help="Also delete the data cache (keeps the forward test, grades and snapshots)."),
    ] = False,
    data_dir: DataDirOpt = Path("data"),
) -> None:
    """Delete generated reports (and optionally the cache that can be downloaded again)."""
    removed, kept = pipeline.clean(output_dir, data_dir, cache)
    for path in removed:
        typer.echo(f"Removed {path}")
    for path in kept:
        typer.echo(f"Kept {path} (cannot be downloaded again; delete by hand if ever needed)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
