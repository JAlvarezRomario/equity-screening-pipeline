"""Forward tracking: does a high score lead to a better return? Pure functions except the snapshot I/O.

Yahoo history is restated and holds no delisted companies, so a backtest on it would flatter the
method. Instead every `rank` saves a dated snapshot of the scores, and `fq evaluate` later measures
what happened next: the forward total return (dividends included, in USD) of the top of each
list against the median company of the same snapshot, and each criterion's information coefficient
(rank correlation of the percentile with the forward return).

Each snapshot records the method that scored it (`Method`, `method_fingerprint`), and results are kept
apart per method. Every `rank` also archives the raw cache as it stood that day (`save_raw_archive`, what
was known then), so `fq evaluate --rescore` can score every archived date again with today's method and
compare methods on the same dates instead of waiting months for each.
"""

import hashlib
import logging
import os
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from fq.ranking import comparable, top_n
from fq.settings import Settings

log = logging.getLogger(__name__)

HORIZONS_MONTHS = (3, 6, 12)
TOP_N = 20
# The goal is to beat the global market: iShares MSCI ACWI (USD, total return; its fees understate the index
# slightly). Followed beside the snapshot's median company, never scored.
MARKET = "ACWI"
STABLE_RANKS = 0.8  # Spearman of ranks between consecutive snapshots below this: a data break? (`stability`)
SNAPSHOT_FLAGS = (
    "Data_Suspect", "Insufficient_Data", "Industry_Excluded_Kept", "Financial_Mode", "Cyclical", "Dislocation",
    "Peak_Margin", "Distress", "Asset_Play", "Revenue_Decline", "Accounting_Flag",
)  # fmt: skip
SNAPSHOT_COLUMNS = (
    "Ticker", "sector", "currentPrice", "Rank", "Value_Rank", "Quality_Rank", "Value_Score", "Quality_Score",
    "Momentum_6M", "EPV_Yield", "currency", "Business_Model", "Ranked_Model", "Benchmark",
    *SNAPSHOT_FLAGS,
)  # fmt: skip
# What decides a score: the scoring code and the lists that set models, cyclicals and exclusions. Index member
# lists are left out (they decide who is scored, which the archive records), and so are the reports.
# Plus every module of `fq.metrics`.
METHOD_MODULES = ("ranking.py", "settings.py", "filters.py", "universe.py", "inflation.py")
METHOD_LISTS = (
    "business_models.txt", "model_overrides.txt", "cyclical_industries.txt", "excluded_industries.txt",
    "excluded_tickers.txt", "share_classes.txt", "accounting_standards.txt", "inflation_basis.txt",
)  # fmt: skip
UNKNOWN_METHOD = "unknown"  # snapshots saved before methods were recorded


def method_files() -> list:
    """The files `method_fingerprint` hashes: the `fq.metrics` package, `METHOD_MODULES` and `METHOD_LISTS`. A
    missing one is an error, so a renamed module cannot silently drop out of the fingerprint."""
    package = resources.files("fq")
    metrics = sorted((f for f in package.joinpath("metrics").iterdir() if f.name.endswith(".py")), key=lambda f: f.name)
    files = metrics + [package.joinpath(m) for m in METHOD_MODULES]
    files += [package.joinpath("lists", n) for n in METHOD_LISTS]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"method files missing: {', '.join(missing)}")
    return files


def method_fingerprint() -> str:
    """8 hex characters identifying the scoring method: a hash of `method_files` (line endings normalised). Any
    edit changes it, comments included: results are only pooled when the code was identical."""
    digest = hashlib.sha1()
    for f in method_files():
        digest.update(f.name.encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:8]


def snapshot(ranked: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """What a later evaluation needs: scores, every criterion percentile, the flags and the method."""
    cols = [c for c in SNAPSHOT_COLUMNS if c in ranked.columns]
    cols += [c for c in ranked.columns if c.startswith(("Pct_", "Cand_Pct_"))]
    out = ranked[cols].copy()
    out["Method"] = method or method_fingerprint()
    return out


def save_snapshot(ranked: pd.DataFrame, folder: Path, label: str, day: pd.Timestamp, method: str | None = None) -> Path:
    """`<folder>/<label>_<YYYY-MM-DD>.parquet`; a second run on the same day replaces the first."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}_{day:%Y-%m-%d}.parquet"
    snapshot(ranked, method).to_parquet(path, index=False)
    return path


def save_published(lists: pd.DataFrame, folder: Path, label: str, day: pd.Timestamp, method: str | None = None) -> Path:
    """What each highlight tab showed (`highlights.published_lists`: Ticker, List), as
    `<folder>/lists/<label>_<YYYY-MM-DD>.parquet`, so `evaluate` follows the tabs as published (cut, trust and
    liquidity rules included) and not as the flags alone would draw them. A second report on the same day replaces it."""
    target = folder / "lists"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{label}_{day:%Y-%m-%d}.parquet"
    lists.assign(Method=method or method_fingerprint()).to_parquet(path, index=False)
    return path


def load_published(folder: Path, label: str) -> dict[pd.Timestamp, pd.DataFrame]:
    """Published tabs of one run label (`save_published`), keyed by date."""
    return load_snapshots(folder / "lists", label)


def save_raw_archive(raw: pd.DataFrame, folder: Path, day: pd.Timestamp, benchmark: list[str]) -> Path:
    """The raw cache as it stood on `day` (`<folder>/raw/<YYYY-MM-DD>.parquet`), with `In_Benchmark` marking that
    day's benchmark members, so a later method can score the same date without look-ahead. One per day (the
    cache only grows within a day, so the last run's copy holds the others).

    A copy byte-identical to the newest earlier archive (an offline run on an unchanged cache) is stored as a
    hard link to it, so every date stays for `--rescore` at no disk cost (Nov 3, 2026: 32 MB a day). Written to a
    temporary file and renamed, never in place, so rewriting a day never alters a file linked to another day."""
    target = folder / "raw"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{day:%Y-%m-%d}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    try:
        raw.assign(In_Benchmark=raw["Ticker"].isin(set(benchmark))).to_parquet(tmp, index=False)
        earlier = [p for p in sorted(target.glob("*.parquet")) if p.stem < path.stem]
        if earlier and same_bytes(tmp, earlier[-1]):
            path.unlink(missing_ok=True)
            try:
                os.link(earlier[-1], path)
                return path
            except OSError:  # no hard links on this file system: keep the full copy
                pass
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def same_bytes(a: Path, b: Path) -> bool:
    """Whether two files hold the same bytes (size first, then SHA-256)."""
    if a.stat().st_size != b.stat().st_size:
        return False

    def digest(p: Path) -> bytes:
        with p.open("rb") as f:
            return hashlib.file_digest(f, "sha256").digest()

    return digest(a) == digest(b)


def load_raw_archives(folder: Path) -> dict[pd.Timestamp, Path]:
    """Archived raw caches by date, oldest first (paths: each is read only when it is scored)."""
    out = {}
    for path in sorted((folder / "raw").glob("*.parquet")):
        day = pd.to_datetime(path.stem, format="%Y-%m-%d", errors="coerce")
        if pd.notna(day):
            out[day] = path
    return out


def load_snapshots(folder: Path, label: str) -> dict[pd.Timestamp, pd.DataFrame]:
    """Snapshots of one run label, keyed by date, oldest first."""
    out = {}
    for path in sorted(folder.glob(f"{label}_*.parquet")):
        day = pd.to_datetime(path.stem.removeprefix(f"{label}_"), format="%Y-%m-%d", errors="coerce")
        if pd.notna(day):
            out[day] = pd.read_parquet(path)
    return out


def to_usd(prices: pd.DataFrame, currencies: dict[str, str], fx_history: dict[str, pd.Series | None]) -> pd.DataFrame:
    """Quote-currency closes in USD, so returns are measured like the fundamentals (the USD investor).

    `fx_history[ccy]` = daily USD value of one unit (`fx._usd_history`); `GBp` (pence) uses GBP / 100.
    A ticker whose currency has no history keeps its quote-currency prices, with a warning.
    """
    out = prices.sort_index().copy()
    missing = []
    for ticker in out.columns:
        ccy = currencies.get(ticker) or "USD"
        if ccy == "USD":
            continue
        base, scale = ("GBP", 0.01) if ccy == "GBp" else (ccy, 1.0)
        hist = fx_history.get(base)
        if hist is None or hist.empty:
            missing.append(ticker)
            continue
        rate = hist.sort_index().reindex(out.index.union(hist.index)).ffill().reindex(out.index)
        out[ticker] = out[ticker] * rate * scale
    if missing:
        log.warning("No FX history for %s: their returns stay in quote currency", ", ".join(missing))
    return out


def forward_returns(prices: pd.DataFrame, start: pd.Timestamp, months: int) -> pd.Series | None:
    """% change per ticker from the last close on or before `start` to the last close on or before
    `start + months`; None while that end date is still in the future (or before the price history)."""
    prices = prices.sort_index().ffill()
    end = start + pd.DateOffset(months=months)
    if prices.empty or end > prices.index[-1]:
        return None
    before, until = prices.loc[:start], prices.loc[:end]
    if before.empty:
        return None
    base = before.iloc[-1]
    return (until.iloc[-1] / base.where(base > 0) - 1) * 100


def _method(snap: pd.DataFrame) -> str:
    """The scoring method a snapshot records (`UNKNOWN_METHOD` before methods were recorded)."""
    return str(snap["Method"].iloc[0]) if "Method" in snap.columns and len(snap) else UNKNOWN_METHOD


def _spearman(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    return float(pair.rank().corr().iloc[0, 1]) if len(pair) >= 10 else np.nan


def _lists(snap: pd.DataFrame, top: int, published: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Ticker lists to follow: the report's trusted top lists, the Value quintiles and, with `published` (that day's
    `save_published`), every highlight tab as the site showed it ("Tab: <name>")."""
    lists = {
        f"Top {top} combined": top_n(snap, "Rank", top)["Ticker"],
        f"Top {top} Value": top_n(snap, "Value_Rank", top)["Ticker"],
        f"Top {top} Quality": top_n(snap, "Quality_Rank", top)["Ticker"],
    }
    if "Compounder" in snap.columns:  # the Mastercard-like list, cheapest first in the report: shown, not tuned
        lists["Compounders"] = snap.loc[snap["Compounder"] == True, "Ticker"]  # noqa: E712
    if "Asset_Play" in snap.columns:
        lists["Asset plays"] = snap.loc[snap["Asset_Play"] == True, "Ticker"]  # noqa: E712
    if "Cannibal" in snap.columns:
        lists["Cannibals"] = snap.loc[snap["Cannibal"] == True, "Ticker"]  # noqa: E712
    if "Dislocation" in snap.columns:
        lists["Dislocation"] = snap.loc[snap["Dislocation"] == True, "Ticker"]  # noqa: E712
        # Controls: does the mood filter beat cheapness alone? Trusted, comparable cheap companies.
        trusted = comparable(snap)
        for flag in ("Data_Suspect", "Insufficient_Data"):
            if flag in snap.columns:
                trusted &= snap[flag] != True  # noqa: E712
        min_value = Settings().dislocation_min_value
        cheap = trusted & (snap["Value_Score"] >= min_value)
        lists[f"Cheap (Value >= {min_value:.0f})"] = snap.loc[cheap, "Ticker"]
        lists["Cheap, explained"] = snap.loc[cheap & (snap["Dislocation"] != True), "Ticker"]  # noqa: E712
        if "Peak_Margin" in snap.columns:  # does leaving cyclicals at peak margins out of the top lists help?
            lists["Cheap, peak margins"] = snap.loc[cheap & (snap["Peak_Margin"] == True), "Ticker"]  # noqa: E712
        if "Distress" in snap.columns:  # does leaving distressed companies out of the top lists help?
            lists["Cheap, distressed"] = snap.loc[cheap & (snap["Distress"] == True), "Ticker"]  # noqa: E712
        if "Revenue_Decline" in snap.columns:  # does leaving shrinking businesses out of the top lists help?
            lists["Cheap, revenue declining"] = snap.loc[cheap & (snap["Revenue_Decline"] == True), "Ticker"]  # noqa: E712
        if "Accounting_Flag" in snap.columns:  # does leaving earnings ahead of cash out of the lists help?
            lists["Cheap, accruals"] = snap.loc[cheap & (snap["Accounting_Flag"] == True), "Ticker"]  # noqa: E712
    scored = snap[comparable(snap)].dropna(subset=["Value_Score"])
    if len(scored) >= 25:
        quintile = pd.qcut(scored["Value_Score"].rank(method="first"), 5, labels=False)
        for q in range(5):
            lists[f"Value quintile {5 - q}"] = scored.loc[quintile == q, "Ticker"]  # 1 = cheapest
    if published is not None:
        for name, group in published.groupby("List", sort=False):
            lists[f"Tab: {name}"] = group["Ticker"]
    return lists


def evaluate_snapshot(
    snap: pd.DataFrame, fwd: pd.Series, top: int = TOP_N, market: float = np.nan, published: pd.DataFrame | None = None
) -> tuple[list[dict], dict[str, float]]:
    """(one row per list: size, mean and excess return over the snapshot's median company and over the market's
    return `market` (`MARKET`); {score or criterion: information coefficient})."""
    fwd = fwd.reindex(snap["Ticker"]).set_axis(snap.index)
    median = float(fwd.median())
    rows = []
    for name, tickers in _lists(snap, top, published).items():
        r = fwd[snap["Ticker"].isin(tickers)].dropna()
        if len(r):
            mean = float(r.mean())
            rows.append({"List": name, "N": len(r), "Return": mean, "Excess": mean - median,
                         "Excess_Market": mean - market})  # fmt: skip
    # Headline ICs over the ranked models only: the others are percentiles within their own peer group.
    ranked, ranked_fwd = snap[comparable(snap)], fwd[comparable(snap)]
    ic = {col: _spearman(ranked[col], ranked_fwd) for col in ("Value_Score", "Quality_Score") if col in snap.columns}
    # Per business model (`model:<name>:<score>`): the evidence needed before a model joins the ranking.
    if "Business_Model" in snap.columns:
        for model, members in snap.groupby("Business_Model").groups.items():
            for col in ("Value_Score", "Quality_Score"):
                if col in snap.columns:
                    ic[f"model:{model}:{col}"] = _spearman(snap.loc[members, col], fwd[members])
    ic |= {
        col.removeprefix("Pct_"): _spearman(ranked[col], ranked_fwd) for col in snap.columns if col.startswith("Pct_")
    }
    # Candidates under observation (not in any score), labelled apart from the ranked criteria.
    ic |= {
        "cand:" + col.removeprefix("Cand_Pct_"): _spearman(ranked[col], ranked_fwd)
        for col in snap.columns
        if col.startswith("Cand_Pct_")
    }
    return rows, ic


def evaluate(
    snapshots: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS_MONTHS,
    top: int = TOP_N,
    published: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(lists: mean over snapshots per method, horizon and list; ic: mean IC per method, horizon and criterion).

    Snapshots whose horizon has not passed yet are skipped. Overlapping snapshots (weekly runs over
    a 12-month horizon) are not independent: read the snapshot count, not only the averages. Each
    snapshot's `Method` keeps results of different scoring methods apart (`UNKNOWN_METHOD` before it was
    recorded). `published` (`load_published`) adds that day's tabs when they were drawn from the same method (a
    rescored snapshot is not what the site showed).
    """
    list_rows, ic_rows = [], []
    for day, snap in snapshots.items():
        method = _method(snap)
        tabs = (published or {}).get(day)
        if tabs is not None and _method(tabs) != method:
            tabs = None
        for months in horizons:
            fwd = forward_returns(prices, day, months)
            if fwd is None:
                continue
            market = float(fwd[MARKET]) if MARKET in fwd.index and MARKET not in set(snap["Ticker"]) else np.nan
            rows, ic = evaluate_snapshot(snap, fwd, top, market, tabs)
            list_rows += [{"Method": method, "Snapshot": day, "Months": months, **r} for r in rows]
            ic_rows += [{"Method": method, "Snapshot": day, "Months": months, "Measure": k, "IC": v}
                        for k, v in ic.items()]  # fmt: skip
    lists = pd.DataFrame(
        list_rows, columns=["Method", "Snapshot", "Months", "List", "N", "Return", "Excess", "Excess_Market"]
    )
    ics = pd.DataFrame(ic_rows, columns=["Method", "Snapshot", "Months", "Measure", "IC"])
    lists_summary = (
        lists.groupby(["Method", "Months", "List"], sort=False)
        .agg(
            Snapshots=("Snapshot", "nunique"),
            Avg_N=("N", "mean"),
            Return=("Return", "mean"),
            Excess=("Excess", "mean"),
            Excess_Market=("Excess_Market", "mean"),
            Hit_Rate=("Excess", lambda e: (e > 0).mean() * 100),
        )  # fmt: skip
        .reset_index()
    )
    ic_summary = (
        ics.groupby(["Method", "Months", "Measure"], sort=False)
        .agg(Snapshots=("Snapshot", "nunique"), IC=("IC", "mean"))
        .reset_index()
    )
    ic_summary["IC_t"] = [
        _ic_t(
            ics.loc[(ics["Method"] == r.Method) & (ics["Months"] == r.Months) & (ics["Measure"] == r.Measure)], r.Months
        )  # fmt: skip
        for r in ic_summary.itertuples()
    ]
    return lists_summary, ic_summary


def newey_west_t(values: pd.Series, lag: int) -> float:
    """t-statistic of the mean with Newey-West standard errors (Bartlett weights up to `lag`): snapshots whose
    horizons overlap share most of their returns, so their ICs are not independent and a plain t overstates."""
    x = pd.Series(values).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def _ic_t(rows: pd.DataFrame, months: int) -> float:
    """Newey-West t of one measure's ICs over snapshots, the lag being how many later snapshots fall inside a
    snapshot's horizon (weekly runs over 12 months overlap ~50 deep)."""
    days = pd.Series(sorted(rows["Snapshot"].unique()))
    if len(days) < 3:
        return np.nan
    lag = max(int(((days > d) & (days < d + pd.DateOffset(months=months))).sum()) for d in days)
    return newey_west_t(rows.sort_values("Snapshot")["IC"], lag)


def stability(snapshots: dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
    """Spearman correlation of each score between consecutive snapshots of the same method (common companies).

    Statements arrive a few at a time, so ranks should drift slowly; a drop below `STABLE_RANKS` points to a
    data break (a Yahoo change, a unit error) more often than to news. Prices move scores too: read it together
    with what was fetched.
    """
    rows, previous = [], {}
    for day, snap in sorted(snapshots.items()):
        method = _method(snap)
        before = previous.get(method)
        if before is not None:
            prev_day, prev = before
            pair = snap.set_index("Ticker").join(prev.set_index("Ticker"), rsuffix="_prev", how="inner")
            row = {"Method": method, "Snapshot": day, "Previous": prev_day, "N": len(pair)}
            for col in ("Rank", "Value_Score", "Quality_Score"):
                if col in pair.columns and f"{col}_prev" in pair.columns:
                    row[col] = _spearman(pair[col], pair[f"{col}_prev"])
            rows.append(row)
        previous[method] = (day, snap)
    return pd.DataFrame(rows, columns=["Method", "Snapshot", "Previous", "N", "Rank", "Value_Score", "Quality_Score"])
