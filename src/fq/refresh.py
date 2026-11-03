"""Which cached companies need fresh fundamentals; the rest only get new prices.

Pure functions: `now` is passed in, nothing is fetched. A due date (earnings call, expected quarterly
or annual filing) only triggers when it falls after the last fetch, so a company that reports
half-yearly or late costs at most one extra fetch per cycle. `select` spends a run's budget of full
fetches by priority; the rest wait for a later run (they still get new prices).
"""

import pandas as pd

from fq import settings as S

# In priority order (`select`): no data, then new reports, then what only improves data already there.
REASONS = ("new", "earnings passed", "annual due", "quarter due", "new fields", "incomplete", "old")


def _stamp(value) -> pd.Timestamp | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, int | float):  # Yahoo epoch seconds (earningsTimestamp)
        return pd.Timestamp(value, unit="s")
    stamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(stamp) else stamp


# Missing in both TTM and the latest annual report = a failed download worth retrying. Not every
# critical concept: banks and insurers never report Operating Income, so they would retry for ever.
# Operating Cash Flow catches an empty cash-flow statement (CVX once came back without one).
RETRY_CONCEPTS = ("Total Revenue", "Net Income", "Operating Cash Flow")


def _incomplete(row: pd.Series, cfg: S.Settings) -> bool:
    """No TTM, or revenue / net income / operating cash flow missing in both TTM and the latest annual report."""
    if _stamp(row.get("PeriodEnd_TTM")) is None:
        return True
    return any(all(pd.isna(row.get(f"{concept}_{p}")) for p in ("TTM", cfg.latest)) for concept in RETRY_CONCEPTS)


def reason(row: pd.Series, cfg: S.Settings, now: pd.Timestamp) -> str | None:
    """Why this cached company needs a full fetch, or None when a price refresh is enough."""
    fetched = _stamp(row.get("Fetched_At"))
    if fetched is None:
        return "old"

    def due(when: pd.Timestamp | None) -> bool:
        return when is not None and fetched < when <= now

    earnings = _stamp(row.get("earningsTimestamp"))
    if earnings is not None and fetched < earnings <= now - pd.Timedelta(days=cfg.earnings_settle_days):
        return "earnings passed"
    quarter = _stamp(row.get("last_quarter")) or _stamp(row.get("PeriodEnd_TTM"))
    if quarter is not None and due(quarter + pd.DateOffset(months=3) + pd.Timedelta(days=cfg.report_lag_days)):
        return "quarter due"
    annual = _stamp(row.get(f"PeriodEnd_{cfg.latest}"))
    if annual is not None and due(annual + pd.DateOffset(years=1) + pd.Timedelta(days=cfg.annual_report_lag_days)):
        return "annual due"
    if row.get("Fields_Version") != S.FIELDS_VERSION:  # fetched before a row was added to what we collect
        return "new fields"
    if (now - fetched).days > cfg.max_cache_age_days:
        return "old"
    if (now - fetched).days > cfg.incomplete_retry_days and _incomplete(row, cfg):
        return "incomplete"
    return None


def plan(existing: pd.DataFrame | None, tickers: list[str], cfg: S.Settings, now: pd.Timestamp) -> dict[str, str]:
    """Tickers needing a full fetch, with the reason; the others only need a price refresh."""
    rows = {} if existing is None or existing.empty else {r["Ticker"]: r for _, r in existing.iterrows()}
    out = {}
    for ticker in tickers:
        why = "new" if ticker not in rows else reason(rows[ticker], cfg, now)
        if why:
            out[ticker] = why
    return out


def select(run_due: dict[str, str], bench_due: dict[str, str], cap: int) -> tuple[list[str], list[str]]:
    """(tickers to fetch now, tickers deferred to a later run) from the due companies of the run and of
    its benchmark (`plan`). The run's own new tickers are always fetched: without data they cannot be
    ranked. Within `cap` more (0 = no cap): the run's due companies, then the benchmark's, each by
    `REASONS` order."""
    priority = {r: i for i, r in enumerate(REASONS)}

    def ordered(due: dict[str, str]) -> list[str]:
        return sorted(due, key=lambda t: priority[due[t]])

    must = [t for t in run_due if run_due[t] == "new"]
    rest = [t for t in ordered(run_due) if t not in must] + [t for t in ordered(bench_due) if t not in run_due]
    room = len(rest) if cap <= 0 else max(cap - len(must), 0)
    return must + rest[:room], rest[room:]
