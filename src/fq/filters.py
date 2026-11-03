"""Pre-ranking scope and data-consistency filters."""

import logging
import re

import pandas as pd

from fq.fx import MINOR_UNITS
from fq.universe import is_excluded, share_classes

log = logging.getLogger(__name__)


def clean_and_filter(
    df: pd.DataFrame, in_scope: list[str], protected: set[str] | frozenset[str] = frozenset(), verbose: bool = True
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Drop out-of-scope, excluded-industry and inconsistent rows.

    `protected` tickers (the portfolio) are never dropped for their industry; they are flagged
    with `Industry_Excluded_Kept` so reports can say that operating metrics fit them poorly.
    Returns the kept frame and a list of (ticker, reason) for every removal. `verbose=False` logs
    removals at debug level (a benchmark filtered alongside the run).
    """
    info = log.info if verbose else log.debug
    if df is None or df.empty:
        return df, []

    scope = set(in_scope)
    removed: list[tuple[str, str]] = []
    keep = []

    for idx, row in df.iterrows():
        ticker, industry = row["Ticker"], row.get("industry")
        cap, ev = row.get("Effective_MarketCap"), row.get("enterpriseValue")
        reason = None
        if ticker not in scope:
            reason = "not in active universe"
        elif pd.isna(industry) or str(industry).strip() in ("", "N/A"):
            reason = "missing industry classification"
        elif is_excluded(ticker, industry) and ticker not in protected:
            reason = f"industry excluded: {industry}"
        elif pd.isna(cap) or cap <= 0:
            reason = "invalid or missing market cap"
        elif pd.isna(ev):
            reason = "cannot compute enterprise value"
        elif pd.isna(row.get("shortName")):
            reason = "missing company name"

        if reason:
            removed.append((ticker, reason))
            # Cached tickers outside this run's selection are expected, not news.
            (log.debug if reason == "not in active universe" else info)("Removing %s: %s", ticker, reason)
        else:
            keep.append(idx)

    kept = df.loc[keep].reset_index(drop=True)
    kept, merged = merge_share_classes(kept, protected, verbose)
    removed += merged
    info("Kept %d of %d companies", len(kept), len(df))
    kept["Industry_Excluded_Kept"] = [
        is_excluded(t, i) and t in protected for t, i in zip(kept["Ticker"], kept["industry"], strict=True)
    ]
    for t in kept.loc[kept["Industry_Excluded_Kept"], "Ticker"]:
        log.info("Keeping %s despite excluded industry (portfolio holding)", t)
    return kept, removed


def merge_share_classes(
    df: pd.DataFrame, protected: set[str] | frozenset[str] = frozenset(), verbose: bool = True
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Keep one ticker per company: share classes (GOOG/GOOGL, FOX/FOXA) and listings on two exchanges
    (IAG.L/IAG.MC) share one set of statements, so both would fill a percentile slot and appear twice.

    Same company = the same latest period end and, in the reporting currency, revenue and net income within
    `SAME_STATEMENTS` (each listing is converted at its own fetch day's rate, `Fin_Rate`, so the USD figures of
    FPE.DE / FPE3.DE differed by 0.08% and were ranked twice), confirmed by the name (a trailing class or
    currency tag such as "MFE A" / "MFE B" or "... EUR" / "... SGD" left out) or, for listings whose names Yahoo
    truncates differently, by the same share count. Keeps the portfolio ticker, else the
    home listing (quoted in the reporting currency, pence as pounds: the main ticker, not a foreign line or
    ADR; user's rule), else the one with more shares outstanding. Classes with unequal rights in
    `lists/share_classes.txt` (Swire A/B) are measured separately and never merged. `Sister_Class` names
    the dropped tickers.
    """
    out = df.copy()
    out["Sister_Class"] = None
    keys = ["Total Revenue_Y0", "Net Income_Y0", "PeriodEnd_Y0"]
    if df.empty or not all(k in df.columns for k in keys):
        return out, []
    linked = set(share_classes())
    name = out["shortName"].map(_base_name)
    shares = pd.to_numeric(out.get("sharesOutstanding", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    quote = out.get("currency", pd.Series(None, index=out.index)).map(lambda c: MINOR_UNITS.get(c, (c, 1))[0])
    home = quote == out.get("financialCurrency", pd.Series(None, index=out.index))
    rate = pd.to_numeric(out.get("Fin_Rate", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    revenue = pd.to_numeric(out["Total Revenue_Y0"], errors="coerce") / rate
    income = pd.to_numeric(out["Net Income_Y0"], errors="coerce") / rate
    candidates = out[out[keys].notna().all(axis=1) & ~out["Ticker"].isin(linked)]
    drop, removed = [], []
    for _, period in candidates.groupby("PeriodEnd_Y0"):
        for group in _close(revenue[period.index].sort_values()):
            for same in _same_company(group, name, shares, income):
                held = out.loc[same, "Ticker"].isin(protected)
                order = pd.DataFrame({"held": held, "home": home[same], "shares": shares[same]})
                best = order.sort_values(["held", "home", "shares"], ascending=False).index[0]
                others = [i for i in same if i != best]
                out.at[best, "Sister_Class"] = ", ".join(out.loc[others, "Ticker"])
                for i in others:
                    reason = f"second listing or share class of {out.at[best, 'Ticker']}"
                    removed.append((out.at[i, "Ticker"], reason))
                    (log.info if verbose else log.debug)("Removing %s: %s", out.at[i, "Ticker"], reason)
                drop += others
    return out.drop(index=drop).reset_index(drop=True), removed


# Two listings of one set of statements, converted on different days: their figures differ by the rate's move.
SAME_STATEMENTS = 0.005


def _near(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= SAME_STATEMENTS * max(abs(a), abs(b))


def _close(revenue: pd.Series) -> list[list]:
    """Runs of rows (sorted by revenue) whose revenue is within `SAME_STATEMENTS` of the run's first."""
    runs: list[list] = []
    first = None
    for i, value in revenue.items():
        if runs and _near(value, first):
            runs[-1].append(i)
        else:
            runs.append([i])
            first = value
    return [r for r in runs if len(r) > 1]


def _base_name(n) -> str:
    """The name without a trailing class or currency tag of up to three characters ("MFE A", "BAYWA AG  V",
    "Stoneweg EUTrust SGD"), lower case, letters and digits only."""
    words = str(n).split()
    if len(words) > 1 and len(words[-1]) <= 3:
        words = words[:-1]
    return re.sub(r"[^a-z0-9]", "", " ".join(words).lower())


def _same_company(index, name: pd.Series, shares: pd.Series, income: pd.Series) -> list[list]:
    """Groups of 2+ rows (the same statements) confirmed as one company: same name or same share count."""
    groups: list[list] = []
    for i in index:
        for g in groups:
            j = g[0]
            if not _near(income[i], income[j]):
                continue
            if name[i] == name[j] or (shares[i] > 0 and shares[j] > 0 and abs(shares[i] / shares[j] - 1) < 0.001):
                g.append(i)
                break
        else:
            groups.append([i])
    return [g for g in groups if len(g) > 1]
