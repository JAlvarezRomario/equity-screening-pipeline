"""Guards against stale caches and silent Yahoo schema changes."""

import logging

import pandas as pd

from fq import settings as S

log = logging.getLogger(__name__)


class CacheError(RuntimeError):
    """The cached data cannot be used as is; the message says what to do."""


def check_schema(df: pd.DataFrame) -> None:
    """Refuse a cache written by another schema version (e.g. before relative periods)."""
    found = df["Schema_Version"].dropna().unique().tolist() if "Schema_Version" in df.columns else []
    if found != [S.SCHEMA_VERSION]:
        raise CacheError(
            f"Cached data has schema {found or 'none'}, this version needs {S.SCHEMA_VERSION}. "
            "Delete it (`fq clean --cache`) and fetch again."
        )


def cache_age_days(df: pd.DataFrame) -> int | None:
    """Age of the oldest row in the cache."""
    if "Fetched_At" not in df.columns:
        return None
    stamps = pd.to_datetime(df["Fetched_At"], errors="coerce").dropna()
    return None if stamps.empty else (pd.Timestamp.now() - stamps.min()).days


def coverage(df: pd.DataFrame, cfg: S.Settings) -> dict[str, float]:
    """Share of companies with each critical concept in their newest period (TTM or latest annual)."""
    out = {}
    for concept in S.CRITICAL_CONCEPTS:
        cols = [f"{concept}_{p}" for p in ("TTM", cfg.latest) if f"{concept}_{p}" in df.columns]
        out[concept] = float(df[cols].notna().any(axis=1).mean()) if cols and len(df) else 0.0
    return out


def report(df: pd.DataFrame, cfg: S.Settings) -> list[str]:
    """Log data-health warnings and return them (empty list = healthy)."""
    warnings = []
    age = cache_age_days(df)
    if age is not None and age > cfg.max_cache_age_days:
        warnings.append(f"Cache is {age} days old (limit {cfg.max_cache_age_days}); consider `fq fetch`.")
    for concept, share in coverage(df, cfg).items():
        if len(df) >= 5 and share < S.CRITICAL_MIN_COVERAGE:
            warnings.append(
                f"Only {share:.0%} of companies have '{concept}': Yahoo may have renamed or dropped it "
                "(add the new name to settings.ALIASES)."
            )
    if "Stale_Annual" in df.columns:
        stale = df.loc[df["Stale_Annual"].astype(bool), "Ticker"].tolist()
        if stale:
            warnings.append(f"Latest annual report is over 15 months old for: {', '.join(stale)}.")
    if "FX_Source" in df.columns:
        stale = sorted({s for s in df["FX_Source"].dropna() if "stale" in s})
        if stale:
            warnings.append(f"Exchange rates were not live when fetching: {', '.join(stale)}.")
    if "financialCurrency" in df.columns:
        missing = df.loc[df["financialCurrency"].isna(), "Ticker"].tolist()
        if missing:
            warnings.append(
                f"No reporting currency stored for {len(missing)} companies (the quote's is used until their next "
                f"full fetch): {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}."
            )
    for message in warnings:
        log.warning(message)
    return warnings
