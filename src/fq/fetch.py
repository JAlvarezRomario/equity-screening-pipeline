"""Fetch fundamentals from Yahoo Finance into one wide DataFrame (one row per ticker).

Columns are `{concept}_{period}` with relative periods: `TTM`, `Y0` (the company's latest annual
report), `Y1`, ... Calendar years never appear in a column name; each row's real period ends are
stored in `PeriodEnd_{period}`.
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from fq import fx, inflation
from fq import settings as S
from fq.universe import share_classes

log = logging.getLogger(__name__)

PRICE_KEYS = ("currentPrice", "previousClose", "targetLowPrice")
CAP_KEYS = ("marketCap", "enterpriseValue")
PCT_COLUMNS = ("returnOnEquity", "profitMargins", "operatingMargins")
QUARTER_GAP_DAYS = (60, 130)  # consecutive quarter-ends are ~91 days apart
RATE_LIMIT_HINTS = ("too many requests", "ratelimit", "rate limit", "429")
STALE_ANNUAL_DAYS = 456  # latest annual report older than ~15 months
HISTORY_START = "1990-01-01"  # long share-count series (`yearly_shares`); Yahoo rarely has more


def _safe(getter) -> pd.DataFrame:
    """Yahoo / yfinance attributes can be missing or raise; treat that as 'no data'."""
    try:
        out = getter()
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:
        log.debug("statement unavailable: %s", exc)
        return pd.DataFrame()


def _safe_series(getter) -> pd.Series:
    try:
        out = getter()
        return out if isinstance(out, pd.Series) else pd.Series(dtype=float)
    except Exception as exc:
        log.debug("series unavailable: %s", exc)
        return pd.Series(dtype=float)


def _by_date(stmt: pd.DataFrame) -> pd.DataFrame:
    """Statement as rows = period end (newest first), columns = concepts."""
    if stmt is None or stmt.empty:
        return pd.DataFrame()
    t = stmt.T.copy()
    t.index = pd.to_datetime(t.index, errors="coerce")
    t = t[t.index.notna()]
    return t.sort_index(ascending=False)


def _value(frame: pd.DataFrame, when, concept: str, used: set[str] | None = None) -> float:
    """Concept at one period end: the name, then its aliases, then a composite sum.

    `used` collects "concept <- row" whenever a substitute row was needed (shown in the report notes).
    """
    if frame.empty or when not in frame.index:
        return np.nan
    row = frame.loc[when]
    if isinstance(row, pd.DataFrame):  # duplicated date
        row = row.iloc[0]
    for name in [concept, *S.ALIASES.get(concept, [])]:
        if name in row.index and pd.notna(row[name]):
            if name != concept and used is not None:
                used.add(f"{concept} <- {name}")
            return float(row[name])
    parts = S.COMPOSITES.get(concept)
    if parts and all(p in row.index and pd.notna(row[p]) for p in parts):
        if used is not None:
            used.add(f"{concept} <- {' + '.join(parts)}")
        return float(sum(row[p] for p in parts))
    return np.nan


def annual_slots(frame: pd.DataFrame, n_slots: int, newest: pd.Timestamp | None = None) -> dict[str, pd.Timestamp]:
    """Map Y0..Y{n-1} to period ends by elapsed years since the newest annual period (`newest`: one anchor shared
    by the three statements, `_annual_anchor`; a period after it is left out).

    A missing year leaves its slot empty instead of shifting older years forward, so Y2 is always
    two years before Y0 whatever the fiscal calendar or gaps in Yahoo's data.
    """
    if frame.empty:
        return {}
    newest = frame.index[0] if newest is None else newest
    slots: dict[str, pd.Timestamp] = {}
    for when in frame.index:
        k = round((newest - when).days / 365.25)
        if 0 <= k < n_slots and f"Y{k}" not in slots:
            slots[f"Y{k}"] = when
    return slots


def _annual_anchor(*statements: pd.DataFrame) -> pd.Timestamp | None:
    """The newest annual period of the first statement that has one (income, then cash flow, then balance sheet).

    Each statement used to count its years from its own newest column, so a balance sheet with one more, nearly
    empty year (LOG.MC: Sep-25 shares only, no Sep-25 income statement) put every balance item one year off."""
    for stmt in statements:
        frame = _by_date(stmt)
        if not frame.empty:
            return frame.index[0]
    return None


def quarter_columns(quarterly: pd.DataFrame, rate: float) -> dict[str, float | str | None]:
    """The `N_QUARTERS` newest quarters of `QUARTERLY_CONCEPTS` one by one (`<concept>_Q0` newest, `PeriodEnd_Q*`),
    at `rate` like the rest of the statements; gaps stay gaps (aliases and composites apply). A quarter with none of
    them (TEF.MC: an empty Mar-26 column ahead of its statements) is skipped, so Q0 is the newest one reported."""
    qtr = _by_date(quarterly)
    if not qtr.empty:
        reported = [w for w in qtr.index if any(pd.notna(_value(qtr, w, c)) for c in S.QUARTERLY_CONCEPTS)]
        qtr = qtr.loc[reported]
    out: dict[str, float | str | None] = {}
    for i, q in enumerate(S.QUARTERS):
        when = qtr.index[i] if i < len(qtr.index) else None
        out[f"PeriodEnd_{q}"] = when.strftime("%Y-%m-%d") if when is not None else None
        for concept in S.QUARTERLY_CONCEPTS:
            value = _value(qtr, when, concept) if when is not None else np.nan
            out[f"{concept}_{q}"] = None if pd.isna(value) else value * rate
    return out


def _contiguous_quarters(frame: pd.DataFrame, n: int = 4) -> bool:
    """True if the n newest rows are consecutive quarters (no missing quarter in between)."""
    if len(frame.index) < n:
        return False
    gaps = [(frame.index[i] - frame.index[i + 1]).days for i in range(n - 1)]
    return all(QUARTER_GAP_DAYS[0] <= g <= QUARTER_GAP_DAYS[1] for g in gaps)


def _statement_columns(
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    ttm_stmt: pd.DataFrame,
    concepts: list[str],
    cfg: S.Settings,
    rate: float,
    ttm_method: str,
    anchor: pd.Timestamp | None = None,
) -> tuple[dict[str, float | None], dict[str, str], dict]:
    """Build `{concept}_{period}` values, `{period: period-end ISO date}` and provenance for one statement.

    Flow items (`ttm_method="sum"`): TTM is the sum of the four newest *contiguous* quarters, all
    present; else Yahoo's own TTM statement; else empty. Balance items (`"latest"`): newest
    quarter that has the value. Annual slots come from `annual_slots`, counted from `anchor` (`_annual_anchor`).

    Provenance: `aliases` (substitute rows used), and for flows `ttm_source` ("quarters" / "yahoo_ttm")
    plus `ttm_check` = relative gap between the quarterly sum and Yahoo's TTM for the first concept,
    when both exist (two independent routes agreeing is strong evidence the TTM is right). When they
    disagree by more than `cfg.suspect_ttm_gap`, the one closer to the latest annual report is used for
    the whole statement: half-yearly reporters often have a "quarter" that is a half-year (ACS.MC: Dec-25
    "quarter" 25.7bn vs ~12-14bn), so the sum counts six months twice; `ttm_source` = "yahoo_ttm_checked"
    (or "quarters_checked" when the sum is the closer one).
    """
    ann, qtr, ttm = _by_date(annual), _by_date(quarterly), _by_date(ttm_stmt)
    slots = annual_slots(ann, cfg.n_years, anchor)
    ends = {period: when.strftime("%Y-%m-%d") for period, when in slots.items()}

    def money(concept: str, raw: float) -> float | None:
        if pd.isna(raw):
            return None
        return raw if concept in S.NON_CURRENCY_CONCEPTS else raw * rate

    out: dict[str, float | None] = {}
    used: set[str] = set()
    meta: dict = {"ttm_source": None, "ttm_check": None}
    ttm_end = None
    use_quarters = True  # decided on the first concept, then kept for the whole statement
    # Balance items come from the newest of the last two quarters with total assets: a sparse newest
    # quarter (a few rows only) is not mixed into the full one before it. The other quarter fills only
    # what that one lacks (closer than falling back to the annual report).
    balance_qs = list(qtr.index[:2]) if ttm_method == "latest" else []
    if len(balance_qs) == 2 and pd.isna(_value(qtr, balance_qs[0], "Total Assets")):
        if pd.notna(_value(qtr, balance_qs[1], "Total Assets")):
            balance_qs.reverse()
    for i, concept in enumerate(concepts):
        for period in cfg.annual_years:
            when = slots.get(period)
            out[f"{concept}_{period}"] = money(concept, _value(ann, when, concept, used)) if when is not None else None

        value = np.nan
        if ttm_method == "sum":
            # Read only when used: reading records any alias it needed in `used` (the report notes).
            def yahoo_ttm(concept=concept) -> float:
                return _value(ttm, ttm.index[0], concept, used) if not ttm.empty else np.nan

            yahoo = np.nan
            if use_quarters and _contiguous_quarters(qtr):
                parts = [_value(qtr, when, concept, used) for when in qtr.index[:4]]
                if not any(pd.isna(x) for x in parts):
                    # An average (shares) over a year is the mean of its quarters; a flow is their sum.
                    value = float(np.mean(parts) if concept in S.AVERAGE_CONCEPTS else sum(parts))
                    yahoo = yahoo_ttm() if i == 0 else np.nan
                    if i == 0 and pd.notna(yahoo) and ttm.index[0] == qtr.index[0] and value != 0:
                        meta["ttm_check"] = abs(yahoo - value) / abs(value)
                        latest = _value(ann, slots.get(cfg.latest), concept) if slots.get(cfg.latest) else np.nan
                        if meta["ttm_check"] > cfg.suspect_ttm_gap and pd.notna(latest):
                            if abs(yahoo - latest) < abs(value - latest):
                                use_quarters, value = False, np.nan  # the sum double-counts: Yahoo's TTM instead
                                meta["ttm_source"] = "yahoo_ttm_checked"
                            else:  # Yahoo's TTM is the outlier (OXY): the sum stays, the dispute is settled
                                meta["ttm_source"] = "quarters_checked"
                    if use_quarters:
                        ttm_end = qtr.index[0]
                        meta["ttm_source"] = meta["ttm_source"] or "quarters"
            if pd.isna(value):
                value = yahoo if pd.notna(yahoo) else yahoo_ttm()
                if pd.notna(value):
                    ttm_end = ttm.index[0]
                    meta["ttm_source"] = meta["ttm_source"] or "yahoo_ttm"
        else:
            for when in balance_qs:
                value = _value(qtr, when, concept, used)
                if pd.notna(value):
                    ttm_end = when if ttm_end is None else max(ttm_end, when)
                    break
        out[f"{concept}_TTM"] = money(concept, value)
    if ttm_end is not None:
        ends["TTM"] = ttm_end.strftime("%Y-%m-%d")
    meta["aliases"] = used
    return out, ends, meta


def split_factors(splits: pd.Series, ends: dict[str, str | None]) -> dict[str, float]:
    """Cumulative split ratio after each period end (only periods with a later split)."""
    if splits is None or len(splits) == 0:
        return {}
    idx = pd.to_datetime(splits.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(np.asarray(splits, dtype=float), index=idx)
    s = s[(s > 0) & (s != 1)]
    out = {}
    for period, end in ends.items():
        if end:
            later = s[s.index > pd.Timestamp(end)]
            if len(later):
                out[period] = float(np.prod(later.to_numpy()))
    return out


def _apply_splits(row: dict, factors: dict[str, float], ticker: str) -> list[str]:
    """Restate share counts to today's share basis. Returns the periods that were changed.

    Yahoo usually reports old share counts as they were, but sometimes already restated; for each
    value we keep whichever of raw / adjusted is closer to today's share count.
    """
    ref = row.get("sharesOutstanding") or row.get("impliedSharesOutstanding")
    changed = []
    for period, factor in factors.items():
        for concept in S.SHARE_COUNT_CONCEPTS:
            key = f"{concept}_{period}"
            raw = row.get(key)
            if raw is None or pd.isna(raw) or raw <= 0:
                continue
            adjusted = raw * factor
            if ref and ref > 0 and abs(np.log(raw / ref)) < abs(np.log(adjusted / ref)):
                continue  # already on today's basis
            row[key] = adjusted
            changed.append(period)
    if changed:
        log.info("%s: share counts split-adjusted for %s", ticker, ", ".join(sorted(set(changed))))
    return sorted(set(changed))


def _align_share_basis(
    row: dict, periods: list[str], ticker: str, splits: dict[str, float] | None = None
) -> float | None:
    """Put statement share counts on the same basis as the quoted price.

    For ADRs the price is per depositary share while statements count ordinary shares (PDD:
    1 ADR = 4 shares), which inflates price x shares - and every per-share figure - by the ratio.
    Yahoo's market cap / price gives the share count the price refers to; when the statement count
    differs by more than 25% all share counts are divided by that ratio. Returns the ratio applied.
    A stock split after the latest statement (`splits`, from `split_factors`) explains a gap of
    its own: that is left to `_apply_splits`, not mistaken for an ADR ratio.
    """
    price, cap = row.get("currentPrice"), row.get("marketCap")
    period, latest = next(((p, row.get(f"Ordinary Shares Number_{p}")) for p in periods
                           if pd.notna(row.get(f"Ordinary Shares Number_{p}"))), (None, None))  # fmt: skip
    if not (price and cap and latest) or price <= 0 or cap <= 0 or latest <= 0:
        return None
    ratio = latest / (cap / price)
    if any(0.8 <= ratio * f <= 1.25 for f in (1.0, (splits or {}).get(period, 1.0))):
        return None
    for p in periods:
        for concept in S.SHARE_COUNT_CONCEPTS:
            value = row.get(f"{concept}_{p}")
            if value is not None and pd.notna(value):
                row[f"{concept}_{p}"] = value / ratio
    log.info("%s: statement shares are %.4gx the quoted share basis (ADR / share class); rescaled", ticker, ratio)
    return ratio


def economic_share_basis(row: dict, periods: list[str], ticker: str, ratio: float, sister_shares) -> float | None:
    """Count shares in units of the quoted class when another class has different economic rights.

    Yahoo counts all classes as equal (Swire Pacific: 0.78bn A + 2.85bn B, but one A = five B), which
    distorts market cap and every per-share figure. Economic shares = own shares + sister shares x
    ratio; statement counts are rescaled to that and market cap is rebuilt. Returns the factor applied.
    """
    own, price = row.get("sharesOutstanding"), row.get("currentPrice")
    latest = next((row.get(f"Ordinary Shares Number_{p}") for p in periods
                   if pd.notna(row.get(f"Ordinary Shares Number_{p}"))), None)  # fmt: skip
    if not (own and sister_shares and latest) or min(own, sister_shares, latest) <= 0:
        log.warning("%s: share-class data missing; share count left as Yahoo reports it", ticker)
        return None
    economic = own + sister_shares * ratio
    factor = economic / latest
    for p in periods:
        for concept in S.SHARE_COUNT_CONCEPTS:
            value = row.get(f"{concept}_{p}")
            if value is not None and pd.notna(value):
                row[f"{concept}_{p}"] = value * factor
    row["sharesOutstanding"] = row["impliedSharesOutstanding"] = economic
    if price and price > 0:
        row["marketCap"] = price * economic
    log.info("%s: shares counted in this class's units (%.4g economic shares, x%.3f)", ticker, economic, factor)
    return factor


def _sister_shares(ticker: str) -> float | None:
    try:
        return (yf.Ticker(ticker).info or {}).get("sharesOutstanding")
    except Exception as exc:  # noqa: BLE001 - any Yahoo failure just means "unknown"
        log.warning("%s: could not read share count (%s)", ticker, exc)
        return None


def _naive_series(series: pd.Series) -> pd.Series:
    """Numeric values on a tz-naive date index, oldest first, non-positive and missing values left out."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    idx = pd.to_datetime(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(pd.to_numeric(np.asarray(series), errors="coerce"), index=idx).sort_index()
    return s[s > 0]


def yearly_shares(points: pd.Series, splits: pd.Series, ref: float | None) -> dict[str, float]:
    """One share count per calendar year from Yahoo's long share series (`get_shares_full`), on today's basis.

    Each point is split-adjusted like the statements (`_apply_splits`: raw or adjusted, whichever is closer to `ref`,
    as Yahoo mixes both); points more than 25% off the median of their nine neighbours are dropped (Yahoo mixes in
    counts on another basis: Swire's 3.6bn of every class among 1.3bn economic shares); a year's count is the median of
    its last quarter's points (the whole year's if that quarter has none), so one-day spikes drop out too. The whole
    series is then rescaled to `ref` when its latest year is more than 25% off (an ADR or share-class basis, as
    `_align_share_basis`). The current year is the latest median so far."""
    s = _naive_series(points)
    if s.empty:
        return {}
    sp = _naive_series(splits)
    sp = sp[sp != 1]
    if len(sp):
        factor = np.array([float(np.prod(sp[sp.index > when].to_numpy())) for when in s.index])
        adjusted = s.to_numpy() * factor
        if ref and ref > 0:
            raw_off, adj_off = np.abs(np.log(s.to_numpy() / ref)), np.abs(np.log(adjusted / ref))
            s = pd.Series(np.where(adj_off < raw_off, adjusted, s.to_numpy()), index=s.index)
        else:
            s = pd.Series(adjusted, index=s.index)
    local = s.rolling(9, center=True, min_periods=1).median()
    s = s[(s / local).between(0.8, 1.25)]
    out = {}
    for year, pts in s.groupby(s.index.year):
        late = pts[pts.index.quarter == 4]
        out[str(year)] = float((late if len(late) else pts).median())
    if not out:  # every point zero or unreadable (OIO): no history, not a failed fetch
        return {}
    latest = out[max(out)]
    if ref and ref > 0 and not 0.8 <= latest / ref <= 1.25:
        out = {y: v * ref / latest for y, v in out.items()}
    return out


def yearly_dividends(dividends: pd.Series, today: pd.Timestamp) -> dict[str, float]:
    """Dividends per share paid in each complete calendar year (quote currency, split-adjusted as Yahoo serves them);
    the current year is left out, as a part year would read as a cut."""
    s = _naive_series(dividends)
    s = s[s.index.year < today.year]
    return {str(year): float(v) for year, v in s.groupby(s.index.year).sum().items()}


def dividend_payments(dividends: pd.Series) -> dict[str, float]:
    """Every dividend payment Yahoo keeps, {"YYYY-MM-DD": per share} (quote currency, split-adjusted as served), so a
    special can be told from a raise cut back later, which yearly sums cannot (`report.dividend_record`)."""
    s = _naive_series(dividends)
    return {f"{d:%Y-%m-%d}": float(v) for d, v in s.items()}


def declared_currency_payments(payments: dict[str, float], quote: str, declared: str) -> dict[str, float] | None:
    """`payments` in the reporting currency `declared` at each payment day's rate (`fx.rate_on`), for a company quoted
    in another currency (a Hong Kong share paying RMB dividends in HKD): the exchange rate then does not read as a
    raise or a cut. A payment older than Yahoo's exchange rates takes the oldest rate known (the record stays whole, in
    one currency). None when the two are the same currency (a minor unit such as pence differs only by scale) or no
    rate is known."""
    base = lambda c: fx.MINOR_UNITS.get(c, (c, 1))[0]  # noqa: E731
    if not payments or not quote or not declared or base(quote) == base(declared):
        return None
    ratios = {}
    for day in payments:
        q, d = fx.rate_on(quote, day), fx.rate_on(declared, day)
        ratios[day] = q / d if q and d else None
    known = [r for r in ratios.values() if r is not None]
    if not known:
        return None
    first = known[0]  # payments are oldest first: earlier gaps are before the rates begin
    return {day: amount * (ratios[day] if ratios[day] is not None else first) for day, amount in payments.items()}


HOLDERS_SHOWN = 10  # largest holders kept per company (`top_holders`)
INSIDERS_SHOWN = 3  # plus the largest insiders, however small (a founder's 0.5% says more than a tenth index fund)
# Yahoo writes a person as "Surname (Given names)"; a roster name without a position in that form is an
# organisation: in the UK the roster also carries the major shareholders' notifications (BlackRock, a family's
# holding company).
_PERSON = re.compile(r"^[^()]+\([^()]+\)$")


def _holder_date(value) -> str | None:
    """A holder table's date as `YYYY-MM-DD`; Yahoo sometimes serves epoch seconds instead of a date."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (int, float, np.number)):
        when = pd.to_datetime(float(value), unit="s", errors="coerce")
    else:
        when = pd.to_datetime(value, errors="coerce")
    if pd.isna(when):
        return None
    return when.tz_localize(None).strftime("%Y-%m-%d") if when.tzinfo else when.strftime("%Y-%m-%d")


def _num(value) -> float | None:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) else None


def top_holders(
    institutions: pd.DataFrame, funds: pd.DataFrame, roster: pd.DataFrame, shares_now: float | None,
    splits: pd.Series | None = None, n: int = HOLDERS_SHOWN,
) -> list[dict]:  # fmt: skip
    """The `n` largest holders from Yahoo's three holder tables plus the `INSIDERS_SHOWN` largest insiders, largest first, each as {"name", "type", "pct" (% of
    the shares), "change" (% change of the holding since the previous filing), "position", "latest" (an insider's
    latest transaction), "latest_date", "date"}. Types: "Institution" and "Fund" (Yahoo's `pctHeld`; its `Value` is
    left out, pence for London lines), "Insider" (a roster name with a position, or a person's name) and "Major
    holder" (a roster organisation without a position). A roster holding has no %: its shares (direct plus indirect) over `shares_now`, the
    count on the quoted price's basis, after the splits since the holding was reported; a share above 100% is a
    count on another basis and is dropped. Funds are often held inside the institutions, so the rows do not add up."""
    rows: list[dict] = []
    for frame, kind in ((institutions, "Institution"), (funds, "Fund")):
        if frame is None or frame.empty or "Holder" not in frame.columns:
            continue
        for _, r in frame.iterrows():
            pct = _num(r.get("pctHeld"))
            if pct is None or pct <= 0 or not str(r.get("Holder") or "").strip():
                continue
            change = _num(r.get("pctChange"))
            rows.append({"name": str(r["Holder"]).strip(), "type": kind, "pct": pct * 100,
                         "change": change * 100 if change is not None else None, "position": None,
                         "latest": None, "latest_date": None, "date": _holder_date(r.get("Date Reported"))})  # fmt: skip
    if roster is not None and not roster.empty and "Name" in roster.columns and shares_now and shares_now > 0:
        sp = _naive_series(splits) if splits is not None else pd.Series(dtype=float)
        sp = sp[sp != 1]
        for _, r in roster.iterrows():
            name = str(r.get("Name") or "").strip()
            direct, indirect = _num(r.get("Shares Owned Directly")), _num(r.get("Shares Owned Indirectly"))
            held = (direct or 0) + (indirect or 0)
            if not name or held <= 0:
                continue
            date = _holder_date(r.get("Position Direct Date")) or _holder_date(r.get("Latest Transaction Date"))
            if date and len(sp):
                held *= float(np.prod(sp[sp.index > pd.Timestamp(date)].to_numpy()))
            pct = held / shares_now * 100
            if pct > 100:
                continue
            position = str(r.get("Position") or "").strip() or None
            kind = "Insider" if position or _PERSON.match(name) else "Major holder"
            latest = str(r.get("Most Recent Transaction") or "").strip() or None
            rows.append({"name": name, "type": kind, "pct": pct, "change": None, "position": position,
                         "latest": latest, "latest_date": _holder_date(r.get("Latest Transaction Date")),
                         "date": date})  # fmt: skip
    best: dict[tuple[str, str], dict] = {}
    for row in rows:  # one row per holder and table (Yahoo repeats a name now and then)
        key = (row["name"].lower(), row["type"])
        if key not in best or row["pct"] > best[key]["pct"]:
            best[key] = row
    ranked = sorted(best.values(), key=lambda r: -r["pct"])
    insiders = [r for r in ranked if r["type"] == "Insider"][:INSIDERS_SHOWN]
    out = ranked[:n] + [r for r in insiders if r not in ranked[:n]]
    for row in out:
        row["pct"] = round(row["pct"], 3)
        row["change"] = round(row["change"], 1) if row["change"] is not None else None
    return out


def holder_columns(obj, splits: pd.Series | None, shares_now: float | None) -> dict:
    """`Holders` (JSON list from `top_holders`, "[]" when Yahoo has none), `Institutions_Count` and `Holders_At` for one
    `yf.Ticker`: one Yahoo request serves the three tables. Nothing at all when the request failed, so a cached
    figure is not overwritten by a blank and a missing `Holders_At` still means "never fetched"."""
    try:
        institutions, funds = obj.institutional_holders, obj.mutualfund_holders
        roster, breakdown = obj.insider_roster_holders, obj.major_holders
    except Exception as exc:
        if any(h in str(exc).lower() for h in RATE_LIMIT_HINTS):
            raise
        log.debug("holders unavailable: %s", exc)
        return {}
    count = None
    if isinstance(breakdown, pd.DataFrame) and "institutionsCount" in breakdown.index:
        count = _num(breakdown.loc["institutionsCount"].iloc[0])
    holders = top_holders(institutions, funds, roster, shares_now, splits)
    return {"Holders": json.dumps(holders, ensure_ascii=False), "Institutions_Count": count,
            "Holders_At": pd.Timestamp.now().strftime("%Y-%m-%d")}  # fmt: skip


def parse_splits(text) -> pd.Series:
    """The cached `Splits` column ("2024-06-10 x10; ...") back as a split series."""
    if not isinstance(text, str) or not text.strip():
        return pd.Series(dtype=float)
    pairs = [p.strip().split(" x") for p in text.split(";") if " x" in p]
    return pd.Series([float(r) for _, r in pairs], index=pd.to_datetime([d for d, _ in pairs]))


def fetch_holders(
    df: pd.DataFrame, tickers: list[str], cfg: S.Settings, save_to=None, every: int = 100
) -> pd.DataFrame:
    """The largest holders alone (`holder_columns`) for cached `tickers`, written into `df` without touching any
    other column: one light request per company, no statements. Saved to `save_to` every `every` companies, so a
    stopped run keeps what it fetched; after a rate limit it waits and tries again (`cfg.fetch_retries`)."""
    out = df.copy()
    for col in ("Holders", "Institutions_Count", "Holders_At"):
        if col not in out.columns:
            out[col] = None
    out["Holders"] = out["Holders"].astype(object)
    out["Holders_At"] = out["Holders_At"].astype(object)
    where = {t: i for i, t in zip(out.index, out["Ticker"], strict=True)}
    todo = [t for t in tickers if t in where]
    for k, ticker in enumerate(todo, 1):
        i = where[ticker]
        price, cap = _num(out.at[i, "currentPrice"]), _num(out.at[i, "marketCap"])
        shares = cap / price if price and cap and price > 0 and cap > 0 else _num(out.at[i, "sharesOutstanding"])
        splits = parse_splits(out.at[i, "Splits"]) if "Splits" in out.columns else None
        log.info("[%d/%d] %s holders", k, len(todo), ticker)
        for attempt in range(1, cfg.fetch_retries + 1):
            try:
                cols = holder_columns(yf.Ticker(ticker), splits, shares)
                for col, value in cols.items():
                    out.at[i, col] = value
                break
            except Exception as exc:
                log.warning("%s: holders attempt %d/%d failed (%s)", ticker, attempt, cfg.fetch_retries, exc)
                time.sleep(30 * 2**attempt)
        time.sleep(cfg.fetch_delay_s)
        if save_to is not None and k % every == 0:
            save_raw(out, save_to)
    if save_to is not None:
        save_raw(out, save_to)
    return out


def _patch_capex_and_fcf(data: dict, periods: list[str], ticker: str) -> list[str]:
    """Fill missing CapEx (PPE purchases, else median margin proxy) and keep FCF = OCF + CapEx.

    Returns the periods whose CapEx is an estimate (margin proxy), so reports can say so.
    """
    estimated = []
    margins = []
    for p in periods:
        cap, rev = data.get(f"Capital Expenditure_{p}"), data.get(f"Total Revenue_{p}")
        if pd.notna(cap) and cap != 0 and pd.notna(rev) and rev > 0:
            margins.append(abs(cap) / rev)
    median_margin = float(np.median(margins)) if margins else 0.05

    for p in periods:
        capex = data.get(f"Capital Expenditure_{p}")
        if capex is None or pd.isna(capex) or capex == 0:
            ppe = data.get(f"Purchase Of PPE_{p}")
            rev = data.get(f"Total Revenue_{p}")
            if ppe is not None and pd.notna(ppe) and ppe != 0:
                data[f"Capital Expenditure_{p}"] = ppe
            elif rev is not None and pd.notna(rev) and rev > 0:
                data[f"Capital Expenditure_{p}"] = -(rev * median_margin)
                estimated.append(p)
                log.debug("%s %s: CapEx proxy at %.1f%% of revenue", ticker, p, median_margin * 100)

    for p in periods:
        capex, ocf = data.get(f"Capital Expenditure_{p}"), data.get(f"Operating Cash Flow_{p}")
        if not (pd.notna(capex) and capex != 0 and pd.notna(ocf) and ocf != 0):
            continue
        fcf_calc = ocf + capex
        fcf = data.get(f"Free Cash Flow_{p}")
        if fcf is None or pd.isna(fcf):
            data[f"Free Cash Flow_{p}"] = fcf_calc
        elif abs(fcf - fcf_calc) > 1.0 and (abs(fcf - ocf) < 1.0 or abs(fcf - fcf_calc) > abs(capex) * 0.9):
            # Yahoo reported FCF == OCF (CapEx was missing upstream) or ignored the patched CapEx.
            data[f"Free Cash Flow_{p}"] = fcf_calc
    return estimated


def estimate_revisions(trend: pd.DataFrame) -> dict[str, float | None]:
    """% change of the analysts' consensus EPS for next fiscal year (else this one) over 30 and 90 days.

    A sharp cut says the market has a reason for a lower price; stable or rising estimates while the
    price falls is what a mood looks like. Measured against the absolute base, so a loss works too.
    """
    out: dict[str, float | None] = {"EPS_Rev_30d": None, "EPS_Rev_90d": None}
    for period in ("+1y", "0y"):
        if trend is None or trend.empty or period not in trend.index:
            continue
        row = trend.loc[period]
        now = pd.to_numeric(row.get("current"), errors="coerce")
        for days in (30, 90):
            base = pd.to_numeric(row.get(f"{days}daysAgo"), errors="coerce")
            if pd.notna(now) and pd.notna(base) and abs(base) > 0.01:
                out[f"EPS_Rev_{days}d"] = float((now - base) / abs(base) * 100)
        if out["EPS_Rev_90d"] is not None:
            break
    return out


def _similar(a, b, tol: float = 0.01) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= tol


def fetch_ticker(ticker: str, cfg: S.Settings, rates: dict[str, float]) -> dict | None:
    """Fetch and normalise one ticker (values in USD). Returns None if Yahoo has no data."""
    obj = yf.Ticker(ticker)
    info = obj.info
    if not info or "symbol" not in info:
        return None

    currency = info.get("currency", "USD")
    fin_currency = S.FINANCIAL_CURRENCY_OVERRIDES.get(ticker, info.get("financialCurrency", currency))
    fin_rate, price_rate = fx.usd_rate(rates, fin_currency), fx.usd_rate(rates, currency)
    # A minor-unit quote (London pence): price in pence, market cap in pounds.
    cap_rate = fx.usd_rate(rates, fx.MINOR_UNITS.get(currency, (currency, 1))[0])
    if fin_rate is None or price_rate is None:  # never read unknown-currency figures as dollars
        log.warning("%s: no exchange rate for %s / %s; skipped", ticker, fin_currency, currency)
        return None

    row: dict = {"Ticker": ticker, "Fin_Rate": fin_rate}  # `_restate_fx` moves statements to a later rate
    for key in S.INFO_KEYS:
        value = info.get(key)
        if value is not None and currency != "USD":
            if key in PRICE_KEYS:
                value *= price_rate
            elif key in CAP_KEYS:
                value *= cap_rate
        row[key] = value
    if ticker in S.NAME_OVERRIDES:
        row["shortName"] = S.NAME_OVERRIDES[ticker]

    q_income = _safe(lambda: obj.quarterly_income_stmt)
    row["last_quarter"] = q_income.columns[0].strftime("%Y-%m-%d") if not q_income.empty else "N/A"

    income, balance, cashflow = (_safe(lambda: obj.income_stmt), _safe(lambda: obj.balance_sheet),
                                 _safe(lambda: obj.cashflow))  # fmt: skip
    raw_mkt_cap = info.get("marketCap")
    if fin_currency == "USD" and currency != "USD" and raw_mkt_cap and raw_mkt_cap > 0:
        local_rate = fx.usd_rate(rates, currency)
        if local_rate is not None and local_rate < 0.20 and not income.empty:
            sample_ni = None
            if "Net Income" in income.index:
                try:
                    val = float(income.loc["Net Income"].iloc[0])
                    if pd.notna(val):
                        sample_ni = val
                except Exception:
                    pass
            sample_rev = None
            if "Total Revenue" in income.index:
                try:
                    val = float(income.loc["Total Revenue"].iloc[0])
                    if pd.notna(val):
                        sample_rev = val
                except Exception:
                    pass
            is_local = False
            if sample_ni is not None and abs(sample_ni) / raw_mkt_cap > 0.02:
                is_local = True
            elif sample_rev is not None and sample_rev / raw_mkt_cap > 0.15:
                is_local = True
            if is_local:
                log.warning(
                    "%s: Yahoo reported financialCurrency='USD' but statement scale matches %s; overriding to %s",
                    ticker,
                    currency,
                    currency,
                )
                fin_currency = currency
                fin_rate = fx.usd_rate(rates, fin_currency)
                row["Fin_Rate"] = fin_rate

    # The currency the statements were read in (Yahoo leaves the field out for some listings, or misreports it).
    row["financialCurrency"] = fx.MINOR_UNITS.get(fin_currency, (fin_currency, 1))[0]
    anchor = _annual_anchor(income, cashflow, balance)  # one Y0 for the three statements
    income_cols, income_ends, income_meta = _statement_columns(
        income, q_income, _safe(lambda: obj.ttm_income_stmt), S.CONCEPTS_INCOME, cfg, fin_rate, "sum", anchor,
    )  # fmt: skip
    balance_cols, balance_ends, balance_meta = _statement_columns(
        balance, _safe(lambda: obj.quarterly_balance_sheet), pd.DataFrame(),
        S.CONCEPTS_BALANCE, cfg, fin_rate, "latest", anchor,
    )  # fmt: skip
    cash_cols, cash_ends, cash_meta = _statement_columns(
        cashflow, _safe(lambda: obj.quarterly_cashflow), _safe(lambda: obj.ttm_cashflow),
        S.CONCEPTS_CASHFLOW, cfg, fin_rate, "sum", anchor,
    )  # fmt: skip
    row.update(income_cols)
    row.update(balance_cols)
    row.update(cash_cols)
    row.update(quarter_columns(q_income, fin_rate))

    ends = {**balance_ends, **cash_ends, **income_ends}  # income statement dates win
    for period in cfg.fiscal_years:
        row[f"PeriodEnd_{period}"] = ends.get(period)

    # Rate at each period end relative to the rate used above: multiplying a stored value by it gives
    # USD at the rate of that time, so growth is measured in USD (a falling currency counts against it).
    for period in cfg.fiscal_years:
        then = fx.rate_on(fin_currency, row[f"PeriodEnd_{period}"])
        row[f"FX_Hist_{period}"] = then / fin_rate if then and fin_rate else None
    missing_fx = [p for p in cfg.fiscal_years if row[f"PeriodEnd_{p}"] and row[f"FX_Hist_{p}"] is None]
    if missing_fx:
        log.warning("%s: no %s rate history for %s; those periods use today's rate", ticker, fin_currency, missing_fx)

    row["TTM_Source"] = income_meta["ttm_source"]
    row["TTM_Check_Gap"] = income_meta["ttm_check"]
    aliases = income_meta["aliases"] | balance_meta["aliases"] | cash_meta["aliases"]
    row["Aliases_Used"] = "; ".join(sorted(aliases)) or None

    splits = _safe_series(lambda: obj.splits)
    ends_by_period = {p: row.get(f"PeriodEnd_{p}") for p in cfg.fiscal_years}
    factors = split_factors(splits, ends_by_period)
    row["Share_Basis_Ratio"] = _align_share_basis(row, ["TTM", *cfg.annual_years], ticker, factors)
    row["Share_Class_Factor"] = None
    if link := share_classes().get(ticker):  # after the ADR check, which trusts Yahoo's market cap
        sister, ratio = link
        row["Share_Class_Factor"] = economic_share_basis(
            row, ["TTM", *cfg.annual_years], ticker, ratio, _sister_shares(sister)
        )
        row["Share_Class_Sister"], row["Share_Class_Ratio"] = sister, ratio
    adjusted = _apply_splits(row, factors, ticker)
    row["Splits"] = "; ".join(f"{d:%Y-%m-%d} x{r:g}" for d, r in splits.items()) if len(splits) else None
    row["Split_Adjusted_Periods"] = ", ".join(adjusted) or None
    # Long per-share record (shown only): share count and dividends per share by calendar year, as JSON
    # {"<year>": value} so no column name carries a year. The share count on the quoted price's basis.
    price, cap = row.get("currentPrice"), row.get("marketCap")
    basis = cap / price if price and cap and price > 0 and cap > 0 else row.get("sharesOutstanding")
    shares_hist = yearly_shares(_safe_series(lambda: obj.get_shares_full(start=HISTORY_START)), splits, basis)
    dividends = _safe_series(lambda: obj.dividends)
    dps_hist = yearly_dividends(dividends, pd.Timestamp.now())
    payments = dividend_payments(dividends)
    payments_fx = declared_currency_payments(payments, currency, fin_currency)
    row["Shares_Hist"] = json.dumps(shares_hist) if shares_hist else None
    row["DPS_Hist"] = json.dumps(dps_hist) if dps_hist else None
    row["DPS_Payments"] = json.dumps(payments) if payments else None
    row["DPS_Payments_FX"] = json.dumps(payments_fx) if payments_fx else None
    row |= holder_columns(obj, splits, basis)  # the largest holders (shown only), one more request

    row["CapEx_Estimated"] = ", ".join(_patch_capex_and_fcf(row, cfg.fiscal_years, ticker)) or None
    row |= estimate_revisions(_safe(lambda: obj.eps_trend))

    newest_annual = row.get(f"PeriodEnd_{cfg.latest}")
    age_days = (pd.Timestamp.now().normalize() - pd.Timestamp(newest_annual)).days if newest_annual else None
    row["Stale_Annual"] = age_days is None or age_days > STALE_ANNUAL_DAYS

    # A TTM that duplicates the latest annual period adds nothing; drop it and flag it.
    same_date = row.get("PeriodEnd_TTM") is not None and row.get("PeriodEnd_TTM") == newest_annual
    same_numbers = _similar(row.get("Total Revenue_TTM"), row.get(f"Total Revenue_{cfg.latest}")) and _similar(
        row.get("Net Income_TTM"), row.get(f"Net Income_{cfg.latest}")
    )
    row["TTM_Identical_To_Latest"] = bool(same_date or same_numbers)
    if row["TTM_Identical_To_Latest"]:
        for key in [k for k in row if k.endswith("_TTM") and not k.startswith("PeriodEnd")]:
            del row[key]
    row["Schema_Version"] = S.SCHEMA_VERSION
    row["Fields_Version"] = S.FIELDS_VERSION
    row["Fetched_At"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return row


FAILED = object()  # `_fetch_all` result: every attempt failed (None = Yahoo has no data)


def _fetch_all(todo: list[str], cfg: S.Settings, rates: dict[str, float]) -> list:
    """`fetch_ticker` over `todo` on `cfg.fetch_workers` threads, results in `todo`'s order (row, None or
    FAILED). After the first rate limit the threads take turns, one ticker at a time, for the rest of the batch."""
    limited = threading.Event()
    one_at_a_time = threading.Lock()

    def attempt(ticker: str):
        if not limited.is_set():
            return fetch_ticker(ticker, cfg, rates)
        with one_at_a_time:
            return fetch_ticker(ticker, cfg, rates)

    def one(item: tuple[int, str]):
        i, ticker = item
        log.info("[%d/%d] %s", i, len(todo), ticker)
        result = FAILED
        for n in range(1, cfg.fetch_retries + 1):
            try:
                result = attempt(ticker)
                if result is None:
                    log.warning("%s: no data on Yahoo, skipped", ticker)
                break
            except Exception as exc:
                rate_limited = any(h in str(exc).lower() for h in RATE_LIMIT_HINTS)
                if rate_limited and not limited.is_set():
                    limited.set()
                    log.warning("Rate limited: fetching one ticker at a time for the rest of this batch")
                log.warning("%s: attempt %d/%d failed (%s)", ticker, n, cfg.fetch_retries, exc)
                time.sleep((30 if rate_limited else cfg.fetch_delay_s) * 2**n)
        time.sleep(cfg.fetch_delay_s)
        return result

    workers = max(1, cfg.fetch_workers)
    if workers == 1 or len(todo) < 2:
        return [one(item) for item in enumerate(todo, 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, enumerate(todo, 1)))


def fetch_universe(
    tickers: list[str],
    cfg: S.Settings,
    existing: pd.DataFrame | None = None,
    only_new: bool = False,
) -> pd.DataFrame:
    """Fetch every ticker (or only those missing from `existing`) and return the merged frame."""
    rates = fx.get_rates(cfg.data_dir / "fx_rates.json")
    inflation.load(cfg.data_dir / "inflation.json", live=True)  # IMF CPI, refreshed monthly (`metrics.inflation`)
    todo = tickers
    if only_new and existing is not None and not existing.empty:
        have = set(existing["Ticker"])
        todo = [t for t in tickers if t not in have]
        log.info("%d new tickers to fetch (%d cached)", len(todo), len(tickers) - len(todo))

    results = _fetch_all(todo, cfg, rates)
    rows = [r for r in results if isinstance(r, dict)]
    no_data = [t for t, r in zip(todo, results, strict=True) if r is None]
    failed = [t for t, r in zip(todo, results, strict=True) if r is FAILED]
    if no_data or failed:
        log.warning(
            "Fetch summary: %d ok, %d without data (%s), %d failed (%s)",
            len(rows), len(no_data), ", ".join(no_data) or "-", len(failed), ", ".join(failed) or "-",
        )  # fmt: skip

    for row in rows:
        row["FX_Source"] = fx.status()
    new = pd.DataFrame(rows)
    if not new.empty:
        new = _attach_price_stats(new, list(new["Ticker"]))
    for col in PCT_COLUMNS:
        if col in new.columns:
            new[col] = pd.to_numeric(new[col], errors="coerce").mul(100).round(2)

    if existing is not None and not existing.empty:
        keep = existing[~existing["Ticker"].isin(new.get("Ticker", []))]
        new = pd.concat([keep, new], ignore_index=True)
    return new


def _price_history(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One year of daily closes, lows and volumes (split-adjusted, quote currency), one batch request."""
    data = yf.download(tickers, period="1y", interval="1d", progress=False, auto_adjust=False)
    close, low, volume = data["Close"], data["Low"], data["Volume"]
    if isinstance(close, pd.Series):  # a single ticker
        close, low, volume = (s.to_frame(tickers[0]) for s in (close, low, volume))
    return close, low, volume


def total_return_history(tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """Daily closes adjusted for dividends and splits (quote currency) since `start`, one batch request."""
    data = yf.download(tickers, start=f"{start:%Y-%m-%d}", interval="1d", progress=False, auto_adjust=True)
    close = data["Close"]
    return close.to_frame(tickers[0]) if isinstance(close, pd.Series) else close


def total_and_price_history(tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """Daily closes (quote currency) since `start`, one batch request, columns (kind, ticker): `Total` adjusted for
    dividends and splits (Yahoo's adjusted close), `Price` for splits only. The gap between the two is the dividends,
    so the forward test can store what they added and check it (`forward._rows`)."""
    data = yf.download(tickers, start=f"{start:%Y-%m-%d}", interval="1d", progress=False, auto_adjust=False)

    def closes(column: str) -> pd.DataFrame:
        close = data[column]
        return close.to_frame(tickers[0]) if isinstance(close, pd.Series) else close

    return pd.concat({"Total": closes("Adj Close"), "Price": closes("Close")}, axis=1)


PRICE_BREAK = 3.0  # a one-day move this large (either way) is a basis change, not a price move


def price_stats(
    close: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame | None = None, months: int = 6
) -> pd.DataFrame:
    """Per ticker: last close, % change over `months` (momentum), % above the 52-week low (0 = at the low) and, with
    `volume`, `Daily_Turnover_Shares`: the median daily traded value (close x volume, traded days of the last year) over
    today's close, so the price in USD turns it into dollars (`Traded_Value_USD`, the highlight lists' liquidity floor).

    All are ratios in the quote currency, so FX does not enter. Momentum is empty when the
    history is shorter than `months` (recent listings). Yahoo glitches are left out: days without
    volume (MEGP.L: untraded days quoted in pounds instead of pence) and everything before a one-day
    move of `PRICE_BREAK`x (SIVE.ST: a reverse split missing from the split history).
    """
    if volume is not None:
        traded = volume.reindex_like(close) > 0
        traded |= ~traded.any()  # a ticker Yahoo gives no volume for at all keeps every day
        close, low = close.where(traded), low.where(traded)
    close = close.ffill()
    step = close / close.shift()
    breaks = ((step >= PRICE_BREAK) | (step <= 1 / PRICE_BREAK)).cumsum()
    since = breaks == breaks.iloc[-1]  # from the last break on (all of it when there is none)
    close, low = close.where(since), low.where(since)
    last = close.iloc[-1]
    turnover = pd.Series(np.nan, index=close.columns)
    if volume is not None:
        shares = volume.reindex_like(close)
        value = (close * shares).where((shares > 0) & since)  # no volume at all: unknown, not illiquid
        turnover = value.loc[close.index[-1] - pd.DateOffset(years=1) :].median() / last.where(last > 0)
    past = close.loc[: close.index[-1] - pd.DateOffset(months=months)]
    base = past.iloc[-1] if len(past) else pd.Series(np.nan, index=close.columns)
    low_52w = low.loc[close.index[-1] - pd.DateOffset(years=1) :].min().where(lambda s: s > 0)
    return pd.DataFrame(
        {
            "Last_Close": last,
            "Momentum_6M": (last / base.where(base > 0) - 1) * 100,
            "Above_52W_Low": (last / low_52w - 1) * 100,
            "Daily_Turnover_Shares": turnover,
        }
    )


PRICE_STATS = ("Momentum_6M", "Above_52W_Low", "Daily_Turnover_Shares")  # stored from `price_stats`


def _store_prices(df: pd.DataFrame, mask: pd.Series, stats: pd.DataFrame) -> pd.DataFrame:
    """Momentum, % above the 52-week low and turnover (reported only, never ranked) for the `mask` rows."""
    stored = [c for c in PRICE_STATS if c in stats.columns]
    added = [c for c in stored if c not in df.columns]
    out = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=added)], axis=1)  # one insert, not many
    for col in stored:
        out.loc[mask, col] = out.loc[mask, "Ticker"].map(stats[col]).astype(float)
    return out


def _attach_price_stats(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Add momentum and % above the 52-week low for `tickers`; leaves them empty if Yahoo fails."""
    if not tickers:
        return df
    try:
        close, low, volume = _price_history(tickers)
        stats = price_stats(close, low, volume)
    except Exception as exc:
        log.warning("Price history unavailable (%s); momentum and 52-week low left empty", exc)
        return df
    return _store_prices(df, df["Ticker"].isin(tickers), stats)


MONEY_CONCEPTS = tuple(
    c for c in (*S.CONCEPTS_INCOME, *S.CONCEPTS_BALANCE, *S.CONCEPTS_CASHFLOW) if c not in S.NON_CURRENCY_CONCEPTS
)


def _restate_fx(df: pd.DataFrame, idx, rates: dict[str, float], cfg: S.Settings) -> bool:
    """Convert one cached row's statements to today's rate, as its new price is (in place).

    A full fetch converts statements at that day's rate (`Fin_Rate`); a price refresh converts the
    price at today's. Without this, FCF / EV and net debt mix two rates and every company in a moved
    currency shifts against the rest. `FX_Hist_*` move the other way, so USD of the time is unchanged.
    Returns False for rows fetched before `Fin_Rate` was stored (left as they are).
    """
    old = pd.to_numeric(df.at[idx, "Fin_Rate"], errors="coerce") if "Fin_Rate" in df.columns else np.nan
    currency = df.at[idx, "financialCurrency"] if "financialCurrency" in df.columns else None
    currency = currency if isinstance(currency, str) else df.at[idx, "currency"] or "USD"
    new = fx.usd_rate(rates, currency)
    if pd.isna(old) or old <= 0 or new is None:
        return False
    factor = new / old
    if factor != 1:
        money = [f"{c}_{p}" for c in MONEY_CONCEPTS for p in cfg.fiscal_years]
        money += [f"{c}_{q}" for c in S.QUARTERLY_CONCEPTS for q in S.QUARTERS]
        for key in money:
            if key in df.columns and pd.notna(df.at[idx, key]):
                df.at[idx, key] = df.at[idx, key] * factor
        for period in cfg.fiscal_years:
            key = f"FX_Hist_{period}"
            if key in df.columns and pd.notna(df.at[idx, key]):
                df.at[idx, key] = df.at[idx, key] / factor
    df.at[idx, "Fin_Rate"] = new
    return True


def price_age_days(df: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    """Days since each row's price was set: the later of its price refresh and its full fetch."""
    stamps = [pd.to_datetime(df[c], errors="coerce") for c in ("Price_At", "Fetched_At") if c in df.columns]
    latest = pd.concat(stamps, axis=1).max(axis=1) if stamps else pd.Series(pd.NaT, index=df.index)
    return (now - latest).dt.total_seconds() / 86400  # NaN (never priced) counts as due


PE_KEYS = ("trailingPE", "forwardPE")  # Yahoo's ratios: earnings fixed until the next full fetch, price not


def _rescale_pe(df: pd.DataFrame, idx, close: pd.DataFrame | None, quote: float) -> None:
    """Move Yahoo's P/E ratios with the price, as market cap and EV are (in place): the earnings behind them change
    only with a full fetch, so P/E x (today's close / the close when the price was last set). The ratio is taken in
    the quote currency, so a currency move does not touch a ratio of two figures in one currency. Left as they are
    without that earlier close."""
    if close is None or df.at[idx, "Ticker"] not in close.columns:
        return
    stamps = [pd.to_datetime(df.at[idx, c], errors="coerce") for c in ("Price_At", "Fetched_At") if c in df.columns]
    then = max((t for t in stamps if pd.notna(t)), default=None)
    if then is None:
        return
    history = close[df.at[idx, "Ticker"]].dropna()
    before = history.loc[: then.normalize()]  # the close of the day the price was last set (or the day before)
    if before.empty or before.iloc[-1] <= 0:
        return
    for key in PE_KEYS:
        value = pd.to_numeric(df.at[idx, key], errors="coerce") if key in df.columns else np.nan
        if pd.notna(value):
            df.at[idx, key] = value * quote / before.iloc[-1]


def refresh_prices(
    df: pd.DataFrame, tickers: list[str], cfg: S.Settings, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Update prices (and momentum / 52-week low) of cached companies without re-downloading fundamentals.

    The new close is converted to USD like a full fetch (pence for `GBp`), and Yahoo's market cap /
    EV are scaled by the price change; metrics are recomputed from price downstream. Fields that
    need a full fetch (analyst data, statements) keep their cached values. Prices younger than
    `cfg.price_refresh_days` are kept, unless they lack the traded value (`Daily_Turnover_Shares`, the highlight lists'
    liquidity floor; added after most prices were stored) and are a day old: those are refreshed at most once a day.
    """
    out = df.copy()
    now = pd.Timestamp.now() if now is None else now
    cached = out[out["Ticker"].isin(tickers)]
    age = price_age_days(cached, now)
    turnover = pd.to_numeric(
        cached.get("Daily_Turnover_Shares", pd.Series(np.nan, index=cached.index)), errors="coerce"
    )
    fresh = set(cached.loc[(age < cfg.price_refresh_days) & ~(turnover.isna() & (age >= 1)), "Ticker"])
    todo = [t for t in tickers if t in set(out["Ticker"]) and t not in fresh]
    if fresh:
        log.info("%d prices are under %d days old; kept", len(fresh), cfg.price_refresh_days)
    if not todo:
        return out
    rates = fx.get_rates(cfg.data_dir / "fx_rates.json")
    close, low, volume = _price_history(todo)
    if isinstance(close, pd.DataFrame) and close.empty:  # Yahoo served no prices at all (throttled): keep the cached
        log.warning("No prices downloaded for %d tickers; cached prices kept", len(todo))
        return out
    stats = price_stats(close, low, volume)
    closes = stats["Last_Close"]
    out = out.astype({c: float for c in PRICE_STATS if c in out.columns})
    priced = out["Ticker"].isin([t for t in todo if pd.notna(closes.get(t)) and closes.get(t) > 0])
    updated, unrestated, missing = 0, 0, []
    for idx in out.index[out["Ticker"].isin(todo)]:
        ticker, currency = out.at[idx, "Ticker"], out.at[idx, "currency"] or "USD"
        quote = closes.get(ticker)
        if quote is None or pd.isna(quote) or quote <= 0:
            missing.append(ticker)
            continue
        rate = fx.usd_rate(rates, currency)  # pence for `GBp`
        if rate is None:
            missing.append(ticker)
            continue
        new_price = float(quote) * rate
        old_price = pd.to_numeric(out.at[idx, "currentPrice"], errors="coerce")
        if pd.notna(old_price) and old_price > 0:
            for key in CAP_KEYS:
                if key in out.columns and pd.notna(out.at[idx, key]):
                    out.at[idx, key] = out.at[idx, key] * new_price / old_price
        _rescale_pe(out, idx, close, float(quote))
        out.at[idx, "currentPrice"] = new_price
        out.at[idx, "Price_At"] = now.strftime("%Y-%m-%d %H:%M")
        out.at[idx, "FX_Source"] = fx.status()
        if not _restate_fx(out, idx, rates, cfg):
            unrestated += 1
        updated += 1
    out = _store_prices(out, priced, stats)
    log.info("Prices updated for %d of %d tickers", updated, len(todo))
    if unrestated:
        log.info("%d cached companies predate the stored statement rate: their statements keep the rate of "
                 "their last full fetch until the next one", unrestated)  # fmt: skip
    if missing:
        log.warning("No fresh price for: %s (cached price kept)", ", ".join(missing))
    return out


def save_raw(df: pd.DataFrame, path) -> None:
    """Persist to parquet, coercing mixed-type object columns so pyarrow accepts them. Written to a temporary file
    and renamed over `path`, so a run stopped mid-write leaves the previous file whole (it once corrupted the cache)."""
    out = df.copy()
    for col in [c for c in out.columns if out[c].dtype == "object"]:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() == out[col].notna().sum():
            out[col] = numeric
        else:
            out[col] = out[col].map(lambda v: None if v is None or pd.isna(v) else str(v))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_raw(path) -> pd.DataFrame:
    return pd.read_parquet(path)
