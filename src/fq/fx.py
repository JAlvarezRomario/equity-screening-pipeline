"""Exchange rates to USD, fetched lazily and cached for the process.

Order of preference: live Yahoo rates (saved to disk on success) -> the last saved live rates ->
a hard-coded table. Whatever was used is reported in `status()` and stored with the data, so a
silent fallback can never slip into a report unnoticed.
"""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Last resort if Yahoo is unreachable and nothing was saved. Value of one unit of currency in USD (2025).
FALLBACK_RATES: dict[str, float] = {
    "USD": 1.00, "CAD": 0.7097, "EUR": 1.1521, "CNY": 0.1407, "JPY": 0.0064,
    "MXN": 0.0541, "BRL": 0.1852, "CLP": 0.00107, "TRY": 0.0236, "TWD": 0.0319,
    "NOK": 0.0976, "ILS": 0.3040, "DKK": 0.1541, "ZAR": 0.0575, "HKD": 0.1285,
    "INR": 0.0112, "GBP": 1.3106, "ARS": 0.0007, "CHF": 1.2376, "AUD": 0.6452,
    "KRW": 0.00068, "PEN": 0.2959, "MYR": 0.2398, "SEK": 0.1047, "COP": 0.00027,
    "SGD": 0.7634, "PHP": 0.0170, "KZT": 0.00192, "VND": 0.000038, "PLN": 0.2717,
    "THB": 0.0309, "GEL": 0.3690, "NZD": 0.5800, "IDR": 0.000061, "AED": 0.2723, "SAR": 0.2667,
    "CZK": 0.0460, "HUF": 0.0028, "EGP": 0.0205, "KWD": 3.26, "QAR": 0.2747,
}  # fmt: skip
# Quotes in a minor unit (pence, cents, agorot): the base currency and units per base unit.
MINOR_UNITS: dict[str, tuple[str, int]] = {
    "GBp": ("GBP", 100),
    "GBX": ("GBP", 100),
    "ZAc": ("ZAR", 100),
    "ILA": ("ILS", 100),
    "KWF": ("KWD", 1000),
}
FALLBACK_ASOF = "2025-12-31"
STALE_DAYS = 7

_cache: dict[str, dict] = {}
_status = {"source": "not loaded", "asof": None}
_history: dict[str, pd.Series | None] = {}
HISTORY_PERIOD = "max"  # dividend records reach back decades (`fetch.declared_currency_payments`)


def status() -> str:
    """Where the rates in use came from, e.g. 'live 2026-10-18' or 'saved 2026-09-01 (stale)'."""
    return f"{_status['source']} {_status['asof'] or ''}".strip()


def _live() -> dict[str, float]:
    currencies = [c for c in FALLBACK_RATES if c != "USD"]
    close = yf.download([f"{c}=X" for c in currencies], period="5d", interval="1d", progress=False, auto_adjust=False)[
        "Close"
    ]
    last = close.ffill().iloc[-1]
    rates = {"USD": 1.0}
    for cur in currencies:
        quote = last.get(f"{cur}=X")  # Yahoo quotes CCY per 1 USD
        if pd.notna(quote) and quote > 0:
            rates[cur] = 1.0 / float(quote)
    return rates


def get_rates(cache_path: Path | None = None) -> dict[str, float]:
    """Return {currency: value of 1 unit in USD}, filling any gap from the next source down."""
    key = str(cache_path)
    if key in _cache:
        return _cache[key]

    saved: dict = {}
    if cache_path and cache_path.is_file():
        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}

    rates = dict(FALLBACK_RATES)
    try:
        live = _live()
        if len(live) < len(FALLBACK_RATES) * 0.8:
            raise RuntimeError(f"only {len(live)} currencies returned")
        rates.update(saved.get("rates", {}))
        rates.update(live)
        _status.update(source="live", asof=date.today().isoformat())
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"asof": _status["asof"], "rates": rates}, indent=1), encoding="utf-8")
        log.info("Live exchange rates loaded")
        if table_only := sorted(set(FALLBACK_RATES) - set(live) - set(saved.get("rates", {}))):
            log.warning("No Yahoo rate for %s: using the built-in %s table", ", ".join(table_only), FALLBACK_ASOF)
    except Exception as exc:
        if saved.get("rates"):
            rates.update(saved["rates"])
            asof = saved.get("asof")
            age = (date.today() - date.fromisoformat(asof)).days if asof else None
            stale = age is None or age > STALE_DAYS
            _status.update(source="saved (stale)" if stale else "saved", asof=asof)
            (log.warning if stale else log.info)("Live FX failed (%s); using rates saved on %s", exc, asof)
        else:
            _status.update(source="built-in table (stale)", asof=FALLBACK_ASOF)
            log.warning("Live FX failed (%s) and no saved rates; using the built-in %s table", exc, FALLBACK_ASOF)
    _cache[key] = rates
    return rates


def usd_rate(rates: dict[str, float], currency: str | None) -> float | None:
    """Value of one unit of `currency` in USD (a minor unit such as pence is divided down). A currency
    missing from `rates` is fetched once from Yahoo's history and added; None when it cannot be found,
    so the caller can skip the company instead of reading its figures as if they were dollars."""
    if not currency:
        return None
    base, units = MINOR_UNITS.get(currency, (currency, 1))
    if base not in rates:
        hist = _usd_history(base)
        if hist is None or hist.empty:
            log.warning("No exchange rate for %s", base)
            return None
        rates[base] = float(hist.iloc[-1])
        log.info("Exchange rate for %s fetched separately (not in the standard list)", base)
    return rates[base] / units


def _usd_history(currency: str) -> pd.Series | None:
    """Daily value of one unit of `currency` in USD (index = date), downloaded once per process."""
    if currency not in _history:
        try:
            close = yf.download(f"{currency}=X", period=HISTORY_PERIOD, interval="1d", progress=False,
                                auto_adjust=False)["Close"]  # fmt: skip
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            close = close[close > 0].dropna()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            _history[currency] = (1.0 / close) if not close.empty else None  # Yahoo quotes CCY per 1 USD
        except Exception as exc:
            log.warning("FX history for %s unavailable (%s)", currency, exc)
            _history[currency] = None
    return _history[currency]


FALLBACK_TREASURY_YIELD = 4.25
_treasury_cache: dict[str, float] = {}


def get_treasury_yield(cache_path: Path | None = None) -> float:
    """10-year US Treasury yield (%) from Yahoo (^TNX), with disk cache and built-in fallback."""
    key = str(cache_path or "default")
    if key in _treasury_cache:
        return _treasury_cache[key]

    saved_yield = None
    if cache_path and cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            saved_yield = float(data.get("yield"))
        except (OSError, ValueError, TypeError):
            saved_yield = None

    try:
        quote = yf.Ticker("^TNX").fast_info.get("lastPrice")
        if quote is None or pd.isna(quote) or quote <= 0:
            close = yf.download("^TNX", period="5d", interval="1d", progress=False, auto_adjust=False)["Close"]
            val = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            quote = float(val.dropna().iloc[-1])
        if pd.notna(quote) and quote > 0:
            res = round(float(quote), 2)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"asof": date.today().isoformat(), "yield": res}, indent=1), encoding="utf-8"
                )
            _treasury_cache[key] = res
            return res
    except Exception as exc:
        log.warning("Could not fetch 10Y Treasury yield (%s)", exc)

    if saved_yield is not None and saved_yield > 0:
        _treasury_cache[key] = saved_yield
        return saved_yield

    _treasury_cache[key] = FALLBACK_TREASURY_YIELD
    return FALLBACK_TREASURY_YIELD


ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?lastNObservations=1&format=jsondata"
BROKER_SPREAD = 1.5  # %: a typical broker's margin over €STR on a euro margin loan
BORROW_BUFFER = 0.5  # %: the user's safety margin on top (user's choice, Nov 5, 2026)
FALLBACK_BORROW_RATE = 3.75  # %: when neither the ECB nor a stored rate is at hand
_borrow_cache: dict[str, float] = {}


def _estr() -> float | None:
    """The latest euro short-term rate (€STR, %) from the ECB's data API, or None."""
    import urllib.request

    with urllib.request.urlopen(ESTR_URL, timeout=20) as resp:  # noqa: S310 - a fixed https URL
        data = json.loads(resp.read().decode("utf-8"))
    for series in data["dataSets"][0]["series"].values():
        values = [obs[0] for obs in series["observations"].values() if obs and obs[0] is not None]
        if values:
            return float(values[-1])
    return None


def get_borrow_rate(cache_path: Path | None = None) -> float:
    """What a euro margin loan costs (%), the Income list's spread over the loan (user's choice, Nov 5, 2026: obtained,
    not hardcoded): €STR from the ECB (`ESTR_URL`) + `BROKER_SPREAD` + `BORROW_BUFFER`. The rate is stored in
    `cache_path` with its date; when the ECB cannot be reached, the stored rate is used, else `FALLBACK_BORROW_RATE`."""
    key = str(cache_path or "default")
    if key in _borrow_cache:
        return _borrow_cache[key]
    saved = None
    if cache_path and cache_path.is_file():
        try:
            saved = float(json.loads(cache_path.read_text(encoding="utf-8")).get("rate"))
        except (OSError, ValueError, TypeError):
            saved = None
    try:
        estr = _estr()
        if estr is not None and 0 <= estr < 20:
            rate = round(estr + BROKER_SPREAD + BORROW_BUFFER, 2)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"asof": date.today().isoformat(), "estr": estr, "rate": rate}
                cache_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            _borrow_cache[key] = rate
            return rate
    except Exception as exc:
        log.warning("Could not fetch the euro short-term rate (%s)", exc)
    _borrow_cache[key] = saved if saved is not None and saved > 0 else FALLBACK_BORROW_RATE
    return _borrow_cache[key]


def rate_on(currency: str, when) -> float | None:
    """Value of one unit of `currency` in USD on `when` (last quote on or before it); None if unknown."""
    base, units = MINOR_UNITS.get(currency, (currency, 1))
    if base == "USD":
        return 1.0 / units
    hist = _usd_history(base)
    if hist is None or when is None or pd.isna(when):
        return None
    past = hist[hist.index <= pd.Timestamp(when)]
    return float(past.iloc[-1]) / units if not past.empty else None
