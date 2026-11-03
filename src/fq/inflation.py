"""Consumer price inflation by currency, so returns earned in any currency can be read in US terms.

Source: the IMF World Economic Outlook (DataMapper API, `PCPIPCH`: annual average CPI, % per year, history and
forecasts for every country), one request for all. Order of preference, as in `fx`: the live download (saved
to disk on success) -> the last saved download -> a built-in table of one recent rate per currency (loud
warning). Only `load(live=True)` touches the network; the metrics read what is saved (`table`), so a rank,
`--offline` and the tests never download.
"""

import json
import logging
import urllib.request
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

IMF_URL = "https://www.imf.org/external/datamapper/api/v1/PCPIPCH"
REFRESH_DAYS = 30  # the WEO is published twice a year; a month-old download is current
# The statements' currency -> its IMF area (currency, not country: the figures are in it).
CURRENCY_AREA: dict[str, str] = {
    "USD": "USA", "CAD": "CAN", "EUR": "EURO", "CNY": "CHN", "JPY": "JPN", "MXN": "MEX", "BRL": "BRA",
    "CLP": "CHL", "TRY": "TUR", "TWD": "TWN", "NOK": "NOR", "ILS": "ISR", "DKK": "DNK", "ZAR": "ZAF",
    "HKD": "HKG", "INR": "IND", "GBP": "GBR", "ARS": "ARG", "CHF": "CHE", "AUD": "AUS", "KRW": "KOR",
    "PEN": "PER", "MYR": "MYS", "SEK": "SWE", "COP": "COL", "SGD": "SGP", "PHP": "PHL", "KZT": "KAZ",
    "VND": "VNM", "PLN": "POL", "THB": "THA", "GEL": "GEO", "NZD": "NZL", "IDR": "IDN", "AED": "ARE",
    "SAR": "SAU", "CZK": "CZE", "HUF": "HUN", "EGP": "EGY", "KWD": "KWT", "QAR": "QAT", "NGN": "NGA",
    "PKR": "PAK", "ISK": "ISL", "RON": "ROU", "BGN": "BGR", "MAD": "MAR", "KES": "KEN", "BHD": "BHR",
    "OMR": "OMN", "JOD": "JOR", "LKR": "LKA", "BDT": "BGD", "UAH": "UKR",
}  # fmt: skip
# Quotes in a minor unit report statements in the base currency (`fx.MINOR_UNITS`).
MINOR_BASE = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS", "KWF": "KWD"}
# Last resort when nothing was ever downloaded: one recent rate (% per year, IMF WEO Oct 2026, recent years and
# the next forecast) used for every year. Rough by design; the warning says so.
FALLBACK: dict[str, float] = {
    "USD": 3.0, "CAD": 2.5, "EUR": 2.5, "CNY": 0.5, "JPY": 2.5, "MXN": 4.0, "BRL": 5.0, "CLP": 4.0,
    "TRY": 35.0, "TWD": 2.0, "NOK": 3.0, "ILS": 3.0, "DKK": 2.0, "ZAR": 4.5, "HKD": 1.8, "INR": 4.0,
    "GBP": 3.0, "ARS": 40.0, "CHF": 0.5, "AUD": 3.0, "KRW": 2.2, "PEN": 2.5, "MYR": 2.0, "SEK": 2.0,
    "COP": 5.5, "SGD": 1.5, "PHP": 3.0, "KZT": 9.0, "VND": 3.5, "PLN": 4.0, "THB": 0.5, "GEL": 3.0,
    "NZD": 2.5, "IDR": 2.5, "AED": 2.0, "SAR": 2.0, "CZK": 2.5, "HUF": 4.5, "EGP": 20.0, "KWD": 2.5,
    "QAR": 2.0, "NGN": 20.0, "PKR": 8.0,
}  # fmt: skip

_status = {"source": "not loaded", "asof": None}
_tables: dict[str, dict[str, dict[int, float]]] = {}


def status() -> str:
    """Where the rates in use came from, e.g. 'IMF 2026-10-28' or 'built-in table'."""
    return f"{_status['source']} {_status['asof'] or ''}".strip()


def _download() -> dict[str, dict[int, float]]:
    with urllib.request.urlopen(IMF_URL, timeout=30) as resp:  # noqa: S310 (fixed https URL)
        values = json.load(resp)["values"]["PCPIPCH"]
    return {area: {int(y): float(v) for y, v in years.items() if v is not None} for area, years in values.items()}


def _read(path: Path | None) -> tuple[dict[str, dict[int, float]], str | None]:
    if not path or not path.is_file():
        return {}, None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        series = {a: {int(y): float(v) for y, v in s.items()} for a, s in saved["series"].items()}
        return series, saved.get("asof")
    except (OSError, ValueError, KeyError, TypeError):
        return {}, None


def load(path: Path | None, live: bool = False) -> dict[str, dict[int, float]]:
    """{IMF area: {year: CPI %}} (empty = use `FALLBACK`). `live` downloads when the saved copy is missing or
    older than `REFRESH_DAYS` and saves it; a failed download keeps the saved copy."""
    key = str(path)
    if key in _tables and not live:
        return _tables[key]
    series, asof = _read(path)
    age = (date.today() - date.fromisoformat(asof)).days if asof else None
    if live and (age is None or age > REFRESH_DAYS):
        try:
            series, asof = _download(), date.today().isoformat()
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"asof": asof, "source": IMF_URL, "series": series}
                path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            log.info("IMF inflation rates downloaded (%d areas)", len(series))
        except Exception as exc:  # network, format: keep what was saved
            log.warning("IMF inflation download failed (%s); using %s", exc, f"the copy of {asof}" if asof else
                        "the built-in table")  # fmt: skip
    if series:
        _status.update(source="IMF", asof=asof)
    else:
        _status.update(source="built-in table", asof=None)
        log.warning("No IMF inflation rates saved; using the built-in table (`fq fetch` downloads them)")
    _tables[key] = series
    return series


def table(path: Path | None) -> dict[str, dict[int, float]]:
    """The saved rates (no download), cached for the process."""
    return load(path, live=False)


def rate(series: dict[str, dict[int, float]], currency: str | None, year: int) -> float | None:
    """CPI inflation (% per year) of `currency` in `year`: the nearest year the IMF has when that year is out of
    its range; the built-in rate without a download; None for an unknown currency."""
    if not currency:
        return None
    currency = MINOR_BASE.get(currency, currency)
    years = series.get(CURRENCY_AREA.get(currency, ""), {})
    if years:
        return years.get(year, years[min(years, key=lambda y: abs(y - year))])
    return FALLBACK.get(currency)


def excess(series: dict[str, dict[int, float]], currency: str | None, year: int) -> float | None:
    """Inflation of `currency` beyond US inflation in `year`, as a fraction: (1 + local) / (1 + US) - 1. A return
    earned in `currency` divided by (1 + excess) is the same return in US terms (Fisher). None when unknown."""
    local, us = rate(series, currency, year), rate(series, "USD", year)
    if local is None or us is None:
        return None
    return (1 + local / 100) / (1 + us / 100) - 1
