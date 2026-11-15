"""Historical macro yields and subsequent 3-year annualized total returns (US 1960–present, Japan 1986–present).

Fetches monthly series for 10-year sovereign bond yields, equity index trailing earnings yields, and cumulative
equity total return indices, computes the yield spread (pp) and forward 3-year annualized total return (% / yr),
caches atomically to `data/macro_yields.parquet` and `data/macro_japan_yields.parquet` (refreshed once a year by
default, preserving cached historical rows), and falls back to the cached file when offline or if a remote source fails.
"""

from __future__ import annotations

import io
import logging
import os
import re
import struct
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

START_DATE = pd.Timestamp("1960-01-01")
MACRO_MAX_AGE_DAYS = 365
FORWARD_MONTHS = 36

FRED_GS10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
MULTPL_EY_URL = "https://www.multpl.com/s-p-500-earnings-yield/table/by-month"
MULTPL_DY_URL = "https://www.multpl.com/s-p-500-dividend-yield/table/by-month"
CACHE_FILE = "macro_yields.parquet"

MOF_JGB_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
JPX_PER_INDEX_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/04.html"
JPX_LONGRANGE_FALLBACK_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/misc/tvdivq00000015r6-att/longrange-perpbr.xlsx"
)
JPX_DIV_YIELD_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/misc/tvdivq0000001w9e-att/historical-y.xls"
)
FRED_JP_EQUITY_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SPASTT01JPM661N"
JPX_CACHE_FILE = "macro_japan_yields.parquet"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _to_month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index).tz_localize(None)
    return idx + pd.offsets.MonthEnd(0)


def compute_forward_3y_cagr(tr_index: pd.Series, months: int = FORWARD_MONTHS) -> pd.Series:
    """Compute the forward 3-year annualized total return (% / yr) from a monthly total return index.

    Aligns to a continuous calendar month-end grid before shifting by `-months` so that any missing month never
    shifts the 36-month horizon. The final `months` observations (where `t + 36m` lies in the future) are `NaN`.
    """
    clean = pd.to_numeric(tr_index, errors="coerce").dropna()
    clean = clean[clean > 0]
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    if clean.empty:
        return pd.Series(dtype=float, name="forward_3y_cagr")
    grid = pd.date_range(clean.index.min(), clean.index.max(), freq="ME")
    aligned = clean.reindex(grid)
    cagr = ((aligned.shift(-months) / aligned) ** (12.0 / months) - 1.0) * 100.0
    out = cagr.reindex(clean.index).round(2)
    out.name = "forward_3y_cagr"
    return out


def _build_synthetic_tr_index(price: pd.Series, div_yield_pct: pd.Series) -> pd.Series:
    """Construct a monthly cumulative total return index from month-end prices and annualized dividend yield (%)."""
    p = pd.to_numeric(price, errors="coerce").dropna()
    p = p[p > 0]
    p = p[~p.index.duplicated(keep="last")].sort_index()
    if p.empty:
        return pd.Series(dtype=float)
    grid = pd.date_range(p.index.min(), p.index.max(), freq="ME")
    p_grid = p.reindex(grid).ffill()
    dy_grid = pd.to_numeric(div_yield_pct, errors="coerce").reindex(grid).ffill().bfill().fillna(0.0)
    monthly_ret = p_grid.pct_change().fillna(0.0) + (dy_grid / 100.0) / 12.0
    monthly_ret.iloc[0] = 0.0
    tr = (1.0 + monthly_ret).cumprod() * 100.0
    return tr.reindex(p.index)


def _splice_total_return(primary_tr: pd.Series, backfill_tr: pd.Series) -> pd.Series:
    """Chain `backfill_tr` onto `primary_tr` at their earliest overlapping month-end."""
    p = pd.to_numeric(primary_tr, errors="coerce").dropna()
    p = p[p > 0]
    p = p[~p.index.duplicated(keep="last")].sort_index()
    b = pd.to_numeric(backfill_tr, errors="coerce").dropna()
    b = b[b > 0]
    b = b[~b.index.duplicated(keep="last")].sort_index()
    if p.empty:
        return b
    if b.empty:
        return p
    common = p.index.intersection(b.index)
    if len(common) > 0:
        anchor = common.min()
        scale = float(p.loc[anchor]) / float(b.loc[anchor])
        b_scaled = b * scale
        # Also scale any tail after primary_tr ends if primary_tr is ever behind backfill_tr
        last_common = common.max()
        tail = b[b.index > last_common]
        if not tail.empty:
            tail_scale = float(p.loc[last_common]) / float(b.loc[last_common])
            b_scaled = pd.concat([b_scaled[b_scaled.index <= last_common], tail * tail_scale])
        return p.combine_first(b_scaled).sort_index()
    return p.combine_first(b).sort_index()


def fetch_fred_gs10(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly 10-year US Treasury yield (%) from FRED (`GS10`), clipped to `start` onward."""
    df = pd.read_csv(FRED_GS10_URL)
    dates = pd.to_datetime(df["observation_date"], errors="coerce")
    vals = pd.to_numeric(df["GS10"], errors="coerce")
    valid = dates.notna() & vals.notna() & (vals > 0)
    series = pd.Series(
        vals[valid].values,
        index=_to_month_end(pd.DatetimeIndex(dates[valid])),
        name="treasury_10y",
        dtype=float,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= start]


def fetch_treasury_history(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly 10-year US Treasury yield (%) from Yahoo (^TNX, 1962+) backfilled from FRED GS10 (1960–1961)."""
    hist = yf.download("^TNX", period="max", interval="1d", progress=False, auto_adjust=False)["Close"]
    series = hist.iloc[:, 0] if isinstance(hist, pd.DataFrame) else hist
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.resample("ME").last().dropna()
    if start < series.index.min():
        try:
            gs10 = fetch_fred_gs10(start=start)
            series = series.combine_first(gs10)
        except Exception as exc:
            log.warning("Could not backfill pre-1962 Treasury yields from FRED GS10 (%s)", exc)
    series = series[series.index >= start].sort_index()
    series.name = "treasury_10y"
    return series.astype(float)


def parse_multpl_table(html: str, start: pd.Timestamp = START_DATE, name: str = "sp500_earnings_yield") -> pd.Series:
    """Parse a Multpl monthly HTML table into a month-end Series (%)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="datatable") or soup.find("table")
    if table is None:
        raise ValueError("No data table found in Multpl HTML")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        raw_date = cells[0].get_text(strip=True)
        raw_val = cells[1].get_text(strip=True).replace("%", "").replace(",", "").strip()
        try:
            dt = pd.Timestamp(raw_date)
            val = float(raw_val)
        except (ValueError, TypeError):
            continue
        dates.append(dt)
        values.append(val)

    if not dates:
        raise ValueError("No rows parsed from Multpl table")

    series = pd.Series(values, index=_to_month_end(pd.DatetimeIndex(dates)), name=name, dtype=float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= start]


def fetch_sp500_earnings_yield(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Download and parse Multpl's monthly S&P 500 TTM earnings yield (%) from `start` onward."""
    req = urllib.request.Request(MULTPL_EY_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed https URL
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")
    return parse_multpl_table(html, start=start, name="sp500_earnings_yield")


def fetch_sp500_dividend_yield(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Download and parse Multpl's monthly S&P 500 trailing dividend yield (%) from `start` onward."""
    req = urllib.request.Request(MULTPL_DY_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed https URL
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")
    return parse_multpl_table(html, start=start, name="sp500_dividend_yield")


def _download_yf_monthly_close(ticker: str, col: str = "Close", start: pd.Timestamp = START_DATE) -> pd.Series:
    """Download a daily Yahoo series and resample to month-end close."""
    hist = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    target = hist[col] if col in hist else hist["Close"]
    series = target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    if series.empty:
        return pd.Series(dtype=float)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.resample("ME").last().dropna()
    return series[series.index >= start].sort_index().astype(float)


def fetch_sp500_total_return(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly S&P 500 cumulative total return index from `start` onward.

    Uses Yahoo Finance `^SP500TR` (1988–present) spliced with `^GSPC` monthly price returns plus Multpl's monthly
    S&P 500 dividend yield for 1960–1987.
    """
    sp500tr = pd.Series(dtype=float)
    try:
        sp500tr = _download_yf_monthly_close("^SP500TR", col="Close", start=start)
    except Exception as exc:
        log.warning("Could not download ^SP500TR (%s)", exc)

    synth_tr = pd.Series(dtype=float)
    if sp500tr.empty or start < sp500tr.index.min():
        try:
            gspc = _download_yf_monthly_close("^GSPC", col="Close", start=start)
            dy = fetch_sp500_dividend_yield(start=start)
            synth_tr = _build_synthetic_tr_index(gspc, dy)
        except Exception as exc:
            log.warning("Could not build pre-1988 S&P 500 synthetic total return (%s)", exc)

    combined = _splice_total_return(sp500tr, synth_tr)
    combined = combined[combined.index >= start].sort_index()
    combined.name = "total_return_index"
    return combined


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _cache_is_fresh(cache_path: Path, max_age_days: int) -> bool:
    if not cache_path.is_file():
        return False
    age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
    return age_days <= max_age_days


def _ensure_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "forward_3y_cagr" not in df.columns:
        if "total_return_index" in df.columns:
            df = df.copy()
            df["forward_3y_cagr"] = compute_forward_3y_cagr(df["total_return_index"]).reindex(df.index)
        else:
            df = df.copy()
            df["forward_3y_cagr"] = np.nan
    return df


def load_macro_yields(
    cache_path: Path,
    max_age_days: int = MACRO_MAX_AGE_DAYS,
    offline: bool = False,
    start: pd.Timestamp = START_DATE,
) -> pd.DataFrame:
    """Load monthly 10Y Treasury yield, S&P 500 TTM earnings yield, spread (pp), and forward 3Y total return (% / yr).

    Uses `cache_path` when fresh (default 365 days) and already containing `forward_3y_cagr`, or when `offline=True`;
    otherwise refreshes from Yahoo (`^TNX`, `^SP500TR`, `^GSPC`) and Multpl, merging with any existing cached history
    so earlier decades are preserved.
    """
    existing = pd.DataFrame()
    if cache_path.is_file():
        try:
            existing = pd.read_parquet(cache_path).sort_index()
        except Exception as exc:
            log.warning("Could not read cached macro yields %s (%s)", cache_path, exc)

    has_cagr = "forward_3y_cagr" in existing.columns and existing["forward_3y_cagr"].notna().any()
    if not existing.empty and (offline or (_cache_is_fresh(cache_path, max_age_days) and has_cagr)):
        if offline or existing.index.min() <= start + pd.offsets.MonthEnd(1):
            out = _ensure_macro_columns(existing)
            return out[out.index >= start].sort_index()

    if offline:
        return pd.DataFrame(columns=["treasury_10y", "sp500_earnings_yield", "spread_pp", "forward_3y_cagr"])

    try:
        ty = fetch_treasury_history(start=start)
        ey = fetch_sp500_earnings_yield(start=start)
        df = pd.concat([ty, ey], axis=1)
        if not existing.empty:
            for col in ("treasury_10y", "sp500_earnings_yield"):
                if col in existing.columns:
                    df[col] = df[col].combine_first(existing[col])
        df = df.dropna(subset=["treasury_10y", "sp500_earnings_yield"]).sort_index()
        if df.empty:
            raise ValueError("Joined macro yields DataFrame is empty")
        df["spread_pp"] = (df["sp500_earnings_yield"] - df["treasury_10y"]).round(4)

        tr = pd.Series(dtype=float)
        try:
            tr = fetch_sp500_total_return(start=start)
        except Exception as exc:
            log.warning("Could not refresh S&P 500 total return series (%s)", exc)
        if not existing.empty and "total_return_index" in existing.columns:
            tr = _splice_total_return(tr, existing["total_return_index"])
        if not tr.empty:
            df["total_return_index"] = tr.reindex(df.index)
            cagr = compute_forward_3y_cagr(tr)
            df["forward_3y_cagr"] = cagr.reindex(df.index)
        elif not existing.empty and "forward_3y_cagr" in existing.columns:
            df["forward_3y_cagr"] = existing["forward_3y_cagr"].reindex(df.index)
        else:
            df["forward_3y_cagr"] = np.nan

        df.index.name = "date"
        _atomic_write_parquet(df, cache_path)
        log.info(
            "Macro yields updated (%d months: %s to %s)",
            len(df),
            df.index[0].strftime("%Y-%m"),
            df.index[-1].strftime("%Y-%m"),
        )
        return df[df.index >= start].sort_index()
    except Exception as exc:
        log.warning("Could not refresh macro yields (%s)", exc)
        if not existing.empty:
            out = _ensure_macro_columns(existing)
            return out[out.index >= start].sort_index()
        return pd.DataFrame(columns=["treasury_10y", "sp500_earnings_yield", "spread_pp", "forward_3y_cagr"])


def parse_mof_jgb_csv(csv_text: str) -> pd.Series:
    """Parse Japan MOF daily JGB CSV into a month-end 10-year yield Series (%)."""
    lines = csv_text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Date,")), None)
    if header_idx is None:
        raise ValueError("No Date header found in MOF JGB CSV")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    if "Date" not in df.columns or "10Y" not in df.columns:
        raise ValueError("Expected Date and 10Y columns in MOF JGB CSV")
    dates = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    vals = pd.to_numeric(df["10Y"], errors="coerce")
    valid = dates.notna() & vals.notna()
    series = pd.Series(vals[valid].values, index=pd.DatetimeIndex(dates[valid]), name="jgb_10y", dtype=float)
    series = series.sort_index().resample("ME").last().dropna()
    return series


def fetch_japan_jgb_history() -> pd.Series:
    """Download and parse Japan Ministry of Finance 10-year JGB yields (%)."""
    req = urllib.request.Request(MOF_JGB_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - fixed https URL
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
    return parse_mof_jgb_csv(text)


def _xlsx_col_idx(cell_ref: str) -> int:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _read_xlsx_sheet_rows(z: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[dict[int, str]]:
    root = ET.fromstring(z.read(sheet_path))
    rows: list[dict[int, str]] = []
    for row in root.iter():
        if not row.tag.endswith("row"):
            continue
        cells: dict[int, str] = {}
        for cell in row:
            ref = cell.attrib.get("r", "")
            t = cell.attrib.get("t", "")
            val = next((child.text for child in cell if child.tag.endswith("v") and child.text is not None), None)
            if val is None:
                continue
            if t == "s":
                val = shared[int(val)]
            cells[_xlsx_col_idx(ref)] = val
        if cells:
            rows.append(cells)
    return rows


def _extract_jpx_per_rows(
    z: zipfile.ZipFile, sheet_path: str, shared: list[str], per_col: int
) -> dict[pd.Timestamp, float]:
    records: dict[pd.Timestamp, float] = {}
    for cells in _read_xlsx_sheet_rows(z, sheet_path, shared):
        raw_y = cells.get(0, "").strip()
        raw_m = cells.get(1, "").strip()
        raw_per = cells.get(per_col, "").strip()
        if not (re.fullmatch(r"\d{4}", raw_y) and re.fullmatch(r"\d{1,2}", raw_m)):
            continue
        try:
            per = float(raw_per)
            if per <= 0:
                continue
            dt = pd.Timestamp(year=int(raw_y), month=int(raw_m), day=1) + pd.offsets.MonthEnd(0)
        except (ValueError, TypeError):
            continue
        records[dt] = 100.0 / per
    return records


def parse_jpx_perpbr_xlsx(raw_bytes: bytes) -> pd.Series:
    """Parse JPX `longrange-perpbr.xlsx` into a month-end earnings yield (%) Series (1971–present).

    Priority per month: Consolidated Weighted avg (2013+) > Consolidated AVG (1999–2012) >
    Non-Consolidated AVG (1971–1998), using TSE 1st Section / Prime Section PER.
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(si.itertext()) for si in ss_root.iter() if si.tag.endswith("si")]

        wb_root = ET.fromstring(z.read("xl/workbook.xml"))
        rels_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"].lstrip("/") for rel in rels_root if rel.tag.endswith("Relationship")
        }
        sheets: dict[str, str] = {}
        for sh in wb_root.iter():
            if sh.tag.endswith("sheet"):
                name = sh.attrib.get("name", "")
                rid = next((v for k, v in sh.attrib.items() if k.endswith("id")), "")
                target = rel_targets.get(rid, "")
                if target and not target.startswith("xl/"):
                    target = f"xl/{target}"
                if name and target:
                    sheets[name] = target

        # Ordered from lowest to highest priority so newer/better bases overwrite older ones on overlap.
        tiers: list[tuple[str, int]] = []
        for name in sheets:
            if name.startswith("Non-Consolidated(AVG)"):
                tiers.append((name, 8))  # col I: TSE 1st Section PER (1971-12+)
        for name in sheets:
            if name.startswith("Consolidated(AVG)"):
                tiers.append((name, 3))  # col D: TSE 1st Section PER (1999-01+)
        for name in sheets:
            if "Consolidated" in name and ("Weighted avg" in name or "WT AVG" in name):
                tiers.append((name, 3))  # col D: TSE 1st Section / Prime PER (2013-01+)

        if not tiers:
            raise ValueError("No recognised PER sheets found in JPX workbook")

        records: dict[pd.Timestamp, float] = {}
        for sheet_name, per_col in tiers:
            records.update(_extract_jpx_per_rows(z, sheets[sheet_name], shared, per_col))

    if not records:
        raise ValueError("No PER rows parsed from JPX workbook")
    return pd.Series(records, name="topix_earnings_yield", dtype=float).sort_index()


def _discover_jpx_longrange_url() -> str:
    try:
        req = urllib.request.Request(JPX_PER_INDEX_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "longrange-perpbr.xlsx" in href:
                return href if href.startswith("http") else f"https://www.jpx.co.jp{href}"
    except Exception as exc:
        log.debug("JPX index page lookup failed (%s); using fallback URL", exc)
    return JPX_LONGRANGE_FALLBACK_URL


def fetch_japan_earnings_yield() -> pd.Series:
    """Download and parse JPX Consolidated Weighted-Average PER into monthly earnings yield (%)."""
    url = _discover_jpx_longrange_url()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw_bytes = resp.read()
    return parse_jpx_perpbr_xlsx(raw_bytes)


def _decode_biff8_rk(rk: int) -> float:
    if rk & 2:
        val = float(struct.unpack("<i", struct.pack("<I", rk))[0] >> 2)
    else:
        val = struct.unpack("<d", struct.pack("<Q", (int(rk) & 0xFFFFFFFC) << 32))[0]
    return val / 100.0 if (rk & 1) else val


def parse_jpx_dividend_yield_xls(raw_bytes: bytes) -> pd.Series:
    """Parse JPX `historical-y.xls` (BIFF8) into a month-end TSE 1st Section / Prime dividend yield (%) Series."""
    bof = next(
        (s for s in range(512, len(raw_bytes), 512) if struct.unpack_from("<H", raw_bytes, s)[0] == 0x0809),
        None,
    )
    if bof is None:
        raise ValueError("No BIFF8 Workbook BOF found in JPX dividend yield file")

    pos = bof
    sheet_offsets: list[int] = []
    while pos + 4 <= len(raw_bytes):
        rt, rl = struct.unpack_from("<HH", raw_bytes, pos)
        if rt == 0x0085 and rl >= 4:
            ofs = struct.unpack_from("<I", raw_bytes, pos + 4)[0]
            sheet_offsets.append(bof + ofs)
        elif rt == 0x000A:
            break
        pos += 4 + rl

    records: dict[pd.Timestamp, float] = {}
    for ofs in reversed(sheet_offsets):
        pos = ofs
        rows: dict[int, dict[int, float]] = {}
        while pos + 4 <= len(raw_bytes):
            rt, rl = struct.unpack_from("<HH", raw_bytes, pos)
            if rt == 0x000A:
                break
            if rt == 0x0203 and rl >= 14:
                r, c, _xf, val = struct.unpack_from("<HHHd", raw_bytes, pos + 4)
                rows.setdefault(r, {})[c] = val
            elif rt == 0x027E and rl >= 10:
                r, c, _xf = struct.unpack_from("<HHH", raw_bytes, pos + 4)
                rk = struct.unpack_from("<I", raw_bytes, pos + 10)[0]
                rows.setdefault(r, {})[c] = _decode_biff8_rk(rk)
            elif rt == 0x00BD and rl >= 12:
                r, fc = struct.unpack_from("<HH", raw_bytes, pos + 4)
                for i in range((rl - 6) // 6):
                    rk = struct.unpack_from("<I", raw_bytes, pos + 10 + i * 6)[0]
                    rows.setdefault(r, {})[fc + i] = _decode_biff8_rk(rk)
            pos += 4 + rl

        for _r, cells in sorted(rows.items()):
            serial = cells.get(0, 0.0)
            dy = cells.get(2) or cells.get(1)
            if 20000 <= serial <= 60000 and dy is not None and 0 < dy < 15:
                dt = pd.Timestamp("1899-12-30") + pd.to_timedelta(int(round(serial)), unit="D")
                records[dt + pd.offsets.MonthEnd(0)] = round(float(dy), 4)

    if not records:
        raise ValueError("No dividend yield rows parsed from JPX historical-y.xls")
    return pd.Series(records, name="topix_dividend_yield", dtype=float).sort_index()


def fetch_japan_dividend_yield() -> pd.Series:
    """Download and parse JPX TSE 1st Section / Prime average dividend yield (%)."""
    req = urllib.request.Request(JPX_DIV_YIELD_URL, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        raw_bytes = resp.read()
    return parse_jpx_dividend_yield_xls(raw_bytes)


def fetch_fred_jp_equity(start: pd.Timestamp = START_DATE) -> pd.Series:
    """Monthly OECD Total Share Prices for All Shares for Japan (`SPASTT01JPM661N`) from FRED."""
    df = pd.read_csv(FRED_JP_EQUITY_URL)
    dates = pd.to_datetime(df["observation_date"], errors="coerce")
    vals = pd.to_numeric(df["SPASTT01JPM661N"], errors="coerce")
    valid = dates.notna() & vals.notna() & (vals > 0)
    series = pd.Series(
        vals[valid].values,
        index=_to_month_end(pd.DatetimeIndex(dates[valid])),
        name="jp_equity_price",
        dtype=float,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= start]


def fetch_japan_total_return(ey: pd.Series | None = None) -> pd.Series:
    """Monthly Japan equity cumulative total return index (1986–present).

    Uses Yahoo Finance `1305.T` (iShares Core TOPIX ETF `Adj Close`, 2008–present) spliced with pre-2008 monthly
    price returns from FRED `SPASTT01JPM661N` (fallback: Yahoo `^N225`) plus TSE 1st Section / Prime dividend yields
    from JPX (`historical-y.xls`, 1998+, backfilled pre-1998 from `0.28 * topix_earnings_yield`).
    """
    jp_start = pd.Timestamp("1986-01-01")
    etf_tr = pd.Series(dtype=float)
    try:
        etf_tr = _download_yf_monthly_close("1305.T", col="Adj Close", start=jp_start)
    except Exception as exc:
        log.warning("Could not download 1305.T total return series (%s)", exc)

    synth_tr = pd.Series(dtype=float)
    if etf_tr.empty or jp_start < etf_tr.index.min():
        price = pd.Series(dtype=float)
        try:
            price = fetch_fred_jp_equity(start=jp_start)
        except Exception as exc:
            log.warning("Could not fetch FRED Japan share price series (%s)", exc)
        try:
            n225 = _download_yf_monthly_close("^N225", col="Close", start=jp_start)
            price = _splice_total_return(price, n225)
        except Exception as exc:
            log.debug("Could not download ^N225 fallback (%s)", exc)

        dy = pd.Series(dtype=float)
        try:
            dy = fetch_japan_dividend_yield()
        except Exception as exc:
            log.warning("Could not fetch JPX dividend yield series (%s)", exc)
        if ey is not None and not ey.empty:
            est_dy = (pd.to_numeric(ey, errors="coerce") * 0.28).clip(lower=0.4, upper=2.5)
            dy = dy.combine_first(est_dy) if not dy.empty else est_dy
        if not price.empty:
            synth_tr = _build_synthetic_tr_index(price, dy)

    combined = _splice_total_return(etf_tr, synth_tr)
    combined = combined[combined.index >= jp_start].sort_index()
    combined.name = "total_return_index"
    return combined


def load_japan_yields(
    cache_path: Path,
    max_age_days: int = MACRO_MAX_AGE_DAYS,
    offline: bool = False,
) -> pd.DataFrame:
    """Load monthly Japan 10Y JGB yield, TSE Prime/1st-Sec earnings yield, spread (pp), and forward 3Y total return."""
    existing = pd.DataFrame()
    if cache_path.is_file():
        try:
            existing = pd.read_parquet(cache_path).sort_index()
        except Exception as exc:
            log.warning("Could not read cached Japan yields %s (%s)", cache_path, exc)

    has_cagr = "forward_3y_cagr" in existing.columns and existing["forward_3y_cagr"].notna().any()
    if not existing.empty and (offline or (_cache_is_fresh(cache_path, max_age_days) and has_cagr)):
        if offline or existing.index.min() <= pd.Timestamp("1986-08-31"):
            return _ensure_macro_columns(existing)

    if offline:
        return pd.DataFrame(columns=["jgb_10y", "topix_earnings_yield", "spread_pp", "forward_3y_cagr"])

    try:
        jgb = fetch_japan_jgb_history()
        ey = fetch_japan_earnings_yield()
        df = pd.concat([jgb, ey], axis=1)
        if not existing.empty:
            for col in ("jgb_10y", "topix_earnings_yield"):
                if col in existing.columns:
                    df[col] = df[col].combine_first(existing[col])
        df = df.dropna(subset=["jgb_10y", "topix_earnings_yield"]).sort_index()
        if df.empty:
            raise ValueError("Joined Japan yields DataFrame is empty")
        df["spread_pp"] = (df["topix_earnings_yield"] - df["jgb_10y"]).round(4)

        tr = pd.Series(dtype=float)
        try:
            tr = fetch_japan_total_return(ey=df["topix_earnings_yield"])
        except Exception as exc:
            log.warning("Could not refresh Japan total return series (%s)", exc)
        if not existing.empty and "total_return_index" in existing.columns:
            tr = _splice_total_return(tr, existing["total_return_index"])
        if not tr.empty:
            df["total_return_index"] = tr.reindex(df.index)
            cagr = compute_forward_3y_cagr(tr)
            df["forward_3y_cagr"] = cagr.reindex(df.index)
        elif not existing.empty and "forward_3y_cagr" in existing.columns:
            df["forward_3y_cagr"] = existing["forward_3y_cagr"].reindex(df.index)
        else:
            df["forward_3y_cagr"] = np.nan

        df.index.name = "date"
        _atomic_write_parquet(df, cache_path)
        log.info(
            "Japan yields updated (%d months: %s to %s)",
            len(df),
            df.index[0].strftime("%Y-%m"),
            df.index[-1].strftime("%Y-%m"),
        )
        return df
    except Exception as exc:
        log.warning("Could not refresh Japan yields (%s)", exc)
        if not existing.empty:
            return _ensure_macro_columns(existing)
        return pd.DataFrame(columns=["jgb_10y", "topix_earnings_yield", "spread_pp", "forward_3y_cagr"])
