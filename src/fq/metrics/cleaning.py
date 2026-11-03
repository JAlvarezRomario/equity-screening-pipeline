"""Implausible Yahoo figures: unit errors and TTM artefacts, checked before any metric."""

import logging

import numpy as np
import pandas as pd

from fq.metrics.common import _col, _num, _slot
from fq.settings import Settings
from fq.universe import business_model

log = logging.getLogger(__name__)


ANOMALY_COLUMNS = [
    "Total Revenue", "Operating Income", "EBIT", "Net Income", "Interest Expense",
    "Pretax Income", "Tax Provision", "Free Cash Flow", "Operating Cash Flow",
]  # fmt: skip


INTEREST_COLUMNS = ("Interest Expense",)


CASH_FLOW_COLUMNS = ("Operating Cash Flow", "Free Cash Flow")


# Operating cash flow mixes in deposit, loan or claims-reserve flows: its swings say nothing about the data.
OCF_UNRELIABLE_MODELS = ("bank", "insurer", "holding", "float")


def _anomaly(row: pd.Series, period: str, prior: str, check_cash_flow: bool = True) -> tuple[str, tuple] | None:
    """(why `period` looks implausible next to `prior`, the columns that reason questions), or None.

    A revenue or EBIT jump questions the whole income statement (a period artefact); interest that jumps
    alone questions only interest (tonies: other financial costs booked as interest); a cash-flow drop only
    the cash flows (a working-capital swing does not make revenue wrong). "Flat debt" rules only fire when
    debt is reported for both periods (missing debt is not flat debt), and EBIT growth is only measured
    between two profits (two losses give a meaningless ratio).
    """
    rev, rev0 = _num(row, f"Total Revenue_{period}"), _num(row, f"Total Revenue_{prior}")
    intr, intr0 = _num(row, f"Interest Expense_{period}"), _num(row, f"Interest Expense_{prior}")
    ebit, ebit0 = _num(row, f"EBIT_{period}"), _num(row, f"EBIT_{prior}")
    debt, debt0 = _num(row, f"Total Debt_{period}"), _num(row, f"Total Debt_{prior}")
    ocf, ocf0 = _num(row, f"Operating Cash Flow_{period}"), _num(row, f"Operating Cash Flow_{prior}")
    debt_ratio = debt / debt0 if pd.notna(debt) and pd.notna(debt0) and debt0 > 1e6 else np.nan

    def flat(limit: float) -> bool:
        return pd.notna(debt_ratio) and debt_ratio < limit

    if pd.notna(rev) and pd.notna(rev0) and rev0 > 1e6:
        ratio = rev / rev0
        if (ratio > 1.5 or ratio < 0.6) and flat(1.2):
            return f"revenue x{ratio:.1f} with flat debt", tuple(ANOMALY_COLUMNS)
    if pd.notna(ebit) and pd.notna(ebit0) and ebit0 > 1e6 and ebit > 0 and ebit / ebit0 > 3.0 and flat(1.5):
        return f"EBIT x{ebit / ebit0:.1f} with flat debt", tuple(ANOMALY_COLUMNS)
    if pd.notna(intr) and pd.notna(intr0) and intr0 > 1e6 and intr / intr0 > 1.8 and flat(1.3):
        return f"interest x{intr / intr0:.1f} with flat debt", INTEREST_COLUMNS
    if check_cash_flow and pd.notna(ocf) and pd.notna(ocf0) and ocf0 > 1e6 and ocf / ocf0 < 0.10:
        return f"operating cash flow x{ocf / ocf0:.2f}", CASH_FLOW_COLUMNS
    # A jump is as suspect as a drop: one quarter of customer advances (Indra Q1-26: EUR 1.4bn of working
    # capital, 4x a year's cash flow) is real cash, but not a year's worth to normalise on. Measured against
    # the better of the two years before, so a recovery from one bad year (LEN) is not a jump.
    before = f"Y{_slot(prior) + 1}"  # the year before `prior` (TTM vs Y0: Y1)
    base = np.nanmax([ocf0, _num(row, f"Operating Cash Flow_{before}")]) if pd.notna(ocf0) else np.nan
    if check_cash_flow and pd.notna(ocf) and pd.notna(base) and base > 1e6 and ocf / base > 3.0:
        return f"operating cash flow x{ocf / base:.1f}: a one-off such as customer advances?", CASH_FLOW_COLUMNS
    return None


def clean_anomalies(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Check each period against the one before it (TTM vs Y0, Y0 vs Y1).

    Only TTM is ever blanked: that is where Yahoo artefacts occur (summed cumulative interim
    reports, partial years), and a TTM confirmed by both routes (quarterly sum = Yahoo's TTM) is
    trusted. Annual figures are audited, so an unusual annual move is only noted in
    `Anomaly_Notes` for the reader to check - real events (a console launch, an impairment, an
    acquisition) look exactly like that.
    """
    out = df.copy()
    out["Anomaly_Notes"] = None
    blanked = 0
    for idx, row in out.iterrows():
        notes = []
        check_cash_flow = business_model(row["Ticker"], row.get("industry")) not in OCF_UNRELIABLE_MODELS
        for period, prior in ((cfg.latest, cfg.prior), ("TTM", cfg.latest)):
            found = _anomaly(row, period, prior, check_cash_flow)
            if found is None:
                continue
            reason, columns = found
            gap = pd.to_numeric(row.get("TTM_Check_Gap"), errors="coerce")
            confirmed = pd.notna(gap) and gap <= 0.02
            if period == "TTM" and not confirmed:
                which = "income data" if columns == tuple(ANOMALY_COLUMNS) else ", ".join(columns)
                log.warning("%s TTM: %s; blanking TTM %s", row["Ticker"], reason, which)
                notes.append(f"TTM {which} removed as implausible vs latest annual ({reason}).")
                blanked += 1
                for base in columns:
                    if f"{base}_TTM" in out.columns:
                        out.at[idx, f"{base}_TTM"] = np.nan
            else:
                which = "latest annual report" if period != "TTM" else "TTM"
                log.info("%s %s: unusual move (%s); kept", row["Ticker"], period, reason)
                notes.append(f"Unusual move in the {which} vs the period before ({reason}); kept, worth checking.")
        out.at[idx, "Anomaly_Notes"] = " ".join(notes) or None
    if blanked:
        log.info("Blanked %d implausible TTM periods", blanked)
    return out


def validate_latest_year(df: pd.DataFrame, concept: str, cfg: Settings) -> pd.DataFrame:
    """Blank the latest year when it is negative revenue or a 100x unit error vs the prior year."""
    cur, prev = f"{concept}_{cfg.latest}", f"{concept}_{cfg.prior}"
    if cur not in df.columns or prev not in df.columns:
        return df
    df[prev] = pd.to_numeric(df[prev], errors="coerce")
    df[cur] = pd.to_numeric(df[cur], errors="coerce")
    if concept == "Total Revenue":
        df.loc[df[cur] < 0, cur] = np.nan
    ratio = df[cur] / df[prev].replace(0, np.nan)
    unit_error = ((ratio.abs() > 100) | (ratio.abs() < 0.01)) & (df[cur].abs() > 1_000_000)
    df.loc[unit_error, cur] = np.nan
    return df


def repair_equity(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Read Yahoo's 0.0 `Stockholders Equity` placeholder as missing and take the equity the statement does report.

    Some balance sheets (PXED, BEP, BIP) carry an exact 0 next to a positive `Common Stock Equity`: as capital
    it made ROIC the cap (profit on no capital) and Quality the best in the market. The parent's equity is then
    common + preferred, else total equity less minorities. A real deficit is never exactly 0, so it stays.
    """
    out = df.copy()
    notes = {}
    for period in cfg.fiscal_years:
        col = f"Stockholders Equity_{period}"
        if col not in out.columns:
            continue
        c = lambda name, p=period: _col(out, f"{name}_{p}")  # noqa: E731
        common = c("Common Stock Equity") + c("Preferred Stock Equity").fillna(0)
        gross = c("Total Equity Gross Minority Interest") - c("Minority Interest").fillna(0)
        reported = common.where(common.notna() & (common != 0), gross.where(gross != 0))
        placeholder = (_col(out, col) == 0) & reported.notna()
        out.loc[placeholder, col] = reported[placeholder]
        for idx in out.index[placeholder]:
            notes.setdefault(idx, []).append(period)
    for idx, periods in notes.items():
        note = f"Stockholders' equity read as 0 in {', '.join(periods)}; taken from the equity reported beside it."
        out.at[idx, "Anomaly_Notes"] = " ".join(filter(None, [out.at[idx, "Anomaly_Notes"], note]))
    if notes:
        log.info("Equity placeholder (0) replaced for %d companies", len(notes))
    return out
