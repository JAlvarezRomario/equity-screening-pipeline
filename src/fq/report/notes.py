"""The company page's explanations: owner-earnings captions, the inflation note and the data notes."""

import numpy as np
import pandas as pd

from fq.ranking import RANK_GROUP_LABELS, RANKED
from fq.report.format import chart_years, period_label, ttm_duplicates_latest
from fq.settings import Settings


def _recent_splits(row: pd.Series, codes: list[str]) -> str:
    """Only the splits inside the reported window (Yahoo lists the whole history)."""
    ends = pd.to_datetime([row.get(f"PeriodEnd_{c}") for c in codes], errors="coerce").dropna()
    start = ends.min() if len(ends) else pd.Timestamp.min
    items = [s.strip() for s in str(row.get("Splits") or "").split(";") if s.strip()]
    recent = [s for s in items if pd.to_datetime(s.split(" ")[0], errors="coerce") > start]
    return ", ".join(recent) or ", ".join(items[-1:])


# What a share's owner earns, by business model: (legend label, definition and why). One source for the chart's
# legend and the caption under it (user's request: say what is shown and why).
# Banks, insurers, holdings and health insurers keep their currency's inflation on their book
# (`flows._capital_retention`; the rate is in the data notes, `inflation_note`).
_RETAINED = " Less book value x inflation in its currency: the capital it keeps to stay the same size in real terms."
OWNER_EARNINGS = {
    "operating": (
        "Owner FCF",
        "cash from operations − capex − stock pay − lease payments − interest, with the capex and working capital "
        "that bought growth beyond inflation added back (and the cash suppliers or customers funded it with taken "
        "out): growth is an investment for later, not a cost of today's earnings, nor a source of them.",
    ),
    "asset_heavy": (
        "Owner FCF (upkeep capex = depreciation)",
        "cash from operations − capex up to depreciation (what keeps the buildings, networks or plant as they "
        "are) − stock pay − lease payments − interest; capex above depreciation buys growth.",
    ),
    "bank": (
        "Clean net income",
        "net income without one-offs. A bank's cash flow mixes in deposits and loans, which are its clients' "
        "money, not the owner's." + _RETAINED,
    ),
    "insurer": (
        "Clean net income",
        "net income without one-offs. Premiums held before claims are paid (float) are policyholders' money, so "
        "cash flow is not used. Investment gains may be inside (Yahoo does not separate them), so Value takes the "
        "lower of the median and mean year." + _RETAINED,
    ),
    "float": (
        "Net income",
        "net income: premiums held before claims are paid are members' money, so cash flow is not used." + _RETAINED,
    ),
}
OWNER_EARNINGS["client_cash"] = (
    OWNER_EARNINGS["operating"][0],
    OWNER_EARNINGS["operating"][1] + " Client cash is not counted as the owner's.",
)
OWNER_EARNINGS["holding"] = (
    "Clean net income",
    "net income without one-offs, including the share of companies it holds stakes in (look-through). Investment "
    "gains may be inside (Yahoo does not separate them), so Value takes the lower of the median and mean year."
    + _RETAINED,
)


def _owner_earnings(row: pd.Series) -> tuple[str, str]:
    model = row.get("Business_Model")
    return OWNER_EARNINGS.get(model if isinstance(model, str) else "operating", OWNER_EARNINGS["operating"])


def owner_earnings_caption(row: pd.Series) -> list[tuple[str, str]]:
    """(element, explanation) lines under the earnings chart, in "How these figures are built"."""
    label, definition = _owner_earnings(row)
    financial = row.get("Financial_Mode") == True  # noqa: E712
    return [
        (
            "Blue bars",
            f"owner earnings per share ÷ today's price, each year: what one share earned for its owner after paying "
            f"lenders, at that year's share count. Here: {label.lower()} = {definition}",
        ),
        (
            "Amber line",
            "earnings power per share, normalised: the level Value relies on, a typical year from this history "
            "(the latest if it rose or fell every year, mid-cycle for cyclicals, capped after a lasting profit fall), "
            "less today's interest. Value's score uses the same earnings before interest, against the price of the "
            "whole business (debt and spare cash at face).",
        ),
        (
            "Other bars and lines",
            "grey bars: clean net income per share ÷ price (accounting profit, to compare with the cash); "
            + ("" if financial else "red: capex; ")
            + "violet: dividends. Dotted lines: 6% and 15% yields, for scale. "
            + ("Return on equity" if financial else "ROIC")
            + " and operating margin are in the returns chart.",
        ),
    ]


def inflation_note(row: pd.Series, cfg: Settings) -> str | None:
    """How returns were put in US terms (`metrics.inflation`): only where it moved them by a point or more, or
    where restated (IAS 29) statements were left as they are."""
    currency = row.get("financialCurrency")
    currency = currency if isinstance(currency, str) else row.get("currency")
    if row.get("Inflation_Basis") == "restated":
        note = (f"Statements in {currency} restated for inflation (IAS 29): returns are already real, so US inflation "
                "is added back to read them in US terms like every other company's")  # fmt: skip
        nominal, real = (pd.to_numeric(row.get(f"ROIC{s}_{cfg.latest}"), errors="coerce") for s in ("_Nominal", ""))
        if pd.notna(nominal) and pd.notna(real):
            note += f" (reported {nominal:.1f}% counts as {real:.1f}%)"
        return note + "."
    excess = pd.to_numeric(row.get(f"Excess_Inflation_{cfg.latest}"), errors="coerce")
    if pd.isna(excess) or abs(excess) < 0.01:
        return None
    word = "above" if excess > 0 else "below"
    note = (f"Returns put in US terms: {currency} inflation ran {abs(excess):.0%} a year {word} US inflation over "
            "these years (IMF)")  # fmt: skip
    nominal, real = (pd.to_numeric(row.get(f"ROIC{s}_{cfg.latest}"), errors="coerce") for s in ("_Nominal", ""))
    if pd.notna(nominal) and pd.notna(real):
        what = "ROE" if row.get("Financial_Mode") == True else "ROIC"  # noqa: E712
        note += f", so a reported {what} of {nominal:.1f}% ({period_label(row, cfg.latest)}) counts as {real:.1f}%"
    retained = pd.to_numeric(row.get("Retention_Rate"), errors="coerce")
    if (row.get("Financial_Mode") == True or row.get("Business_Model") == "float") and pd.notna(retained):  # noqa: E712
        note += f"; it keeps {retained:.1f}% of its book a year to stay the same size"
    return note + "."


def data_notes(row: pd.Series, cfg: Settings, periods: list[str]) -> list[str]:
    """Data-quality remarks (also drawn on the chart)."""
    notes = []
    codes = chart_years(row, periods)
    if row.get("Industry_Excluded_Kept") == True:  # noqa: E712
        notes.append("Excluded industry or financial business, kept because it is a portfolio holding.")
    if isinstance(row.get("Model_Review"), str):
        notes.append(
            f"Measured as {row.get('Business_Model')}, but its statements suggest another model may fit: "
            f"{row['Model_Review']}. Out of the highlight lists until the model is confirmed in model_overrides.txt."
        )
    model = row.get("Business_Model")
    if model == "float":
        notes.append(
            "Measured as an insurer with policyholder float: its cash backs claims, so it is not subtracted "
            "from EV or invested capital, and 'FCF' = net income. Debt still counts."
        )
    elif model == "client_cash":
        notes.append(
            "Holds client money (payroll, clearing or payment balances): that cash is not subtracted from EV "
            "or invested capital. Otherwise measured like any operating company."
        )
    elif model == "asset_heavy":
        notes.append(
            "Measured as asset-heavy (REIT or utility): FCF deducts only capex up to depreciation; the capex "
            "above it buys growth, so it is left out of earnings power. "
            "Percentiles come from the other REITs and utilities."
        )
    if note := inflation_note(row, cfg):
        notes.append(note)
    if (pd.to_numeric(pd.Series([row.get(f"ROIC_{p}") for p in periods]), errors="coerce") >= cfg.roic_cap).any():
        notes.append(f"ROIC capped at {cfg.roic_cap:.0f}% (little or no invested capital); higher says nothing more.")
    if isinstance(row.get("Sister_Class"), str):
        notes.append(
            f"Other share class {row['Sister_Class']} (same company, same statements) is not ranked separately."
        )
    if row.get("Financial_Mode") == True:  # noqa: E712
        kind = {"bank": "a lender", "insurer": "an insurer", "holding": "a holding company"}.get(model, "a financial")
        notes.append(
            f"Measured as {kind}: deposits and float are not debt, so EV = market cap and 'ROIC' = return on "
            "equity. Quality uses ROE, return on assets, ROE stability, equity / assets and share "
            "reduction against its peer group, and book value per share compounding and the worst revenue year "
            "against every company. Read the NAV chart and book value too."
        )
    group = row.get("Rank_Group")
    if isinstance(group, str) and group != RANKED:
        why = {
            "financial": "banks and insurers are judged against their own kind, in their own lists",
            "holding": "a holding is judged on its book, in the Below book list",
            "real_estate": "a REIT is judged against other REITs, in the Real estate list",
            "utility": "a utility is judged against other utilities, in the Utilities list",
            "apart": "its business model is not ranked yet",
            "insufficient": "it lacks the data for a Value or a Quality score",
            "unprofitable": "its owner earnings or its return on capital are not positive in a normal year",
        }.get(group, RANK_GROUP_LABELS.get(group, group))
        notes.append(
            f"Scored but not in the rankings: {why}. The rankings and the quadrant hold profitable operating "
            "companies only, so a score here is read against its own kind."
        )
    if ttm_duplicates_latest(row):
        notes.append(f"TTM matches the latest annual report ({period_label(row, cfg.latest)}); TTM is hidden.")
    elif pd.isna(row.get("Total Revenue_TTM")):
        notes.append("No trailing-twelve-month data available (needs four consecutive quarters).")
    if pd.isna(row.get(f"Total Revenue_{cfg.latest}")):
        notes.append("No annual revenue reported in the latest slot.")
    if row.get("Stale_Annual") == True:  # noqa: E712
        notes.append(f"Latest annual report is old ({period_label(row, cfg.latest)}); newest figures are TTM only.")
    if row.get("Earnings_Collapse") == True:  # noqa: E712
        notes.append(
            f"TTM profit is more than {cfg.earnings_collapse_pct:.0f}% below the latest annual report: Value's "
            "normal earnings are capped at the TTM (the cut is not assumed to reverse)."
        )
    missing = [period_label(row, c) if pd.notna(row.get(f"PeriodEnd_{c}")) else c for c in codes
               if c != "TTM" and pd.isna(row.get(f"Total Revenue_{c}"))]  # fmt: skip
    if missing:
        notes.append(f"No data from Yahoo for: {', '.join(missing)}.")

    if isinstance(row.get("Mood_Check"), str):
        notes.append(row["Mood_Check"])
    acq, fcf = (
        pd.to_numeric(row.get("Acquisition_Yield"), errors="coerce"),
        pd.to_numeric(row.get("FCFy_Norm"), errors="coerce"),
    )
    if pd.notna(acq) and pd.notna(fcf) and acq > 0.2 * abs(fcf) and acq > 0:
        notes.append(
            f"Acquisitions average {acq:.1f}% of market cap a year (normalised FCF yield {fcf:.1f}%): free cash flow "
            "does not deduct them, so part of the revenue growth was bought rather than built."
        )
    if row.get("Lease_Adjusted") == True:  # noqa: E712 - handles NaN / numpy bool
        if row.get("Accounting_Standard") == "ifrs":
            notes.append("Lease liabilities are left out of debt; reported under IFRS, so the lease liability due "
                         "within 12 months is deducted from free cash flow as a year's lease cost.")  # fmt: skip
        else:
            notes.append("Lease liabilities are left out of debt: under US GAAP the rent is already in operating "
                         "profit and cash flow.")  # fmt: skip
    if isinstance(row.get("Anomaly_Notes"), str):
        notes.append(row["Anomaly_Notes"])
    if isinstance(row.get("Suspect_Reasons"), str):
        notes.append(f"Data suspect ({row['Suspect_Reasons']}): ranked, but left out of the top lists until checked.")
    if row.get("Insufficient_Data") == True:  # noqa: E712
        notes.append("Too little data for a reliable Value or Quality rank; listed after the ranked companies.")
    if row.get("Cyclical") == True:  # noqa: E712
        why = (
            "a cyclical industry"
            if row.get("Cyclical_Industry") == True  # noqa: E712
            else "operating profit changes sign or the margin swings widely"
        )
        notes.append(
            f"Cyclical ({why}): a few years of history may not cover a full cycle, so Value uses a mid-cycle "
            "FCF margin; yields can still look cheap near a peak."
        )
    if row.get("Peak_Margin") == True:  # noqa: E712
        notes.append(
            f"Peak margins: the latest operating margin is {row.get('Peak_Margin_Ratio'):.1f}x its weaker years'; "
            "the cycle may be at its top, so it is scored but left out of the top lists."
        )
    if row.get("Distress") == True:  # noqa: E712
        notes.append(
            f"Distress: Altman Z'' {row.get('Altman_Z'):.2f} with negative free cash flow or thin interest cover; "
            "scored, but left out of the top lists and the mood list (cheap may mean failing)."
        )
    ratio = pd.to_numeric(row.get("Share_Basis_Ratio"), errors="coerce")
    if pd.notna(ratio):
        notes.append(f"Price is quoted per {ratio:.4g} statement shares (ADR / share class); share counts rescaled.")
    if pd.notna(pd.to_numeric(row.get("Share_Class_Factor"), errors="coerce")):
        notes.append(
            f"Two share classes with different economic rights (one {row.get('Share_Class_Sister')} share = "
            f"{row.get('Share_Class_Ratio'):g} shares of this class): market cap and per-share figures count all "
            "shares in this class's units."
        )
    if row.get("TTM_Source") == "yahoo_ttm":
        notes.append("TTM taken from Yahoo's TTM statement (no four consecutive quarters, e.g. half-yearly reporter).")
    gap = pd.to_numeric(row.get("TTM_Check_Gap"), errors="coerce")
    if row.get("TTM_Source") == "yahoo_ttm_checked":
        notes.append(f"The last four 'quarters' add up to a figure {gap:.0%} away from Yahoo's TTM (often a "
                     "half-year reported as one quarter); Yahoo's TTM is used, being closer to the annual report.")  # fmt: skip
    elif row.get("TTM_Source") == "quarters_checked":
        notes.append(f"Yahoo's TTM is {gap:.0%} away from the sum of the last four quarters; the quarterly sum is "
                     "used, being closer to the annual report.")  # fmt: skip
    elif pd.notna(gap) and gap > 0.02:
        notes.append(f"TTM revenue: quarterly sum and Yahoo's TTM differ by {gap:.0%}; treat TTM with care.")
    if isinstance(row.get("CapEx_Estimated"), str):
        labels = ", ".join(period_label(row, p) for p in row["CapEx_Estimated"].split(", "))
        notes.append(f"CapEx estimated from the median CapEx margin (not reported) for: {labels}.")
    if isinstance(row.get("Aliases_Used"), str):
        notes.append(f"Substitute Yahoo rows used: {row['Aliases_Used']}.")
    if "stale" in str(row.get("FX_Source", "")):
        notes.append(f"Currency conversion used {row['FX_Source']} exchange rates.")
    if isinstance(row.get("Split_Adjusted_Periods"), str):
        notes.append(f"Share counts restated for stock splits ({_recent_splits(row, codes)}).")

    prev = np.nan
    for c in codes:
        s = row.get(f"Ordinary Shares Number_{c}")
        if pd.isna(s):
            s = row.get(f"Share Issued_{c}")
        if pd.notna(prev) and pd.notna(s) and prev > 0 and (s - prev) / prev > 0.05:
            notes.append(f"Dilution: shares increased {(s - prev) / prev * 100:.1f}% in {period_label(row, c)}.")
        prev = s
    return notes
