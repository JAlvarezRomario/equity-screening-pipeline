"""Display formatting for the HTML site (no markup here): numbers, the ranking table's columns and their help,
and the report periods shared by charts and tables. Company-page pieces: `tiles`, `cards`, `notes`."""

import pandas as pd

from fq.report.links import slug
from fq.report.pills import list_pills
from fq.settings import Settings, display_name

NA = "N/A"
DASH = "–"  # a missing value in a table: recedes where "N/A" shouts (the CSV keeps blanks)


def fmt(value, spec: str = ".2f", scale: float = 1.0, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value) or value == "":
            return NA
        return f"{float(value) * scale:{spec}}{suffix}"
    except (TypeError, ValueError):
        return NA


# (label, column, format spec, scale, suffix). The combined ranking: the two scores its order sums, earnings power
# (Value's main criterion) and a little context (user's choice, Oct 21, 2026: the other criteria are on each company
# page and in full_analysis.csv, not in this table).
SUMMARY_COLUMNS = [
    ("Value Score", "Value_Score", ".0f", 1, ""),
    ("Quality Score", "Quality_Score", ".0f", 1, ""),
    ("Owner Yield", "Owner_Yield", ".1f", 1, "%"),
    ("Return Quality", "Return_Quality", ".1f", 1, "%"),
    ("Market Cap (B$)", "Market Cap", ",.1f", 1, ""),
    ("Analysts", "numberOfAnalystOpinions", ".0f", 1, ""),
    ("Above 52w Low", "Above_52W_Low", ".0f", 1, "%"),
    ("Dividend WHT", "Dividend_WHT", "g", 1, "%"),
]  # fmt: skip

# First column of each group in a table: a hairline separates scores, earnings power and context.
GROUP_STARTS = frozenset({"Value_Score", "Owner_Yield", "Market Cap"})

# One line per column for the header tooltips and the company page: what it is and its unit. Whether higher or
# lower is better and the weight are added by `column_help` from the criteria in settings, so they cannot drift.
COLUMN_HELP = {
    "Value_Score": "Value score: weighted percentile (0-100) of the Value criteria against the benchmark; 50 = median.",
    "Quality_Score": "Quality score: weighted percentile (0-100) of the Quality criteria, across the market (banks, "
                     "insurers and holdings against their own kind); 50 = median.",
    "Rank": "Combined rank: the worse of the Value and Quality positions (ties: their sum), lowest first.",
    "Value_Rank": "Position on Value among the ranked companies.",
    "Quality_Rank": "Position on Quality among the ranked companies.",
    "Owner_Yield": "Owner yield: normalised owner earnings per share projected 3 years forward (crediting smooth real trend, charging declines; none for cyclicals) over today's price, %.",
    "Return_Quality": "Return quality: normalised ROIC (ROE for financials; banks at a standard 8% equity/assets) adjusted for its smooth trend over 3 years, shrunk by variability, with a haircut for severe leverage, %.",
    "Owner_Trend": "Annual per-share real growth trend of owner earnings before interest, in constant currency, net of local inflation, x its R², %/yr.",
    "Owner_Smoothness": "Smoothness (R²) of the per-share owner earnings trend (0 = erratic, 1 = perfectly smooth).",
    "ROIC_Slope": "Annual trend of ROIC/ROE, percentage points per year.",
    "Quality_Haircut": "Factor applied to return on capital for leverage (1.0 = safe, 0.75 = moderate, 0.50 = severe); for banks, equity/assets over the standard 8%.",
    "Archetype_Source": "Provenance of business archetype: override, signature (claims, lender, fee network, client cash, plant) or industry.",
    "EPV_Yield": "Earnings power yield: normalised owner earnings at zero real growth, before interest, over the "
                 "value of the business (shares + debt at face - spare cash), %. Banks, insurers, holdings: net "
                 "income over market cap.",
    "Aftertax_Yield_Avg": "Clean income before interest, after tax, over the value of the business, %: what the owner "
                          "keeps.",
    "ROIC_Norm": "Normalised return on invested capital (cash not subtracted), %, in US terms (local inflation "
                 "beyond US inflation taken out). ROE for banks, insurers and holdings.",
    "ROIC_Variability": "ROIC spread around its own trend line, relative to its level, x: a steady moat.",
    "Gross_Margin_StdDev": "Spread of the gross margin over the years, percentage points: pricing power.",
    "Gross_Margin_Held": "Newest gross margin less the margin of the year with the highest US inflation, percentage "
                         "points: costs passed on (pricing power). Shown and tested by lists, never scored.",
    "Gross_Profitability_Avg": "Gross profit / operating assets (spare cash left out), %.",
    "Revenue_Worst_Change": "Worst yearly revenue change, %: does demand hold in a bad year?",
    "FCF_Positive_Share": "Share of years with positive free cash flow (all capex charged), %.",
    "Per_Share_Compounding": "USD revenue per share trend x its R², % a year (banks, insurers, holdings: book value "
                             "per share): the holder's share growing.",
    "Incremental_ROIC": "Extra after-tax operating profit per extra invested capital, oldest year to the latest, %, "
                        "in US terms: growth that earns its keep.",
    "ROIC_Ex_Goodwill": "Normalised ROIC without goodwill and intangibles in invested capital, %: the operating "
                        "return apart from what acquisitions cost (ROIC with them judges that spending).",
    "ROIC_Trend": "Mean ROIC of the latest year and TTM less that of three and four years ago, percentage points.",
    "NetDebt_to_EBIT": "Net debt / median EBIT, x (net cash = 0; net debt with no operating profit = 99).",
    "FCF_Conversion": "Free cash flow / net income, median, x: profits that are cash.",
    "SBC_to_Revenue": "Stock-based pay / revenue, median, %: managers paid in the owners' shares.",
    "ROA_Avg": "Return on assets, %.",
    "Equity_to_Assets": "Common equity / total assets, %: a lender's leverage.",
    "Share_Reduction_Rate": "Shares retired per year, %; negative = dilution.",
    "Owner_Earnings_Compounding": "USD owner earnings per share trend x its adjusted R², % a year: what a share earns, "
                                  "compounding. Shown and used by the Compounders list, never scored.",
    "Shares_Change_10Y": "Share count change per year over up to 10 years of Yahoo's long share series, %: negative = "
                         "buybacks. Shown, never scored.",
    "Dividend_Years_No_Cut": "Consecutive complete years, back from the last, that paid a dividend no more than 5% "
                             "below the year before. Shown, never scored.",
    "DPS_Growth_10Y": "Dividend per share growth per year over 10 years, quote currency, %. Shown, never scored.",
    "DPS_Growth_20Y": "Dividend per share growth per year over 20 years, quote currency, %. Shown, never scored.",
    "Market Cap": "Market capitalisation, billions of USD.",
    "numberOfAnalystOpinions": "Analysts covering the company (Yahoo).",
    "Above_52W_Low": "Price above its 52-week low, % (0 = at the low). Shown, never scored.",
    "Dividend_WHT": "Statutory dividend withholding tax on a non-resident individual, before treaties or refunds, by "
                    "country (lists/dividend_wht.txt), %. Shown, never scored.",
    "Dividend_Yield": "Cash dividends paid over the last twelve months / market cap, %. Shown, never scored.",
    "Revenue_Growth_Steady": "USD revenue growth: the latest year's when it rose every year (plus the step-up when "
                             "growth sped up every year), else the trend x its R², %.",
    "Revenue_Change_Recent": "Revenue change, latest twelve months vs the year before, annualised, %.",
    "EBIT_Change_Recent": "Operating profit change, latest twelve months vs the year before, annualised, %.",
    "FCFy_Norm": "Normalised free cash flow yield (all capex charged), %.",
    "Effective_NAV_to_Price": "Tangible book value per share / share price (common book when intangibles are not "
                              "reported), enterprise-de-levered when in net debt so leverage cannot flatter the "
                              "discount, %.",
    "NAV_to_Price": "Tangible book value per share / share price, %.",
    "NCAV_to_Price": "Net current asset value (current assets less every liability, preferred and minorities) per "
                     "share / share price, %: 100 or more = the fixed assets come at no cost (N/A "
                     "for banks, insurers and holdings).",
    "BV_to_Price": "Common book value per share / share price, %.",
    "Repeat_Count": "How many highlight lists the company qualifies for (hover the number for which): every company a screened list admits, and the 20 shown by Cheap and good, Cheapest and Best businesses.",
    "Net_Cash_to_Cap": "Shareholder-basis net cash (spare cash less debt and preferred stock) / market cap, % "
                       "(negative = net debt; N/A for banks, insurers and holdings).",
    "Book_Per_Share_Compounding": "USD book value per share trend x its R², % a year.",
    "Dividend_Unbroken_Years": "Calendar years back from the last complete one with the regular dividend per share "
                               "raised or held (specials left out, payment timing allowed for, in the declared "
                               "currency). Shown, never scored.",
    "Recession_Dividend_Note": "How the regular dividend per share (specials left out, declared currency) went through "
                               "each global recession on file (lists/recessions.txt): its lowest two-year mean over "
                               "the recession and the two years after, against the two years before. 2020 is not "
                               "judged for banks and insurers, whose supervisors suspended dividends. Shown, and "
                               "tested by the 20-year lists; never scored.",
    "Recession_Dividend_Fall": "The worst fall of the regular dividend through a global recession on file, %. Shown, "
                               "never scored.",
    "Regular_DPS_Growth_10Y": "Regular dividend per share (specials left out, declared currency), % a year over 10 "
                              "years. Shown, never scored.",
    "heldPercentInsiders":"Shares held by insiders (managers, directors, founding families, a parent company), % "
                           "(Yahoo). Shown, never scored.",
}  # fmt: skip


def _weights() -> dict[str, list[tuple[str, str, float]]]:
    """column -> [(dimension, direction, weight)] for operating and financial mode."""
    s, out = Settings(), {}
    for dim, crits in (("Value", s.value_criteria), ("Quality", s.quality_criteria)):
        for col, direction, w in crits:
            out.setdefault(col, []).append((dim, direction, w))
    for dim, crits in (("Value", s.financial_value_criteria), ("Quality", s.financial_quality_criteria)):
        for col, direction, w in crits:
            if not any(d == dim and x == w for d, _, x in out.get(col, [])):
                out.setdefault(col, []).append((f"{dim} (banks, insurers, holdings)", direction, w))
    return out


def column_help(col: str) -> str:
    """`COLUMN_HELP` plus, for a scored criterion, its dimension, weight and which way is better."""
    text = COLUMN_HELP.get(col, "")
    roles = _weights().get(col)
    if not roles:
        return text
    better = "Higher is better." if roles[0][1] == "top" else "Lower is better."
    weights = "; ".join(f"{dim}, weight {w:g}" for dim, _, w in roles)
    return f"{text} {better} Scored: {weights}.".strip()


def first_sort(col: str) -> str:
    """Direction of a header's first click: best first for a criterion, highest first for other numbers."""
    roles = _weights().get(col)
    return "asc" if roles and roles[0][1] == "bottom" else "desc"


def headers(columns, groups: bool = False) -> list[dict]:
    """Header cells: label, tooltip, first sort direction and group class."""
    return [{"label": label, "help": column_help(col), "first": first_sort(col),
             "cls": "gs" if groups and col in GROUP_STARTS else ""}
            for label, col, *_ in columns]  # fmt: skip


def benchmark(df: pd.DataFrame) -> tuple[str, int] | None:
    """(display name, companies) of the benchmark the scores are relative to, or None for a run ranked on its own."""
    if "Benchmark" not in df.columns or df["Benchmark"].dropna().empty:
        return None
    return display_name(str(df["Benchmark"].dropna().iloc[0])), int(df["Benchmark_N"].dropna().iloc[0])


def benchmark_note(df: pd.DataFrame) -> str:
    """One sentence on what the scores and ranks are relative to."""
    bench = benchmark(df)
    if bench is None:
        return "Scores (0-100) are percentiles within this run; ranks are within this run."
    name, n = bench
    return (f"Scores (0-100) are percentiles against the {name} benchmark ({n} companies), so they match the "
            f"{name} report; ranks are within this group of {len(df)}.")  # fmt: skip


def summary_columns(df: pd.DataFrame) -> list[tuple]:
    """SUMMARY_COLUMNS plus the position in the benchmark, when the run has one."""
    bench = benchmark(df)
    if bench is None:
        return SUMMARY_COLUMNS
    extra = [(f"Value # in {bench[0]}", "Value_Rank_Bench", ",.0f", 1, ""),
             (f"Quality # in {bench[0]}", "Quality_Rank_Bench", ",.0f", 1, "")]  # fmt: skip
    at = [c[1] for c in SUMMARY_COLUMNS].index("Quality_Score") + 1
    return [*SUMMARY_COLUMNS[:at], *extra, *SUMMARY_COLUMNS[at:]]


SCORE_TINTS = {"Value_Score": "tint-v", "Quality_Score": "tint-q"}
# Changes coloured green when up and red when down: only where up is plainly better (not accruals or asset growth),
# plus the 6-month price momentum (user's choice, Nov 1, 2026: shown, never scored).
GOOD_WHEN_UP = frozenset({"Momentum_6M", "Revenue_Worst_Change", "Per_Share_Compounding", "ROIC_Trend", "Share_Reduction_Rate",
                          "Revenue_Change_Recent", "EBIT_Change_Recent", "EPS_Rev_90d", "Revenue_Growth_Steady",
                          "Book_Compounding", "Book_Per_Share_Compounding", "Net_Cash_to_Cap",
                          "Owner_Earnings_Compounding", "DPS_Growth_10Y", "DPS_Growth_20Y"})  # fmt: skip


def table_cell(value, col: str, spec: str, scale: float, suffix: str, groups: bool = False) -> dict:
    """One table cell: text, CSS classes and, for a score, its 0-100 value for the tint."""
    text = fmt(value, spec, scale, suffix)
    cls = ["num"]
    if groups and col in GROUP_STARTS:
        cls.append("gs")
    if text == NA:
        return {"text": DASH, "cls": " ".join([*cls, "na"]), "s": None}
    number = float(value)
    if col in SCORE_TINTS:
        cls.append(SCORE_TINTS[col])
    elif col in GOOD_WHEN_UP and round(number * scale, 1) != 0:  # a change: its sign is the message
        cls.append("pos" if number > 0 else "neg")
    return {"text": text, "cls": " ".join(cls), "s": round(number) if col in SCORE_TINTS else None}


def summary_rows(df: pd.DataFrame, columns=SUMMARY_COLUMNS, groups: bool = False) -> list[dict]:
    """One dict per company: raw ticker/name fields plus formatted metric cells (`table_cell`)."""
    text = lambda r, c: r.get(c) if pd.notna(r.get(c)) else DASH  # noqa: E731
    warn_flags = (
        "Cyclical",
        "Peak_Margin",
        "Distress",
        "Data_Suspect",
        "Earnings_Collapse",
        "Revenue_Decline",
        "Accounting_Flag",
    )
    return [{"rank": fmt(r.get("Rank"), ",.0f").replace(NA, DASH), "ticker": r["Ticker"], "slug": slug(r["Ticker"]),
             "name": text(r, "shortName"), "industry": text(r, "industry"), "country": text(r, "country"),
             "pills": list_pills(r),
             "compounder": bool(r.get("Compounder") == True),  # noqa: E712
             "operating": bool(r.get("Financial_Mode") != True and r.get("Business_Model") != "asset_heavy"),  # noqa: E712
             "q70": bool(pd.notna(r.get("Quality_Score")) and float(r.get("Quality_Score")) >= 70.0),
             "clean": not any(r.get(f) == True for f in warn_flags),  # noqa: E712
             "wht0": bool(r.get("Dividend_WHT") == 0),
             "cells": [table_cell(r.get(col), col, spec, scale, suffix, groups) for _, col, spec, scale, suffix in columns]}
            for _, r in df.iterrows()]  # fmt: skip


def ttm_duplicates_latest(row: pd.Series) -> bool:
    return bool(row.get("TTM_Identical_To_Latest") == True)  # noqa: E712 - handles NaN / numpy bool


def show_nav_chart(row: pd.Series) -> bool:
    """True when book value or adjusted tangible book per share reaches the configured share of price."""
    return bool(row.get("Show_NAV_Chart") == True)  # noqa: E712 - handles NaN / numpy bool


def report_periods(df: pd.DataFrame, cfg: Settings) -> list[str]:
    """Period codes shown in every chart and table of one report, oldest to newest.

    Annual slots with no data for ANY company are dropped (so a quiet Yahoo change never leaves
    empty columns), the rest are kept for every company: all charts share the same x-structure.
    """
    codes = []
    for code in cfg.chronological:
        cols = [f"{c}_{code}" for c in ("Total Revenue", "Net Income", "Free Cash Flow") if f"{c}_{code}" in df.columns]
        if cols and df[cols].notna().any().any():
            codes.append(code)
    return codes


def period_range(df: pd.DataFrame, periods: list[str]) -> str:
    """Human summary of the fiscal years covered, from real period-end dates."""
    ends = pd.concat([pd.to_datetime(df[f"PeriodEnd_{c}"], errors="coerce") for c in periods if c != "TTM"
                      and f"PeriodEnd_{c}" in df.columns], ignore_index=True).dropna()  # fmt: skip
    span = f"Fiscal years ending {ends.min():%b-%Y} to {ends.max():%b-%Y}" if not ends.empty else "Fiscal years"
    return span + (" + TTM" if "TTM" in periods else "")


def period_label(row: pd.Series, code: str) -> str:
    """Real period end of this company for a slot, e.g. 'Dec-25' (TTM: 'TTM Mar-26')."""
    end = pd.to_datetime(row.get(f"PeriodEnd_{code}"), errors="coerce")
    if pd.isna(end):
        return "TTM" if code == "TTM" else "n/a"
    return f"TTM {end:%b-%y}" if code == "TTM" else f"{end:%b-%y}"


def chart_years(row: pd.Series, periods: list[str]) -> list[str]:
    """The report's periods for this company: drops periods without data or date, and duplicate TTM."""
    out = []
    for p in periods:
        if p == "TTM" and ttm_duplicates_latest(row):
            continue
        if pd.isna(row.get(f"PeriodEnd_{p}")):
            continue
        has_data = any(
            pd.notna(row.get(f"{col}_{p}")) for col in ("Total Revenue", "Net Income", "Operating Income", "ROIC")
        )
        if has_data:
            out.append(p)
    return out or [p for p in periods if pd.notna(row.get(f"PeriodEnd_{p}"))] or periods[-1:]
