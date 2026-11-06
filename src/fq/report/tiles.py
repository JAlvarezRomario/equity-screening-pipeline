"""A company page facts, header chips, score and context tiles, and the score panel (no markup here)."""

import numpy as np
import pandas as pd

from fq.report.format import DASH, column_help, fmt, show_nav_chart, table_cell
from fq.settings import Settings

# Facts on each company page: (label, column, spec, scale, suffix). `company_facts` groups them: the ranking, the
# Value and Quality criteria of the company's model (read from settings), what is shown but not scored, and the
# candidates under observation.
COMPANY_FACTS = [
    ("Rank", "Rank", ",.0f", 1, ""),
    ("Value score (percentile)", "Value_Score", ".0f", 1, ""),
    ("Quality score (percentile)", "Quality_Score", ".0f", 1, ""),
    ("Value rank", "Value_Rank", ",.0f", 1, ""),
    ("Quality rank", "Quality_Rank", ",.0f", 1, ""),
    # Criteria (shown in the group of the company's model).
    ("Owner yield", "Owner_Yield", ".1f", 1, "%"),
    ("Return quality", "Return_Quality", ".1f", 1, "%"),
    ("Owner trend", "Owner_Trend", "+.1f", 1, "%/yr"),
    ("Owner smoothness", "Owner_Smoothness", ".2f", 1, ""),
    ("ROIC slope", "ROIC_Slope", "+.1f", 1, " pp/yr"),
    ("Quality haircut", "Quality_Haircut", ".2f", 1, "x"),
    ("Archetype source", "Archetype_Source", "", 1, ""),
    ("Earnings power yield", "EPV_Yield", ".1f", 1, "%"),
    ("After-tax yield", "Aftertax_Yield_Avg", ".1f", 1, "%"),
    ("ROIC (norm.)", "ROIC_Norm", ".1f", 1, "%"),
    ("ROIC variability", "ROIC_Variability", ".2f", 1, "x"),
    ("Gross margin std dev", "Gross_Margin_StdDev", ".1f", 1, " pp"),
    ("Gross margin since the inflation peak", "Gross_Margin_Held", "+.1f", 1, " pp"),
    ("Gross profit / assets", "Gross_Profitability_Avg", ".1f", 1, "%"),
    ("Worst revenue year", "Revenue_Worst_Change", "+.1f", 1, "%"),
    ("Years with positive FCF", "FCF_Positive_Share", ".0f", 1, "%"),
    ("Per-share compounding / yr", "Per_Share_Compounding", "+.1f", 1, "%"),
    ("Incremental ROIC", "Incremental_ROIC", ".1f", 1, "%"),
    ("ROIC trend", "ROIC_Trend", "+.1f", 1, " pp"),
    ("Net debt / EBIT", "NetDebt_to_EBIT", ".1f", 1, "x"),
    ("FCF / net income", "FCF_Conversion", ".2f", 1, "x"),
    ("SBC / revenue", "SBC_to_Revenue", ".1f", 1, "%"),
    ("Return on assets", "ROA_Avg", ".2f", 1, "%"),
    ("Equity / assets", "Equity_to_Assets", ".1f", 1, "%"),
    ("Shares reduced / yr", "Share_Reduction_Rate", "+.1f", 1, "%"),
    # Shown, not scored.
    ("FCF yield (norm.)", "FCFy_Norm", ".1f", 1, "%"),
    ("Pre-tax yield", "Pretax_Yield_Avg", ".1f", 1, "%"),
    ("Reported FCF (Y0, M$)", "FCF_Reported_Y0", ",.0f", 1e-6, ""),
    ("Growth capex add-back (Y0, M$)", "Growth_Capex_Owner_Y0", ",.0f", 1e-6, ""),
    ("Growth working capital add-back (Y0, M$)", "Growth_WC_Owner_Y0", ",.0f", 1e-6, ""),
    ("Net interest today / market cap", "Net_Financing_Yield", ".1f", 1, "%"),
    ("Leverage adjustment", "Leverage_Adjustment", "+.1f", 1, " pp"),
    ("Forward P/E", "forwardPE", ".1f", 1, ""),
    ("Price (USD)", "currentPrice", ",.2f", 1, ""),
    ("Market cap (B$)", "Market Cap", ",.2f", 1, ""),
    ("Enterprise value (B$)", "EV", ",.2f", 1, ""),
    ("Dividend yield", "Dividend_Yield", ".1f", 1, "%"),
    ("Interest cover", "Interest_Coverage", ".1f", 1, "x"),
    ("Revenue CAGR", "Revenue_Growth_CAGR", ".1f", 1, "%"),
    ("Revenue / share CAGR", "Revenue_Per_Share_CAGR", ".1f", 1, "%"),
    ("Owner earnings / share compounding / yr", "Owner_Earnings_Compounding", "+.1f", 1, "%"),
    ("Share count change / yr, 10 years", "Shares_Change_10Y", "+.1f", 1, "%"),
    ("Years of dividends without a cut", "Dividend_Years_No_Cut", ".0f", 1, ""),
    ("Dividend / share growth / yr, 10 years", "DPS_Growth_10Y", "+.1f", 1, "%"),
    ("Dividend / share growth / yr, 20 years", "DPS_Growth_20Y", "+.1f", 1, "%"),
    ("Tangible book / price", "NAV_to_Price", ".0f", 1, "%"),
    ("Net current assets / price", "NCAV_to_Price", ".0f", 1, "%"),
    ("Book value / price", "BV_to_Price", ".0f", 1, "%"),
    ("Momentum 6 months", "Momentum_6M", "+.1f", 1, "%"),
    ("Above 52-week low", "Above_52W_Low", ".0f", 1, "%"),
    ("Revenue, recent change", "Revenue_Change_Recent", "+.1f", 1, "%"),
    ("EBIT, recent change", "EBIT_Change_Recent", "+.1f", 1, "%"),
    ("EPS estimates, 90 days", "EPS_Rev_90d", "+.1f", 1, "%"),
    ("Acquisitions / market cap (avg)", "Acquisition_Yield", ".1f", 1, "%"),
    ("Book compounding / yr", "Book_Compounding", "+.1f", 1, "%"),
    ("Altman Z''", "Altman_Z", ".2f", 1, ""),
    ("Institutions", "heldPercentInstitutions", ".1f", 100, "%"),
    ("Analysts", "numberOfAnalystOpinions", ".0f", 1, ""),
    ("Employees", "fullTimeEmployees", ",.0f", 1, ""),
    ("Last quarter", "last_quarter", "", 1, ""),
    # Candidates under observation (`settings.QUALITY_CANDIDATES`): reported, never scored.
    ("Insiders", "heldPercentInsiders", ".1f", 100, "%"),
    ("Asset growth", "Asset_Growth", "+.1f", 1, "%"),
    ("Accruals / assets", "Accruals", "+.1f", 1, "%"),
    ("Beneish M-score", "Beneish_M", ".2f", 1, ""),
    ("ROIC ex goodwill", "ROIC_Ex_Goodwill", ".1f", 1, "%"),
]  # fmt: skip
RANKING_FACTS = ("Rank", "Value_Score", "Quality_Score", "Value_Rank", "Quality_Rank")


# One line under the company name on its HTML page, after country and industry (user's choice: the figures read
# at a glance); `company_facts` leaves them out there so they are not shown twice.
HEADLINE_FACTS = [
    ("Price", "currentPrice", ",.2f", 1, " $"),
    ("Market cap", "Market Cap", ",.1f", 1, " B$"),
    ("EV", "EV", ",.1f", 1, " B$"),
    ("P/E", "trailingPE", ".1f", 1, ""),
    ("Fwd P/E", "forwardPE", ".1f", 1, ""),
    ("Momentum 6M", "Momentum_6M", "+.1f", 1, "%"),
    ("Above 52w low", "Above_52W_Low", ".0f", 1, "%"),
    ("Traded / day", "Traded_Value_USD", ",.1f", 1e-6, " M$"),
    ("Analysts", "numberOfAnalystOpinions", ".0f", 1, ""),
    ("Insiders", "heldPercentInsiders", ".1f", 100, "%"),
    ("Employees", "fullTimeEmployees", ",.0f", 1, ""),
    ("Dividend WHT", "Dividend_WHT", "g", 1, "%"),
]  # fmt: skip

# Context in large type under the score drivers (`score_tiles`; financial mode swaps two). `company_facts` then
# leaves them out.
KEY_TILES = [
    ("Earnings power / EV, before interest", "EPV_Yield", ".1f", 1, "%"),
    ("ROIC (norm.)", "ROIC_Norm", ".1f", 1, "%"),
    ("Net debt / EBIT", "NetDebt_to_EBIT", ".1f", 1, "x"),
    ("Dividend yield", "Dividend_Yield", ".1f", 1, "%"),
]  # fmt: skip
# The recession tile's colour: red past the 20-year lists' bar (`screens.RECESSION_MAX_FALL`, kept equal here: the
# page does not import the screens), green when held within the cut tolerance (`dividend_record.HELD`).
RECESSION_TILE_CUT, RECESSION_TILE_HELD = 25.0, 5.0
FINANCIAL_TILES = {"ROIC_Norm": ("ROE (norm.)", "ROIC_Norm", ".1f", 1, "%"),
                   "NetDebt_to_EBIT": ("Equity / assets", "Equity_to_Assets", ".1f", 1, "%")}  # fmt: skip


def is_financial(row: pd.Series) -> bool:
    return row.get("Financial_Mode") == True  # noqa: E712 - handles NaN / numpy bool


def key_tiles(row: pd.Series) -> list[dict]:
    """`KEY_TILES` of this company's model: label, value, tooltip, CSS class."""
    tiles = [FINANCIAL_TILES.get(t[1], t) if is_financial(row) else t for t in KEY_TILES]
    out = []
    for label, col, spec, scale, suffix in tiles:
        cell = table_cell(row.get(col), col, spec, scale, suffix)
        v = pd.to_numeric(row.get(col), errors="coerce")
        if col == "NetDebt_to_EBIT" and (v < 0 or pd.to_numeric(row.get("Net_Cash_to_Cap"), errors="coerce") > 0):
            cell = {"text": "Net cash", "cls": "pos"}  # not "0.0x": more cash than debt
        elif col == "Dividend_Yield" and v == 0:
            cell = {"text": "None", "cls": "muted"}  # pays no dividend
        out.append({"label": label, "value": cell["text"], "help": column_help(col), "col": col,
                    "cls": cell["cls"].replace("num", "").strip()})  # fmt: skip
    note = row.get("Recession_Dividend_Note")
    if isinstance(note, str) and note:  # the dividend through past recessions (`dividend_record`, rule 7)
        fall = pd.to_numeric(row.get("Recession_Dividend_Fall"), errors="coerce")
        cls = "neg" if fall < -RECESSION_TILE_CUT else ("pos" if fall >= -RECESSION_TILE_HELD else "")
        out.append({"label": "Dividend through recessions", "value": note, "col": "Recession_Dividend_Note",
                    "help": column_help("Recession_Dividend_Note"), "cls": cls})  # fmt: skip
    return out


# One plain line under each scored criterion's tile: what it measures, for a reader new to the terms.
DRIVER_NOTES = {
    "Owner_Yield": "what one share earns its owner in a normal year, over the price",
    "Return_Quality": "return on capital (ROE for financials), adjusted for its trend, steadiness and debt",
    "Per_Share_Compounding": "how fast one share of the business grew each year",
    "Share_Reduction_Rate": "shares bought back (+) or issued (-) each year",
}


def score_tiles(row: pd.Series) -> list[dict]:
    """The scored criteria of the company's model, Value's first, as tiles: the value, its name (the index column's),
    a plain line on what it measures and how it ranks ("better than 96% · 20% of Quality"; `tile-pct`, which the
    grading leaves out). Replaces the criteria card: the page says once what the scores are built from."""
    s = Settings()
    family = "financial" if is_financial(row) else "operating"
    specs = {col: (label, spec, scale, suffix) for label, col, spec, scale, suffix in COMPANY_FACTS}
    out = []
    for dim in ("Value", "Quality"):
        for col, _, weight in s.criteria(dim, family):
            if col not in specs:
                continue
            label, spec, scale, suffix = specs[col]
            cell = table_cell(row.get(col), col, spec, scale, suffix)
            pct = pd.to_numeric(row.get(f"Pct_{col}"), errors="coerce")
            share = f"all of {dim}" if weight >= 1 else f"{weight:.0%} of {dim}"
            out.append({"label": label, "value": cell["text"], "help": column_help(col), "col": col,
                        "cls": cell["cls"].replace("num", "").strip(), "note": DRIVER_NOTES.get(col),
                        "pct": (f"better than {int(round(pct))}% · " if pd.notna(pct) else "") + share,
                        "pct_cls": "pos" if pct >= 75 else "neg" if pct <= 25 else "", "dim": dim[0].lower()})  # fmt: skip
    return out


# How each model is measured, under the company's name (operating companies need no line).
MODEL_LINES = {
    "bank": "Bank: measured on net income, ROE and book value",
    "insurer": "Insurer: measured on net income, ROE and book value",
    "holding": "Holding company: measured on net income, ROE and book value",
    "asset_heavy": "Asset-heavy: upkeep capex = depreciation",
    "client_cash": "Holds client cash, not counted as spare",
    "float": "Float business: measured on net income",
    "operating": "Operating company: measured on owner FCF and ROIC",
}


def model_line(row: pd.Series) -> str | None:
    model = row.get("Business_Model")
    text = MODEL_LINES.get(model, f"{model.capitalize()}" if isinstance(model, str) else None)
    src = row.get("Archetype_Source")
    if text and isinstance(src, str) and src:
        return f"{text} (source: {src})"
    return text


def net_cash_pill(row: pd.Series) -> dict | None:
    """Highlighted measure of Net cash / Net debt as % of market cap for company page."""
    if is_financial(row):
        eq_assets = pd.to_numeric(row.get("Equity_to_Assets"), errors="coerce")
        if pd.isna(eq_assets):
            return None
        val = float(eq_assets)
        return {
            "label": "Equity / assets",
            "value": f"{val:.1f}%",
            "cls": "neutral",
            "title": f"Equity / assets: tangible equity is {val:.1f}% of total assets (Bank/Insurer model)",
            "sub": None,
        }
    val = pd.to_numeric(row.get("Net_Cash_to_Cap"), errors="coerce")
    if pd.isna(val):
        return None
    val = float(val)
    if val >= 0:
        return {
            "label": "Net cash",
            "value": f"+{val:.1f}%",
            "cls": "pos",
            "title": f"Net cash: spare cash exceeds total debt and preferreds by {val:.1f}% of market capitalization",
            "sub": "of cap",
        }
    else:
        return {
            "label": "Net debt",
            "value": f"{val:.1f}%",
            "cls": "neg",
            "title": f"Net debt: total debt and preferreds exceed spare cash by {abs(val):.1f}% of market capitalization",
            "sub": "of cap",
        }


UNRELIABLE_FLAGS = ("Data_Suspect", "Insufficient_Data", "Accounting_Flag")  # with an open `Model_Review`


def unreliable(df: pd.DataFrame) -> pd.Series:
    """Rows whose figures may be wrong (`unreliable_reasons` is not empty): labelled on their page, left off the map."""
    out = pd.Series(False, index=df.index)
    for flag in UNRELIABLE_FLAGS:
        if flag in df.columns:
            out |= df[flag] == True  # noqa: E712
    if "Model_Review" in df.columns:
        out |= df["Model_Review"].map(lambda v: isinstance(v, str) and bool(v.strip()))
    return out


def unreliable_reasons(row: pd.Series) -> list[str]:
    """Why the company's own figures may be wrong (empty when they are trusted), for the company page's "Data not
    reliable" banner (user's choice, Nov 3, 2026: a page with Value 99 and "1 of 9,486" read as a find when the
    numbers behind it were suspect). Only doubts about the numbers or the yardstick count: a real decline, distress
    or a peak margin is a fact about the business, shown by its tag, not a reason to doubt the page."""

    def flag(name: str) -> bool:
        return row.get(name) == True  # noqa: E712

    def text(name: str) -> str | None:
        v = row.get(name)
        return v if isinstance(v, str) and v.strip() else None

    out = []
    if flag("Data_Suspect"):
        out.append(f"Figures that look wrong: {text('Suspect_Reasons') or 'flagged as suspect'}.")
    if flag("Insufficient_Data"):
        out.append("Not enough data to score both Value and Quality.")
    if flag("Accounting_Flag"):
        out.append(f"Earnings ahead of cash: {text('Accounting_Reasons') or 'accruals'}.")
    if text("Model_Review"):
        out.append(
            f"Business model to confirm, so it may be measured with the wrong yardstick: {text('Model_Review')}."
        )
    return out


def score_panel(row: pd.Series, counts: dict[str, int]) -> dict:
    """The two scores (0-100 or None), the ranks out of their counts and one exact sentence on where the company
    stands against the median company. With unreliable figures (`unreliable_reasons`) the scores stay as measured
    but no rank is quoted: the company is ranked after every trusted one (`ranking.py`)."""

    def score(col: str) -> int | None:
        v = pd.to_numeric(row.get(col), errors="coerce")
        return int(round(v)) if pd.notna(v) else None

    def rank(col: str, among: str) -> str | None:
        v = pd.to_numeric(row.get(col), errors="coerce")
        return f"{int(v):,} of {counts.get(col, 0):,} {among}" if pd.notna(v) else None

    value, quality = score("Value_Score"), score("Quality_Score")
    if unreliable_reasons(row):
        return {"value": value, "quality": quality, "rank": "After every trusted company", "value_rank": None,
                "quality_rank": None, "sentence": "Not ranked with trusted companies: its figures need checking.",
                "unreliable": True}  # fmt: skip
    return {"value": value, "quality": quality, "rank": rank("Rank", "ranked"), "value_rank": rank("Value_Rank", "scored"),
            "quality_rank": rank("Quality_Rank", "scored"), "sentence": score_position(value, quality)}  # fmt: skip


def score_position(value: int | None, quality: int | None) -> str:
    """Where two 0-100 scores sit against the median company (50)."""
    if value is None or quality is None:
        return "Not enough data to score both Value and Quality."
    side = lambda s: "above" if s > 50 else "below" if s < 50 else "at"  # noqa: E731
    v, q = side(value), side(quality)
    if v == q:
        return f"{v.capitalize()} the median company on both Value and Quality."
    return f"{v.capitalize()} the median company on Value, {q} it on Quality."


NEXT_REPORT_SOON_DAYS = 30  # the "Next report" chip says "in N days" and stands out within this


def headline_facts(row: pd.Series, now: pd.Timestamp | None = None) -> list[tuple[str, str, str]]:
    """`HEADLINE_FACTS` as (label, value, CSS class) chips, a missing figure left out (not "N/A"), a change coloured
    as in the tables (`GOOD_WHEN_UP`: the 6-month momentum), then the next report's date when Yahoo has one still
    ahead (`earningsTimestamp`), with the days to go when it is near."""
    chips = []
    for label, col, spec, scale, suffix in HEADLINE_FACTS:
        if pd.notna(pd.to_numeric(row.get(col), errors="coerce")):
            cell = table_cell(row.get(col), col, spec, scale, suffix)
            chips.append((label, fmt(row.get(col), spec, scale, suffix), cell["cls"].replace("num", "").strip()))
    when = pd.to_datetime(pd.to_numeric(row.get("earningsTimestamp"), errors="coerce"), unit="s", errors="coerce")
    now = now or pd.Timestamp.now()
    if pd.notna(when) and when > now:
        days = (when.normalize() - now.normalize()).days
        if days <= NEXT_REPORT_SOON_DAYS:
            soon = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
            chips.append(("Next report", f"{when:%d %b %Y} · {soon}", "soon"))
        else:
            chips.append(("Next report", f"{when:%d %b %Y}", ""))
    return chips


# Measures of one business model only: not listed as N/A for every other company.
OPTIONAL_FACTS = frozenset({"ROA_Avg", "FCF_Reported_Y0", "Growth_Capex_Owner_Y0", "Growth_WC_Owner_Y0",
                            "Book_Compounding", "Equity_to_Assets", "Net_Financing_Yield",
                            "Leverage_Adjustment"})  # fmt: skip


def rank_counts(df: pd.DataFrame) -> dict[str, int]:
    """How many companies hold each rank, so a page can say "12 of 480"."""
    return {c: int(df[c].notna().sum()) for c in RANKING_FACTS if c.endswith("Rank") and c in df.columns}


def company_facts(
    row: pd.Series, also_skip: frozenset[str] = frozenset(), counts: dict[str, int] | None = None
) -> list[tuple[str, list[dict]]]:
    """Key figures in groups: the ranking (ranks out of `counts`), the Value and Quality criteria of the company's
    model in the settings' order, what is shown but not scored, and the candidates under observation. The
    book-value ratios are left out when the NAV chart already shows them, `also_skip` columns when the page shows
    them already (the headline line). Each item: label, value, tooltip and CSS class."""
    s = Settings()
    family = "financial" if row.get("Financial_Mode") == True else "operating"  # noqa: E712 - NaN / numpy bool
    value = [c for c, *_ in s.criteria("Value", family)]
    quality = [c for c, *_ in s.criteria("Quality", family)]
    weights_map = {c: w for c, _, w in [*s.criteria("Value", family), *s.criteria("Quality", family)]}
    candidates = [c for c, _ in s.quality_candidates]
    others = {c for crit in (s.value_criteria, s.quality_criteria, s.financial_value_criteria,
                             s.financial_quality_criteria) for c, *_ in crit}  # fmt: skip
    specs = {col: (label, spec, scale, suffix) for label, col, spec, scale, suffix in COMPANY_FACTS}
    shown = [c for c in specs if c not in {*RANKING_FACTS, *value, *quality, *candidates}]
    skip = ({"NAV_to_Price", "BV_to_Price"} if show_nav_chart(row) else set()) | set(also_skip)
    counts = counts or {}

    def item(col: str) -> dict | None:
        if col not in specs or col in skip:
            return None
        v = row.get(col)
        if pd.isna(v) and (col in OPTIONAL_FACTS or (col in others and col not in {*value, *quality})):
            return None
        if col in ("Growth_Capex_Owner_Y0", "Growth_WC_Owner_Y0") and (pd.isna(v) or float(v) == 0.0):
            return None
        label, spec, scale, suffix = specs[col]
        if col == "ROIC_Norm" and family == "financial":
            label = "ROE (norm.)"
        cell = table_cell(v, col, spec, scale, suffix) if spec else {"text": str(v) if pd.notna(v) else DASH, "cls": ""}
        text = cell["text"]
        if col in counts and cell["cls"] != "num na":
            text = f"{text} of {counts[col]:,}"
        pct_val = pd.to_numeric(row.get(f"Pct_{col}"), errors="coerce") if col in weights_map else np.nan
        pct = f"p{int(round(pct_val))} · w{weights_map[col]:g}" if pd.notna(pct_val) else None
        pct_cls = "pos" if pd.notna(pct_val) and pct_val >= 75 else "neg" if pd.notna(pct_val) and pct_val <= 25 else ""
        return {"label": label, "value": text, "help": column_help(col), "cls": cell["cls"].replace("num", "").strip(),
                "pct": pct, "pct_cls": pct_cls}  # fmt: skip

    groups = [("Ranking", RANKING_FACTS), ("Value criteria", value), ("Quality criteria", quality),
              ("Shown, not scored", shown), ("Under observation, not scored", candidates)]  # fmt: skip
    out = [(name, [i for i in map(item, cols) if i]) for name, cols in groups]
    return [(name, items) for name, items in out if items]
