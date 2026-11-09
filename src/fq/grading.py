"""Moat and management grades (1 to 5) by Gemini: the gate on the highlight lists.

The only module that calls an LLM (user's choice, Oct 31, 2026). A grade never enters a score, a rank or a flag; it
only decides where a company sits inside a highlight list it already made (`passes`). The procedure: a first pass; a
second, independent one only when the first clears the gate, the two then averaged (user's rule, Nov 6, 2026: a
company failing on the first pass could make no list); then a synthesis that justifies the grade. The company's
statements and description are passed in (`grade_input`; never its price, user's point, Nov 7, 2026) as context, not
the verdict: the prompt also asks what the figures cannot show (revenue dependency on one customer, platform, product or licence, with a cap on the moat; user's
rule, Nov 6, 2026). No search tool: the model's own knowledge and our figures. Grades are kept in `data/grades.parquet` (`load`, `save`) and redone only when due
(`due`: a new annual report or a year old). The google-genai client is imported only when a grading runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"
GRADE_MODEL = "gemini-4-argon"  # the two grading passes
SYNTHESIS_MODEL = "gemini-3.8-flash"  # the justification of the averaged grade
THINKING = "medium"
GRADE_OUTPUT_TOKENS = 32768
SYNTHESIS_OUTPUT_TOKENS = 16384
PASSES = 2
SECOND_PASS_ONLY_IF_GATE = True  # the second pass runs only when the first clears `passes` (user, Nov 6, 2026)
NO_SECOND_NOTE = "(none: one note only)"  # Note B of the synthesis when only the first pass ran
WORKERS = 6  # companies graded at once
CALL_TIMEOUT_MS = 600_000  # one model call (about a minute normally); past it the call raises and `_call` retries
MAX_AGE_DAYS = 365  # re-grade after a year, or when a new annual report lands (`due`)
MAX_PER_RUN = 60  # companies graded by one `fq run -u benchmark` (~$6 now); `fq grade` has no cap

# The gate (user's rule, Oct 31, 2026): both grades at least MIN_BOTH. Until Nov 9, 2026 either grade at least 4.25 also
# passed; dropped (user's choice): an outstanding moat does not make up for a management that treats minority holders
# badly (no shown company passed by that route that day, 4 of 818 stored grades could).
MIN_BOTH = 3.0
# The Shortlist's stricter gate (user's rule, Oct 31, 2026): both at least MIN_BOTH and either at least SHORTLIST_MIN_TOP.
SHORTLIST_MIN_TOP = 4.0
# Best businesses (user's choice, Nov 7, 2026; the bar of the retired Best grades tab): both at least BEST_MIN_BOTH and
# either above it (grades step by 0.25).
BEST_MIN_BOTH = 4.0
# Cheapest (user's choice, Nov 7, 2026): both grades at least CHEAPEST_MIN_BOTH, walking down the Value ranking.
CHEAPEST_MIN_BOTH = 3.0
# Temporary trouble (user's choice, Nov 7, 2026): both at least TROUBLE_MIN_BOTH and either at least TROUBLE_MIN_TOP.
TROUBLE_MIN_BOTH = 3.0
TROUBLE_MIN_TOP = 3.5

MOAT_LABEL = "Economic Moat Score"
MGMT_LABEL = "Management Quality Score"

# (start date, input, output, cached input) in USD per 1M tokens; thinking is billed as output.
# Argon's introductory rates end Nov 30, 2026, Flash's Dec 31, 2026.
RATES = {
    GRADE_MODEL: ((pd.Timestamp("2000-01-01"), 2.00, 10.00, 0.10), (pd.Timestamp("2026-12-01"), 4.00, 20.00, 0.20)),
    SYNTHESIS_MODEL: ((pd.Timestamp("2000-01-01"), 0.75, 3.75, 0.075), (pd.Timestamp("2027-01-01"), 1.50, 7.50, 0.15)),
}
EST_COST_USD = 0.10  # one company, both passes, at the introductory rates, measured on MA and AMR (Oct 30, 2026)

COLUMNS = ["Ticker", "Moat", "Management", "Moat_1", "Mgmt_1", "Moat_2", "Mgmt_2", "Synthesis", "Report_1",
           "Report_2", "Graded_At", "Y0_End", "Model", "Prompt_Hash", "Synthesis_Hash", "Tokens_In", "Tokens_Out",
           "Cost_USD"]  # fmt: skip
# Older grading-pass hashes whose passes count as current (only their synthesis is redone, `resynthesize`). Emptied
# Nov 6, 2026: "1ab400c79f69" (before the synthesis had a hash of its own, Oct 31) was made with the grading prompt
# replaced that day, and no stored grade carried it any more. "ca61a6316a06" (Nov 7, 2026, user's choice): the
# price-free input before the management line on the whole record; its grades stay until due for another reason
# (a new annual report, `MAX_AGE_DAYS`), so the line costs no re-grade.
PASS_HASH_ALIASES: frozenset[str] = frozenset({"ca61a6316a06"})


def passes(moat, mgmt) -> bool | None:
    """The gate: True when both grades reach `MIN_BOTH`; None when ungraded."""
    if pd.isna(moat) or pd.isna(mgmt):
        return None
    return bool(moat >= MIN_BOTH and mgmt >= MIN_BOTH)


def passes_best_walk(moat, mgmt) -> bool | None:
    """The Best businesses tab's bar: both grades reach `BEST_MIN_BOTH` and either is above it; None when ungraded
    (waiting for its grade, `highlights._graded_walk`)."""
    if pd.isna(moat) or pd.isna(mgmt):
        return None
    return bool(moat >= BEST_MIN_BOTH and mgmt >= BEST_MIN_BOTH and max(moat, mgmt) > BEST_MIN_BOTH)


def passes_cheapest(moat, mgmt) -> bool | None:
    """The Cheapest tab's bar: both grades reach `CHEAPEST_MIN_BOTH`; None when ungraded (waiting for its grade)."""
    if pd.isna(moat) or pd.isna(mgmt):
        return None
    return bool(moat >= CHEAPEST_MIN_BOTH and mgmt >= CHEAPEST_MIN_BOTH)


def passes_trouble(moat, mgmt) -> bool:
    """The Temporary trouble tab's bar: both grades reach `TROUBLE_MIN_BOTH` and either `TROUBLE_MIN_TOP`; False when
    ungraded (the list must be graded: an ungraded member is shaded until `fq grade` reaches it)."""
    if pd.isna(moat) or pd.isna(mgmt):
        return False
    return bool(moat >= TROUBLE_MIN_BOTH and mgmt >= TROUBLE_MIN_BOTH and max(moat, mgmt) >= TROUBLE_MIN_TOP)


def passes_shortlist(moat, mgmt) -> bool | None:
    """The Shortlist's gate: both grades reach `MIN_BOTH` and either reaches `SHORTLIST_MIN_TOP`; None when ungraded."""
    if pd.isna(moat) or pd.isna(mgmt):
        return None
    return bool(moat >= MIN_BOTH and mgmt >= MIN_BOTH and max(moat, mgmt) >= SHORTLIST_MIN_TOP)


def score_after(label: str, text: str) -> float | None:
    """The 0-5 score written after `label`, whatever markdown surrounds it; the last one (the verdict comes last)."""
    flat = re.sub(r"[*_`#]+", "", text or "")
    found = list(re.finditer(label + r"\s*:?\s*([0-5](?:\.\d+)?)\s*(?:/\s*5)?", flat))
    return float(found[-1].group(1)) if found else None


# --- The grading input: the business, not the stock ---------------------------------------------------------------
# What `grade_input` carries; bump it when that changes, so every grade is redone (`prompt_hash`, `due`).
# 2 (Oct 31, 2026): this system's conclusions taken out of the page. 3 (Nov 7, 2026, user's point): the statements
# and the description from the ranked row instead of the rendered page, with no price at all: the page carried the
# price, P/E, 6-month momentum, the 52-week low and some fifteen yields over today's price, and over 516 graded ranked
# companies, Quality held fixed, management grades rose with momentum (Spearman +0.24) and both grades fell with Value
# (-0.34, -0.36). A grade is a gate on lists that already test price, so it must judge the business alone.
INPUT_VERSION = 3

# Money lines, per period, in millions (label, column concept); cash out is negative, as reported.
STATEMENT_LINES = (
    ("Revenue", "Total Revenue"),
    ("Gross profit", "Gross Profit"),
    ("Operating income (EBIT)", "EBIT"),
    ("Net income, one-offs out", "Net_Income_Clean"),
    ("Operating cash flow", "Operating Cash Flow"),
    ("Capital expenditure", "Capital Expenditure"),
    ("Stock-based compensation", "Stock Based Compensation"),
    ("Free cash flow (all capex, stock pay, leases and interest charged)", "FCF_Reported"),
    (
        "Owner earnings (free cash flow after interest, with the capex and working capital that bought growth added "
        "back; lenders and insurers: clean net income less the capital kept against inflation)",
        "Owner_Earnings",
    ),
    ("Acquisitions, net", "Net Business Purchase And Sale"),
    ("Dividends paid", "Cash Dividends Paid"),
    ("Shares bought back", "Repurchase Of Capital Stock"),
    ("Shares issued", "Issuance Of Capital Stock"),
    ("Cash and short-term investments", "Cash Cash Equivalents And Short Term Investments"),
    ("Total debt", "Total Debt"),
    ("Common equity", "Common Stock Equity"),
    ("Goodwill and intangibles", "Goodwill And Other Intangible Assets"),
    ("Total assets", "Total Assets"),
)
# Per-period ratios and counts (label, column concept, format), as stored (percentages already x 100).
RATIO_LINES = (
    ("Shares outstanding (millions)", "Ordinary Shares Number", "shares"),
    ("Return on invested capital, % (on equity for financials; in US terms)", "ROIC", ".1f"),
    ("Return on capital without goodwill and intangibles, %", "ROIC_Ex_Goodwill", ".1f"),
    ("Operating margin, %", "operatingMargins", ".1f"),
)
# One figure each (label, column, format, unit): computed from the statements, never from a price.
FACTS = (
    ("Normalised return on invested capital (on equity for financials)", "ROIC_Norm", ".1f", "%"),
    ("Return earned on the capital added since the oldest report", "Incremental_ROIC", ".0f", "%"),
    ("Gross profit / operating assets", "Gross_Profitability_Avg", ".1f", "%"),
    ("Gross margin now less in the year of the highest inflation", "Gross_Margin_Held", "+.1f", " pp"),
    ("Free cash flow / net income, median", "FCF_Conversion", ".2f", "x"),
    ("Years with positive free cash flow", "FCF_Positive_Share", ".0f", "%"),
    ("Stock-based pay / revenue, median", "SBC_to_Revenue", ".1f", "%"),
    ("Accruals (net income less operating cash flow) / assets", "Accruals", ".1f", "%"),
    ("Beneish M-score (above -1.78 is the usual manipulation warning)", "Beneish_M", ".2f", ""),
    ("Net debt / EBIT (0 = net cash)", "NetDebt_to_EBIT", ".1f", "x"),
    ("Interest cover", "Interest_Coverage", ".1f", "x"),
    ("Equity / assets", "Equity_to_Assets", ".1f", "%"),
    ("Altman Z''", "Altman_Z", ".2f", ""),
    ("Revenue per share, real growth trend a year (x its R²)", "Per_Share_Compounding", "+.1f", "%"),
    ("Share count change a year, up to 10 years (negative = bought back)", "Shares_Change_10Y", "+.1f", "%"),
    ("Years of dividends without a cut", "Dividend_Years_No_Cut", ".0f", ""),
    ("Revenue, latest twelve months against the year before", "Revenue_Change_Recent", "+.1f", "%"),
    ("EBIT, latest twelve months against the year before", "EBIT_Change_Recent", "+.1f", "%"),
)
LENDER_LINES = (("Net interest income", "Net Interest Income"),)  # financial mode only: elsewhere it is interest paid
LONG_HISTORY_YEARS = 25  # calendar years of the dividend and share record shown


def _num(v) -> float:
    out = pd.to_numeric(v, errors="coerce")
    return float(out) if pd.notna(out) else float("nan")


def _present(v) -> bool:
    return pd.notna(_num(v))


def _period_label(row: pd.Series, period: str) -> str:
    end = pd.to_datetime(row.get(f"PeriodEnd_{period}"), errors="coerce")
    label = f"{end:%b-%y}" if pd.notna(end) else period
    return f"TTM {label}" if period == "TTM" else label


def _table(title: str, columns: list[str], rows: list[tuple[str, list[str]]]) -> str:
    lines = [f"### {title}", "| | " + " | ".join(columns) + " |", "|" + "---|" * (len(columns) + 1)]
    lines += ["| " + name + " | " + " | ".join(cells) + " |" for name, cells in rows]
    return "\n".join(lines)


def _cells(values: list[float], spec: str) -> list[str]:
    return [format(v, spec) if pd.notna(v) else "–" for v in values]


def _long_history(row: pd.Series) -> str | None:
    """Dividends per share (quote currency) and the share count by calendar year, the last `LONG_HISTORY_YEARS`."""
    series = {}
    specs = (("Dividend per share", "DPS_Hist", 1.0, ".4g"), ("Shares (millions)", "Shares_Hist", 1e-6, ",.1f"))
    for name, col, scale, spec in specs:
        raw = row.get(col)
        try:
            values = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            values = {}
        if values:
            series[name] = ({str(k): _num(v) * scale for k, v in values.items()}, spec)
    if not series:
        return None
    years = sorted({y for v, _ in series.values() for y in v})[-LONG_HISTORY_YEARS:]
    rows = [(name, _cells([v.get(y, float("nan")) for y in years], spec)) for name, (v, spec) in series.items()]
    return _table("The long record, by calendar year (dividend per share in the quote currency)", years, rows)


def grade_input(row: pd.Series, cfg=None) -> str:
    """What the grader reads about one company (`INPUT_VERSION` 3, user's point, Nov 7, 2026): who it is (the business
    description, country, industry, employees, insiders, the measuring model), its statements per period in millions,
    ratios computed from them, and the long dividend and share record. Never a price or anything divided by one
    (price, market cap, P/E, yields, momentum, the 52-week low, traded value), nor the market's mood (analysts,
    estimate revisions), withholding, or anything this system scores or concludes: the grade judges the business and
    its managers, not the stock. Built from the ranked row, not from the site, so the site's layout never moves a
    grade."""
    from fq.settings import Settings

    cfg = cfg or Settings()
    get = row.get
    name = get("shortName") if isinstance(get("shortName"), str) else get("Ticker")
    out = [f"## {name} ({get('Ticker')})"]
    who = " · ".join(v for v in (get("country"), get("sector"), get("industry")) if isinstance(v, str))
    if who:
        out.append(who)
    model = get("Business_Model")
    financial = get("Financial_Mode") == True  # noqa: E712 - numpy bool
    if isinstance(model, str):
        out.append(f"Measured as: {model.replace('_', ' ')} business"
                   + (" (net income, return on equity and book value)" if financial else ""))  # fmt: skip
    if isinstance(get("longBusinessSummary"), str):
        out.append(get("longBusinessSummary"))
    if _present(get("fullTimeEmployees")):
        out.append(f"- Employees: {_num(get('fullTimeEmployees')):,.0f}")
    if _present(get("heldPercentInsiders")):
        out.append(f"- Held by insiders: {_num(get('heldPercentInsiders')) * 100:.1f}%")

    periods = list(cfg.chronological)
    if get("TTM_Identical_To_Latest") == True:  # noqa: E712 - TTM repeats the latest annual report
        periods.remove("TTM")
    # A period with neither revenue nor clean net income is only a dividend from an older year: left out.
    periods = [p for p in periods if _present(get(f"Total Revenue_{p}")) or _present(get(f"Net_Income_Clean_{p}"))]
    if periods:
        heads = [_period_label(row, p) for p in periods]
        money = []
        lines = (*STATEMENT_LINES[:3], *(LENDER_LINES if financial else ()), *STATEMENT_LINES[3:])
        for label, concept in lines:
            values = [_num(get(f"{concept}_{p}")) / 1e6 for p in periods]
            if any(pd.notna(v) for v in values):
                money.append((label, _cells(values, ",.0f")))
        out += ["", _table("The statements, millions of US dollars at today's exchange rate (one constant rate, so "
                           "the trend is the company's own currency's); cash out is negative", heads, money)]  # fmt: skip
        ratios = []
        for label, concept, spec in RATIO_LINES:
            values = [_num(get(f"{concept}_{p}")) for p in periods]
            if spec == "shares":
                values, spec = [v / 1e6 for v in values], ",.1f"
            if any(pd.notna(v) for v in values):
                ratios.append((label, _cells(values, spec)))
        if ratios:
            out += ["", _table("Returns, margins and shares", heads, ratios)]
    figures = [f"- {label}: {format(_num(get(col)), spec)}{unit}" for label, col, spec, unit in FACTS
               if _present(get(col))]  # fmt: skip
    if figures:
        out += ["", "### Figures computed from the statements", *figures]
    history = _long_history(row)
    if history:
        out += ["", history]
    return "\n".join(out).strip()


# --- Prompts and calls --------------------------------------------------------------------------------------------


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def prompt_hash() -> str:
    """What the grades were produced with: the grading prompt, model, passes and page (stored; another hash makes a
    grade due). The synthesis has its own (`synthesis_hash`), so rewording it never redoes the costly passes."""
    digest = hashlib.sha1(_prompt("grade_prompt.md").encode())
    digest.update(repr((GRADE_MODEL, THINKING, PASSES, SECOND_PASS_ONLY_IF_GATE, INPUT_VERSION)).encode())
    return digest.hexdigest()[:12]


def synthesis_hash() -> str:
    """What the published text was written with: the synthesis prompt and model (`resynthesize` when it changes)."""
    digest = hashlib.sha1(_prompt("grade_synthesis.md").encode())
    digest.update(SYNTHESIS_MODEL.encode())
    return digest.hexdigest()[:12]


def _synthesis_prompt(company: str, ticker: str, moat: float, mgmt: float, report_1: str, report_2: str) -> str:
    """The synthesis prompt; Note B reads `NO_SECOND_NOTE` when only the first pass ran (an empty `report_2`)."""
    report_2 = report_2 if isinstance(report_2, str) and report_2.strip() else NO_SECOND_NOTE
    return (_prompt("grade_synthesis.md").replace("{{COMPANY_NAME}}", company).replace("{{TICKER}}", ticker)
            .replace("{{MOAT_GRADE}}", f"{moat:.2f}").replace("{{MGMT_GRADE}}", f"{mgmt:.2f}")
            .replace("{{FIRST_REPORT}}", report_1).replace("{{BIS_REPORT}}", report_2))  # fmt: skip


def grade_prompt(company: str, ticker: str, page: str, today: pd.Timestamp) -> str:
    return _prompt("grade_prompt.md").format(company_name=f"{company} ({ticker})",
                                             execution_date=f"{today:%Y-%m-%d}", company_figures=page)  # fmt: skip


def cost_usd(model: str, tokens_in: int, cached: int, tokens_out: int, when: pd.Timestamp) -> float:
    """A call's cost at the rates in force on `when` (thinking counted in `tokens_out`)."""
    rate = [r for r in RATES[model] if when >= r[0]][-1]
    return ((tokens_in - cached) * rate[1] + cached * rate[3] + tokens_out * rate[2]) / 1e6


def estimate_usd(n: int, when: pd.Timestamp) -> float:
    """What grading `n` companies costs at most, from the measured `EST_COST_USD` (both passes) scaled to the rates on
    `when`; a company failing the gate on its first pass costs about half."""
    now = [r for r in RATES[GRADE_MODEL] if when >= r[0]][-1][1] / RATES[GRADE_MODEL][0][1]
    return n * EST_COST_USD * now


def _call(client, model: str, prompt: str, max_out: int, label: str) -> tuple[str, dict]:
    """One generate_content call, no tools, retried twice on errors; returns (text, usage)."""
    from google.genai import types

    config = types.GenerateContentConfig(max_output_tokens=max_out,
                                         thinking_config=types.ThinkingConfig(thinking_level=THINKING))  # fmt: skip
    for attempt in range(3):
        try:
            start = time.monotonic()
            response = client.models.generate_content(model=model, contents=[prompt], config=config)
            parts = (response.candidates[0].content.parts or []) if response.candidates else []
            text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
            meta = response.usage_metadata
            out = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
            usage = {"in": meta.prompt_token_count or 0, "cached": meta.cached_content_token_count or 0, "out": out}
            usage["usd"] = cost_usd(model, usage["in"], usage["cached"], out, pd.Timestamp.now())
            log.debug(
                "%s: %.0fs, %d in, %d out, $%.4f", label, time.monotonic() - start, usage["in"], out, usage["usd"]
            )
            if text:
                return text, usage
            log.warning("%s: empty answer (attempt %d)", label, attempt + 1)
        except Exception as exc:  # network, quota, server errors: wait and retry
            log.warning("%s failed (attempt %d): %s", label, attempt + 1, exc)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"{label}: no answer after 3 attempts")


def grade_company(client, ticker: str, company: str, page: str, today: pd.Timestamp) -> dict:
    """A first grading pass; when it clears the gate (`passes`), a second independent one and the two averaged; when it
    does not (`SECOND_PASS_ONLY_IF_GATE`), the first alone, with pass 2 left empty. Then the synthesis that justifies
    the grade. A pass whose report lacks either score is asked again once; without both scores the company stays
    ungraded (an error)."""
    prompt = grade_prompt(company, ticker, page, today)
    reports, grades, usage = [], [], []
    for n in range(1, PASSES + 1):
        if n > 1 and SECOND_PASS_ONLY_IF_GATE and not passes(*grades[0]):
            break
        for _ in range(2):
            text, use = _call(client, GRADE_MODEL, prompt, GRADE_OUTPUT_TOKENS, f"{ticker} pass {n}")
            usage.append(use)
            moat, mgmt = score_after(MOAT_LABEL, text), score_after(MGMT_LABEL, text)
            if moat is not None and mgmt is not None:
                break
        else:
            raise RuntimeError(f"{ticker} pass {n}: no moat and management scores in the report")
        reports.append(text)
        grades.append((moat, mgmt))
    moat = sum(g[0] for g in grades) / len(grades)
    mgmt = sum(g[1] for g in grades) / len(grades)
    second = grades[1] if len(grades) > 1 else (float("nan"), float("nan"))
    report_2 = reports[1] if len(reports) > 1 else ""
    synthesis = _synthesis_prompt(company, ticker, moat, mgmt, reports[0], report_2)
    text, use = _call(client, SYNTHESIS_MODEL, synthesis, SYNTHESIS_OUTPUT_TOKENS, f"{ticker} synthesis")
    usage.append(use)
    return {"Ticker": ticker, "Moat": moat, "Management": mgmt, "Moat_1": grades[0][0], "Mgmt_1": grades[0][1],
            "Moat_2": second[0], "Mgmt_2": second[1], "Synthesis": text, "Report_1": reports[0],
            "Report_2": report_2, "Graded_At": today, "Model": GRADE_MODEL, "Prompt_Hash": prompt_hash(),
            "Synthesis_Hash": synthesis_hash(),
            "Tokens_In": sum(u["in"] for u in usage), "Tokens_Out": sum(u["out"] for u in usage),
            "Cost_USD": round(sum(u["usd"] for u in usage), 4)}  # fmt: skip


# --- The cache ----------------------------------------------------------------------------------------------------


def load(path: Path) -> pd.DataFrame:
    """The stored grades (one row per ticker), empty when none."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(path)


def save(grades: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grades.drop_duplicates("Ticker", keep="last").to_parquet(path, index=False)


def due(candidates, ranked: pd.DataFrame, grades: pd.DataFrame, now: pd.Timestamp) -> list[str]:
    """The candidates to grade, in their order: never graded, graded with other prompts or another page
    (`Prompt_Hash`), on an older annual report (`PeriodEnd_Y0` moved), or `MAX_AGE_DAYS` ago or more."""
    current = prompt_hash()
    stored = grades.set_index("Ticker") if not grades.empty else None
    y0 = ranked.set_index("Ticker")["PeriodEnd_Y0"] if "PeriodEnd_Y0" in ranked.columns else pd.Series(dtype=object)
    out = []
    for t in dict.fromkeys(candidates):
        if stored is None or t not in stored.index:
            out.append(t)
            continue
        row = stored.loc[t]
        graded = pd.to_datetime(row.get("Graded_At"), errors="coerce")
        y0_now = pd.to_datetime(y0.get(t), errors="coerce")
        y0_then = pd.to_datetime(row.get("Y0_End"), errors="coerce")
        if (
            pd.isna(graded)
            or (now - graded).days >= MAX_AGE_DAYS
            or row.get("Prompt_Hash") not in {current, *PASS_HASH_ALIASES}
        ):
            out.append(t)
        elif pd.notna(y0_now) and (pd.isna(y0_then) or y0_now > y0_then):
            out.append(t)
    return out


def attach(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """`df` with each company's stored grade: `Grade_Moat`, `Grade_Mgmt`, `Grade_Pass` (`passes`; None ungraded),
    the two passes, `Grade_Synthesis` and `Grade_Date`, for the lists and the company pages."""
    grades = load(path)
    out = df.drop(columns=[c for c in df.columns if c.startswith("Grade_")])
    if grades.empty:
        return out.assign(Grade_Moat=float("nan"), Grade_Mgmt=float("nan"), Grade_Pass=None)
    g = grades.set_index("Ticker")
    pick = lambda col: out["Ticker"].map(g[col]) if col in g.columns else None  # noqa: E731
    out = out.assign(Grade_Moat=pick("Moat"), Grade_Mgmt=pick("Management"), Grade_Moat_1=pick("Moat_1"),
                     Grade_Mgmt_1=pick("Mgmt_1"), Grade_Moat_2=pick("Moat_2"), Grade_Mgmt_2=pick("Mgmt_2"),
                     Grade_Synthesis=pick("Synthesis"),
                     Grade_Date=pd.to_datetime(pick("Graded_At"), errors="coerce").dt.strftime("%Y-%m-%d"))  # fmt: skip
    return out.assign(Grade_Pass=[passes(m, g_) for m, g_ in zip(out["Grade_Moat"], out["Grade_Mgmt"], strict=True)])


def synthesis_due(grades: pd.DataFrame, tickers=None) -> list[str]:
    """Graded companies (of `tickers`, else all) whose published text was written with another synthesis prompt, and
    whose passes are current (`prompt_hash` or an alias): one whose passes are due is re-graded instead, synthesis
    and all, so rewriting its text now would be spent for nothing."""
    if grades.empty:
        return []
    current = synthesis_hash()
    passes_ok = grades.get("Prompt_Hash", pd.Series(None, index=grades.index)).isin({prompt_hash(), *PASS_HASH_ALIASES})
    stale = grades[(grades.get("Synthesis_Hash", pd.Series(None, index=grades.index)) != current) & passes_ok]
    out = stale["Ticker"].tolist()
    return out if tickers is None else [t for t in out if t in set(tickers)]


def _client():
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    # Without a timeout a call that never answers holds its worker forever (Nov 7, 2026: all six hung for two hours).
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=CALL_TIMEOUT_MS))


def resynthesize(tickers: list[str], ranked: pd.DataFrame, path: Path, workers: int = WORKERS, client=None) -> int:
    """Rewrite the published synthesis of `tickers` from their stored reports and fixed grades (the synthesis model
    only: about a cent a company). Grades, reports and `Graded_At` stay; `Synthesis_Hash` is brought up to date, and
    `Prompt_Hash` only from an alias (`PASS_HASH_ALIASES`): a new synthesis never makes passes from another grading
    prompt look current, which would cancel their re-grade. Saved after each company. Returns how many were
    rewritten."""
    client = client or _client()
    names = ranked.set_index("Ticker")["shortName"] if "shortName" in ranked.columns else pd.Series(dtype=object)
    lock, done, spent = threading.Lock(), [0], [0.0]

    def one(ticker: str) -> None:
        row = load(path).set_index("Ticker").loc[ticker]
        name = names.get(ticker) if isinstance(names.get(ticker), str) else ticker
        prompt = _synthesis_prompt(name, ticker, row["Moat"], row["Management"], row["Report_1"], row["Report_2"])
        text, use = _call(client, SYNTHESIS_MODEL, prompt, SYNTHESIS_OUTPUT_TOKENS, f"{ticker} synthesis")
        with lock:  # read, change one row, save: other threads' rows are kept
            grades = load(path)
            if "Synthesis_Hash" not in grades.columns:  # grades stored before the synthesis had its own hash
                grades["Synthesis_Hash"] = None
            i = grades.index[grades["Ticker"] == ticker][0]
            before_hash = grades.loc[i, "Prompt_Hash"]
            kept = prompt_hash() if before_hash in PASS_HASH_ALIASES else before_hash
            grades.loc[i, ["Synthesis", "Prompt_Hash", "Synthesis_Hash"]] = [text, kept, synthesis_hash()]
            for col, add in (("Tokens_In", use["in"]), ("Tokens_Out", use["out"]), ("Cost_USD", use["usd"])):
                before = pd.to_numeric(grades.loc[i, col], errors="coerce")
                grades.loc[i, col] = (0 if pd.isna(before) else before) + add
            save(grades, path)
            done[0] += 1
            spent[0] += use["usd"]
            log.info("Synthesis %s (%d/%d), $%.4f (total $%.2f)", ticker, done[0], len(tickers), use["usd"], spent[0])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, t) for t in tickers]):
            try:
                future.result()
            except Exception as exc:
                log.error("Synthesis not rewritten: %s", exc)
    return done[0]


# --- A grading run ------------------------------------------------------------------------------------------------


def run(tickers: list[str], ranked: pd.DataFrame, path: Path, workers: int = WORKERS) -> pd.DataFrame:
    """Grade `tickers` from their rows in `ranked` (`grade_input`: the business, never the price), saving each grade
    as it comes so an interrupted run keeps what it paid for. Returns the new grades; a company that fails is logged
    and skipped."""
    client = _client()
    today = pd.Timestamp.now().normalize()
    rows = ranked.drop_duplicates("Ticker").set_index("Ticker", drop=False)
    lock, new = threading.Lock(), []

    def one(ticker: str) -> dict | None:
        if ticker not in rows.index:
            log.warning("%s: not in the ranked run, not graded", ticker)
            return None
        row = rows.loc[ticker]
        name = row.get("shortName") if isinstance(row.get("shortName"), str) else ticker
        result = grade_company(client, ticker, name, grade_input(row), today)
        result["Y0_End"] = pd.to_datetime(row.get("PeriodEnd_Y0"), errors="coerce")
        with lock:  # save as we go
            new.append(result)
            save(pd.concat([load(path), pd.DataFrame([result])], ignore_index=True), path)
        return result

    spent = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log.error("%s not graded: %s", ticker, exc)
                continue
            if result is None:
                continue
            spent += result["Cost_USD"]
            log.info("Graded %s (%d/%d): moat %.2f, management %.2f, %s, $%.3f (total $%.2f)", ticker, i,
                     len(tickers), result["Moat"], result["Management"],
                     "passes" if passes(result["Moat"], result["Management"]) else "fails", result["Cost_USD"], spent)  # fmt: skip
    return pd.DataFrame(new, columns=COLUMNS)


# --- The checks: are the numbers a fair basis? ---------------------------------------------------------------------
# User's design, Nov 8, 2026: after the moat and management grade, a second bar on the companies a list shows. The
# earnings check asks whether the figures that made a company selectable mislead (a cyclical peak or windfall,
# one-off or accounting oddities, AI and robotics disruption, and, with net cash of at least `CASH_QUESTION_MIN_PCT`
# of the market value, why the cash sits idle); the book check, for the Balance sheet plays list only, whether the
# book is at most what the assets less the liabilities are really worth. One Argon pass each; a verdict (pass / fail)
# that shades the company on its lists as a grade fail does, and a short summary for its page. Never a score.

CHECK_MODEL = GRADE_MODEL
CHECK_OUTPUT_TOKENS = 32768
# What `check_input` carries; bump it when that changes, so every check is redone (`check_hash`).
CHECK_INPUT_VERSION = 3  # 2 (Nov 8, 2026): accounting notes kept (EV, FCF yield as terms); 3: the earnings
# check sees what sits behind the earnings (`EARNINGS_LINES`), the book check retained earnings (consultant review)
CASH_QUESTION_MIN_PCT = 20.0  # net cash / market cap, %, from which the earnings check asks about the cash (user)
EARNINGS, BOOK = "earnings", "book"
CHECK_KINDS = (EARNINGS, BOOK)
BOOK_LISTS = frozenset({"Below book"})  # the stored tab names judged on their book (shown as Balance sheet plays)
CHECK_TAGS = {EARNINGS: ("peak earnings", "one-offs", "AI risk", "cash catch"), BOOK: ("book overstated",)}
CHECK_TOPICS = {
    EARNINGS: ("Cycle", "One-offs and accounting", "AI and robotics", "Net cash"),
    BOOK: ("Assets carried high", "Hidden liabilities", "Assets carried low", "Net assets vs book"),
}
CHECK_STATUSES = ("ok", "caution", "fail")
# The earnings check fails on a caution on both of these (user's rule, Nov 8, 2026): two flattering forces at once.
TWO_CAUTIONS = ("Cycle", "One-offs and accounting")
# In an industry AI substitutes or disrupts (`lists/ai_fragile.txt`, the screen the score lists skip), a caution on AI
# and robotics fails the earnings check (user's rule, Nov 9, 2026: Artner, staffing, passed on an AI caution): there
# the company must show why it is safe.
AI_FRAGILE_FILE = "ai_fragile.txt"
# The pill of a failing topic (`parse_check`): what the reader sees beside the ticker.
TOPIC_TAGS = {"Cycle": "peak earnings", "One-offs and accounting": "one-offs", "AI and robotics": "AI risk",
              "Net cash": "cash catch", "Assets carried high": "book overstated", "Hidden liabilities": "book overstated",
              "Net assets vs book": "book overstated"}  # fmt: skip
CHECK_WORKERS = 12  # checks made at once: one call each, waiting on the model, not this machine (Nov 8, 2026)
EST_CHECK_USD = (
    0.025  # one check, one Argon pass, at the introductory rates (measured on 5 checks, Nov 8, 2026: $0.022)
)
CHECK_COLUMNS = ["Ticker", "Kind", "Pass", "Headline", "Tag", "Topics", "Why", "Report", "Checked_At", "Y0_End",
                 "Model", "Check_Hash", "Tokens_In", "Tokens_Out", "Cost_USD"]  # fmt: skip
# The book check's balance sheet, per period (label, column concept), millions of US dollars.
BOOK_LINES = (
    ("Current assets", "Current Assets"),
    ("Receivables", "Receivables"),
    ("Short-term investments", "Other Short Term Investments"),
    ("Property, plant and equipment, net", "Net PPE"),
    ("Goodwill and intangibles", "Goodwill And Other Intangible Assets"),
    ("Total assets", "Total Assets"),
    ("Current liabilities", "Current Liabilities"),
    ("Total liabilities", "Total Liabilities Net Minority Interest"),
    ("Lease obligations", "Capital Lease Obligations"),
    ("Minority interest", "Minority Interest"),
    ("Preferred stock", "Preferred Stock Equity"),
    ("Common equity", "Common Stock Equity"),
    ("Retained earnings", "Retained Earnings"),
    ("Tangible book value", "Tangible Book Value"),
)
# The earnings check's view behind the clean figures, per period (consultant review, Nov 8, 2026): reported net income
# and the unusual items the system took out (is a one-off still inside, or already stripped?), depreciation against
# capex (a capex holiday) and the change in working capital (a free cash flow jump that is only working capital).
EARNINGS_LINES = (
    ("Net income, as reported", "Net Income"),
    ("Unusual items, pre-tax (taken out of the clean figures)", "Total Unusual Items"),
    ("Depreciation and amortisation", "Depreciation And Amortization"),
    ("Capital expenditure", "Capital Expenditure"),
    ("Change in working capital (cash out is negative)", "Change In Working Capital"),
)
# A clause of a data note that speaks of a price, a market value, a yield or a verdict of ours is left out of the
# check's input (user's rule, Nov 7, 2026: the LLM judges the business, not the stock; review of Nov 8, 2026).
# A clause of a data note that speaks of the stock is left out of the check's input (user's rule, Nov 7, 2026: the LLM
# judges the business, not the stock): the market's view and our verdicts on it (`_STOCK_WORDS`), and any figure over
# a price or a score of ours (`_PRICED_NUMBER`: a yield, P/E, market value or Value / Quality with a number). Words
# such as EV, yield or Value used as accounting or method terms stay (review of Nov 8, 2026: "not subtracted from EV
# or invested capital" is why client cash is set apart, and "earning power more than twice the reported FCF yield"
# is the suspect-figures flag the earnings check needs most). Our list boilerplate is cut, not its sentence.
_STOCK_WORDS = re.compile(r"(?i)\b(?:cheap|dear|mood|percentiles?|quadrant|the market|rankings)\b")
_PRICED_NUMBER = re.compile(
    r"(?i:\b(?:yields?|P/E|prices?|priced|market cap|market value)\b[^.;:]*?\d)|\b(?:Value|Quality)\b\s*\(?\s*\d"
)
_BOILERPLATE = re.compile(
    r"(?i)[,;]?\s*(?:so it is |it is )?(?:ranked|scored)?,?\s*(?:but )?(?:left )?out of (?:the )?(?:top|highlight) "
    r"lists?(?: and the mood list \([^)]*\))?(?: until [^.;]*)?"
)


# Lists drawn from companies that already passed their own list's check (The List, Nov 9, 2026): no check of their own.
PRE_CHECKED_LISTS = frozenset({"The List"})


def check_kind(list_name: str) -> str | None:
    """The check that judges a member of the stored tab `list_name`: its book on `BOOK_LISTS`, else its earnings;
    None on `PRE_CHECKED_LISTS`, whose members passed their own list's check already."""
    if list_name in PRE_CHECKED_LISTS:
        return None
    return BOOK if list_name in BOOK_LISTS else EARNINGS


def price_free(notes) -> list[str]:
    """`notes` with our list boilerplate cut ("ranked, but left out of the top lists until checked") and every clause
    that speaks of the stock taken out (`_STOCK_WORDS`, `_PRICED_NUMBER`); a note left empty is dropped. The check
    reads the system's data notes and warning flags, never the price."""
    out = []
    for note in notes or []:
        text = _BOILERPLATE.sub("", str(note).strip())
        clauses = re.split(r"(?<=[.;:])\s+", text)
        kept = [c for c in clauses if c and not _STOCK_WORDS.search(c) and not _PRICED_NUMBER.search(c)]
        joined = re.sub(r"[:;,]\s*\.", ".", " ".join(kept))  # what the cut boilerplate left: "):." -> ")."
        joined = re.sub(r"\.{2,}", ".", joined).strip().rstrip(":;,").strip()
        if joined and re.search(r"[A-Za-z]{3}", joined):
            out.append(joined)
    return list(dict.fromkeys(out))


def asks_about_cash(row: pd.Series) -> bool:
    """True when the earnings check asks why the cash sits idle: an operating company whose shareholder-basis net cash
    is at least `CASH_QUESTION_MIN_PCT` of its market value."""
    net_cash = _num(row.get("Net_Cash_to_Cap"))
    return bool(pd.notna(net_cash) and net_cash >= CASH_QUESTION_MIN_PCT and row.get("Financial_Mode") != True)  # noqa: E712


def check_input(row: pd.Series, notes, kind: str, cfg=None) -> str:
    """What a check reads about one company: the grade's input (`grade_input`: the business, never the price), the
    book check's balance sheet by period, and the system's data notes and warning flags with every priced clause taken
    out (`price_free`). The moat and management reasoning is not passed."""
    from fq.settings import Settings

    cfg = cfg or Settings()
    out = [grade_input(row, cfg)]
    spec = (
        (BOOK_LINES, "The balance sheet, millions of US dollars at today's exchange rate")
        if kind == BOOK
        else (EARNINGS_LINES, "Behind the earnings, millions of US dollars at today's exchange rate")
    )
    table = _period_table(row, cfg, *spec)
    if table:
        out += ["", table]
    clean = price_free(notes)
    if clean:
        out += ["", "### Data notes and warning flags", *[f"- {n}" for n in clean]]
    return "\n".join(out).strip()


def _period_table(row: pd.Series, cfg, lines_spec, title: str) -> str | None:
    """A table of `lines_spec` (label, concept) by period, in millions, the periods with any figure; None when empty."""
    periods = list(cfg.chronological)
    if row.get("TTM_Identical_To_Latest") == True:  # noqa: E712
        periods.remove("TTM")
    periods = [p for p in periods if any(_present(row.get(f"{c}_{p}")) for _, c in lines_spec)]
    lines = []
    for label, concept in lines_spec:
        values = [_num(row.get(f"{concept}_{p}")) / 1e6 for p in periods]
        if any(pd.notna(v) for v in values):
            lines.append((label, _cells(values, ",.0f")))
    return _table(title, [_period_label(row, p) for p in periods], lines) if lines else None


def _check_prompt_text(kind: str, cash: bool) -> str:
    if kind == BOOK:
        return _prompt("check_book.md")
    text = _prompt("check_earnings.md")
    question = _prompt("check_cash.md").replace("{{CASH_MIN}}", f"{CASH_QUESTION_MIN_PCT:g}") if cash else ""
    topic = ',\n    {"name": "Net cash", "status": "ok | caution | fail", "line": "at most 12 words"}' if cash else ""
    return text.replace("{{CASH_QUESTION}}", question).replace("{{CASH_TOPIC}}", topic)


def check_hash(kind: str) -> str:
    """What a check was made with: its prompts, model, thinking and input version (another hash makes it due)."""
    names = ("check_book.md",) if kind == BOOK else ("check_earnings.md", "check_cash.md")
    digest = hashlib.sha1("".join(_prompt(n) for n in names).encode())
    digest.update(repr((kind, CHECK_MODEL, THINKING, CHECK_INPUT_VERSION, CASH_QUESTION_MIN_PCT)).encode())
    return digest.hexdigest()[:12]


def check_prompt(kind: str, company: str, ticker: str, figures: str, today: pd.Timestamp, cash: bool = False) -> str:
    """The prompt of one check: `figures` is `check_input`; `cash` adds the net cash question (earnings only)."""
    tags = " / ".join(f'"{t}"' for t in CHECK_TAGS[kind])
    return (_check_prompt_text(kind, cash).replace("{{COMPANY_NAME}}", f"{company} ({ticker})")
            .replace("{{EXECUTION_DATE}}", f"{today:%Y-%m-%d}").replace("{{TAGS}}", tags)
            .replace("{{COMPANY_FIGURES}}", figures))  # fmt: skip


def ai_fragile(row) -> bool:
    """True when the company's Yahoo industry is on `lists/ai_fragile.txt` (the AI screen of the screened lists)."""
    from fq.universe import _read_lines, industry_key

    industry = row.get("industry") if hasattr(row, "get") else None
    keys = {industry_key(line.split("#")[0].strip()) for line in _read_lines(AI_FRAGILE_FILE)}
    return isinstance(industry, str) and industry_key(industry) in keys


def parse_check(text: str, kind: str, fragile: bool = False) -> dict | None:
    """The verdict block of a check's answer (the last JSON object in it), checked and tidied: `Pass` (bool),
    `Headline`, `Tag` (one of `CHECK_TAGS`, failing checks only), `Topics` (name, status, line; only the topics of
    `CHECK_TOPICS`, in their order) and `Why` (markdown bullets). `fragile` (an AI-fragile industry, `ai_fragile`)
    makes a caution on AI and robotics a fail. None when no readable verdict is found: the check is then not stored as
    made, so it is asked again, and it never counts as a pass."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.DOTALL)
    candidates = blocks or re.findall(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text or "", flags=re.DOTALL)
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            continue
        given = {str(t.get("name", "")).strip().lower(): t for t in data.get("topics") or [] if isinstance(t, dict)}
        topics = []
        for name in CHECK_TOPICS[kind]:
            t = given.get(name.lower())
            if t is None:
                continue
            status = str(t.get("status", "")).strip().lower()
            topics.append({"name": name, "status": status if status in CHECK_STATUSES else "caution",
                           "line": _clean_text(t.get("line"))})  # fmt: skip
        why = data.get("why")
        if isinstance(why, list):
            why = "\n".join(f"- {_clean_text(w).lstrip('- ')}" for w in why)
        tag = str(data.get("tag") or "").strip()
        tag = next((t for t in CHECK_TAGS[kind] if t.lower() == tag.lower()), None)
        failed = verdict == "fail"
        status = {t["name"]: t["status"] for t in topics}
        # A failing topic fails the check, and on the earnings check so do cautions on both forces that flatter a
        # record (`TWO_CAUTIONS`; user's rule, Nov 8, 2026), whatever verdict the model wrote.
        if "fail" in status.values() or (kind == EARNINGS and all(status.get(n) == "caution" for n in TWO_CAUTIONS)):
            failed = True
        ai_rule = kind == EARNINGS and fragile and status.get("AI and robotics") == "caution"
        if ai_rule:
            failed = True

        # The pill names a topic that was asked (Nov 8, 2026: Haidilao failed on its cycle, Redeia on two cautions,
        # both tagged "cash catch" with no cash question): the first failing topic, else the model's tag when its
        # topic is in the answer, else the first topic under a caution, else the kind's first tag.
        def first(status: str, items: list = topics) -> str | None:
            return next(
                (TOPIC_TAGS[t["name"]] for t in items if t["status"] == status and t["name"] in TOPIC_TAGS), None
            )

        asked = {TOPIC_TAGS[t["name"]] for t in topics if t["name"] in TOPIC_TAGS}
        if failed:
            tag = (
                first("fail")
                or (tag if tag in asked else None)
                or ("AI risk" if ai_rule else None)
                or first("caution")
                or CHECK_TAGS[kind][0]
            )
        return {"Pass": not failed, "Headline": _clean_text(data.get("headline")),
                "Tag": (tag or CHECK_TAGS[kind][0]) if failed else None, "Topics": json.dumps(topics),
                "Why": str(why or "").strip()}  # fmt: skip
    return None


def _clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def check_company(client, ticker: str, company: str, figures: str, kind: str, today: pd.Timestamp,
                  cash: bool = False, fragile: bool = False) -> dict:  # fmt: skip
    """One check: one pass of `CHECK_MODEL`, asked once more when its answer has no readable verdict; without one the
    check is not made (an error), so it stays due."""
    prompt = check_prompt(kind, company, ticker, figures, today, cash)
    usage = []
    for _ in range(2):
        text, use = _call(client, CHECK_MODEL, prompt, CHECK_OUTPUT_TOKENS, f"{ticker} {kind} check")
        usage.append(use)
        parsed = parse_check(text, kind, fragile)
        if parsed is not None:
            break
    else:
        raise RuntimeError(f"{ticker} {kind} check: no readable verdict")
    return {"Ticker": ticker, "Kind": kind, **parsed, "Report": text, "Checked_At": today, "Model": CHECK_MODEL,
            "Check_Hash": check_hash(kind), "Tokens_In": sum(u["in"] for u in usage),
            "Tokens_Out": sum(u["out"] for u in usage), "Cost_USD": round(sum(u["usd"] for u in usage), 4)}  # fmt: skip


def load_checks(path: Path) -> pd.DataFrame:
    """The stored checks (one row per ticker and kind), empty when none."""
    if not path.exists():
        return pd.DataFrame(columns=CHECK_COLUMNS)
    return pd.read_parquet(path)


def save_checks(checks: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checks.drop_duplicates(["Ticker", "Kind"], keep="last").to_parquet(path, index=False)


def checks_due(candidates, ranked: pd.DataFrame, checks: pd.DataFrame, now: pd.Timestamp) -> list[tuple[str, str]]:
    """The (ticker, kind) candidates to check, in their order: never checked (or with no verdict), checked with
    another prompt (`check_hash`), on an older annual report (`PeriodEnd_Y0` moved) or `MAX_AGE_DAYS` ago or more."""
    current = {k: check_hash(k) for k in CHECK_KINDS}
    stored = checks.set_index(["Ticker", "Kind"]) if not checks.empty else None
    y0 = ranked.set_index("Ticker")["PeriodEnd_Y0"] if "PeriodEnd_Y0" in ranked.columns else pd.Series(dtype=object)
    out = []
    for ticker, kind in dict.fromkeys(candidates):
        if stored is None or (ticker, kind) not in stored.index:
            out.append((ticker, kind))
            continue
        row = stored.loc[(ticker, kind)]
        checked = pd.to_datetime(row.get("Checked_At"), errors="coerce")
        y0_now = pd.to_datetime(y0.get(ticker), errors="coerce")
        y0_then = pd.to_datetime(row.get("Y0_End"), errors="coerce")
        if (
            pd.isna(checked)
            or pd.isna(row.get("Pass"))
            or (now - checked).days >= MAX_AGE_DAYS
            or row.get("Check_Hash") != current[kind]
            or (pd.notna(y0_now) and (pd.isna(y0_then) or y0_now > y0_then))
        ):
            out.append((ticker, kind))
    return out


def estimate_checks_usd(n: int, when: pd.Timestamp) -> float:
    """What `n` checks cost, from `EST_CHECK_USD` scaled to the rates on `when`."""
    now = [r for r in RATES[CHECK_MODEL] if when >= r[0]][-1][1] / RATES[CHECK_MODEL][0][1]
    return n * EST_CHECK_USD * now


def attach_checks(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """`df` with each company's stored checks, per kind: `Check_Earnings_Pass` / `Check_Book_Pass` (True, False, or
    None unchecked), `_Headline`, `_Tag`, `_Topics` (JSON), `_Why` and `_Date`, for the lists and the company pages."""
    checks = load_checks(path)
    out = df.drop(columns=[c for c in df.columns if c.startswith("Check_")])
    for kind in CHECK_KINDS:
        prefix = f"Check_{kind.capitalize()}_"
        part = checks[checks["Kind"] == kind].set_index("Ticker") if not checks.empty else pd.DataFrame()
        if part.empty:
            out = out.assign(**{f"{prefix}Pass": None})
            continue
        pick = lambda col: out["Ticker"].map(part[col]) if col in part.columns else None  # noqa: B023, E731
        passed = [None if pd.isna(v) else bool(v) for v in pick("Pass")]
        out = out.assign(**{f"{prefix}Pass": pd.Series(passed, index=out.index, dtype=object),
                            f"{prefix}Headline": pick("Headline"), f"{prefix}Tag": pick("Tag"),
                            f"{prefix}Topics": pick("Topics"), f"{prefix}Why": pick("Why"),
                            f"{prefix}Date": pd.to_datetime(pick("Checked_At"), errors="coerce").dt.strftime("%Y-%m-%d")})  # fmt: skip
    return out


def run_checks(todo: list[tuple[str, str]], ranked: pd.DataFrame, notes: dict[str, list[str]], path: Path,
               workers: int = CHECK_WORKERS) -> pd.DataFrame:  # fmt: skip
    """Make the (ticker, kind) checks of `todo` from their rows in `ranked` (`check_input`) and `notes` (the data
    notes and warning flags by ticker, priced clauses taken out here), saving each as it comes. Returns the new
    checks; a check that fails is logged and skipped (it stays due)."""
    client = _client()
    today = pd.Timestamp.now().normalize()
    rows = ranked.drop_duplicates("Ticker").set_index("Ticker", drop=False)
    lock, new = threading.Lock(), []

    def one(ticker: str, kind: str) -> dict | None:
        if ticker not in rows.index:
            log.warning("%s: not in the ranked run, not checked", ticker)
            return None
        row = rows.loc[ticker]
        name = row.get("shortName") if isinstance(row.get("shortName"), str) else ticker
        cash = kind == EARNINGS and asks_about_cash(row)
        figures = check_input(row, notes.get(ticker, []), kind)
        result = check_company(client, ticker, name, figures, kind, today, cash, ai_fragile(row))
        result["Y0_End"] = pd.to_datetime(row.get("PeriodEnd_Y0"), errors="coerce")
        with lock:
            new.append(result)
            save_checks(pd.concat([load_checks(path), pd.DataFrame([result])], ignore_index=True), path)
        return result

    spent = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, t, k): (t, k) for t, k in todo}
        for i, future in enumerate(as_completed(futures), 1):
            ticker, kind = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log.error("%s %s check not made: %s", ticker, kind, exc)
                continue
            if result is None:
                continue
            spent += result["Cost_USD"]
            log.info("Checked %s %s (%d/%d): %s%s, $%.3f (total $%.2f)", ticker, kind, i, len(todo),
                     "passes" if result["Pass"] else "fails", f" ({result['Tag']})" if result["Tag"] else "",
                     result["Cost_USD"], spent)  # fmt: skip
    return pd.DataFrame(new, columns=CHECK_COLUMNS)
