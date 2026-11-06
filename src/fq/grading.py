"""Moat and management grades (1 to 5) by Gemini: the gate on the highlight lists.

The only module that calls an LLM (user's choice, Oct 31, 2026). A grade never enters a score, a rank or a flag; it
only decides where a company sits inside a highlight list it already made (`passes`). The procedure: a first pass; a
second, independent one only when the first clears the gate, the two then averaged (user's rule, Nov 6, 2026: a
company failing on the first pass could make no list); then a synthesis that justifies the grade. The company's page in
this system is passed in as markdown (`page_markdown`) as context, not the verdict: the prompt also asks what the
figures cannot show (revenue dependency on one customer, platform, product or licence, with a cap on the moat; user's
rule, Nov 6, 2026). No search tool: the model's own knowledge and our figures. Grades are kept in `data/grades.parquet` (`load`, `save`) and redone only when due
(`due`: a new annual report or a year old). The google-genai client is imported only when a grading runs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import struct
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
MAX_AGE_DAYS = 365  # re-grade after a year, or when a new annual report lands (`due`)
MAX_PER_RUN = 60  # companies graded by one `fq run -u benchmark` (~$6 now); `fq grade` has no cap

# The gate (user's rule, Oct 31, 2026): both grades at least MIN_BOTH, or either at least MIN_ONE.
MIN_BOTH = 3.0
MIN_ONE = 4.25
# The Shortlist's stricter gate (user's rule, Oct 31, 2026): both at least MIN_BOTH and either at least SHORTLIST_MIN_TOP.
SHORTLIST_MIN_TOP = 4.0

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
# replaced that day, and no stored grade carried it any more.
PASS_HASH_ALIASES: frozenset[str] = frozenset()


def passes(moat, mgmt) -> bool | None:
    """The gate: True when both grades reach `MIN_BOTH` or either reaches `MIN_ONE`; None when ungraded."""
    if pd.isna(moat) or pd.isna(mgmt):
        return None
    return bool((moat >= MIN_BOTH and mgmt >= MIN_BOTH) or max(moat, mgmt) >= MIN_ONE)


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


# --- The company page as markdown -------------------------------------------------------------------------------


def _array(v):
    """A Plotly array, plain or binary-encoded (`{"dtype", "bdata"}`, little-endian)."""
    if isinstance(v, dict) and "bdata" in v:
        code = {"i1": "b", "u1": "B", "i2": "h", "u2": "H", "i4": "i", "u4": "I", "f4": "f", "f8": "d"}[v["dtype"]]
        raw = base64.b64decode(v["bdata"])
        return list(struct.unpack("<" + code * (len(raw) // struct.calcsize(code)), raw))
    return v


def _number(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _chart_tables(soup) -> list[str]:
    """Each Plotly figure on the page as one markdown table: a row per series, a column per period, labelled with the
    axis' tick text (Dec-24, TTM Jun-26) where the figure has one."""
    tables, decoder = [], json.JSONDecoder()
    for script in soup.find_all("script"):
        src = script.string or ""
        found = re.search(r'Plotly\.newPlot\(\s*"([^"]+)",\s*', src)
        if not found:
            continue
        try:
            traces, end = decoder.raw_decode(src, found.end())
            layout, _ = decoder.raw_decode(src, re.compile(r"\s*,\s*").match(src, end).end())
        except (json.JSONDecodeError, AttributeError):
            continue
        axis = layout.get("xaxis", {})
        ticks = dict(zip(map(str, axis.get("tickvals") or []), axis.get("ticktext") or [], strict=False))
        series = []
        for trace in traces:
            y = _array(trace.get("y"))
            if not isinstance(y, list) or not y or all(v is None for v in y):
                continue
            x = _array(trace.get("x")) or list(range(len(y)))
            series.append((trace.get("name") or "series", dict(zip(map(str, x), y, strict=False))))
        if not series:
            continue
        periods = list(dict.fromkeys(p for _, values in series for p in values))
        lines = [f"### Chart data: {found.group(1).replace('_', ' ')}",
                 "| series | " + " | ".join(ticks.get(p, p) for p in periods) + " |",
                 "|" + "---|" * (len(periods) + 1)]  # fmt: skip
        lines += ["| " + name + " | " + " | ".join(_number(values.get(p)) for p in periods) + " |"
                  for name, values in series]  # fmt: skip
        tables.append("\n".join(lines))
    return tables


# What `page_markdown` carries; bump it when that changes, so every grade is redone (`prompt_hash`, `due`).
# 2 (Oct 31, 2026): this system's conclusions taken out (badges, scores, ranks, percentiles, the cheapness verdict).
PAGE_VERSION = 2
# Data notes that are this system's verdicts, not data (`ranking.dislocation`'s Mood_Check).
CONCLUSION_NOTES = ("Cheap (Value",)
_PERCENTILE = re.compile(r" p\d+ · w[\d.]+")  # a criterion's percentile and weight, `p82 · w0.2`
_SCORING = re.compile(r"\s*(?:Higher|Lower) is better\.|\s*Scored: [^)]*?\.(?=\)|$)|\s*Shown[^.]*?never scored\.")


def _neutral(line: str) -> str:
    """A page line without the scoring: percentiles, weights, better-direction and "Scored / never scored" clauses."""
    line = _SCORING.sub("", _PERCENTILE.sub("", line))
    line = re.sub(
        r"(#+ )[0-9]+ data notes?$", lambda m: m.group(1) + "Data notes", line
    )  # the count is stale once notes are cut
    return line.replace(" ()", "")


def page_markdown(html: str) -> str:
    """A company page (`company.html.j2`) as compact markdown: headings, the summary and notes, every figure as
    `label: value (definition)` and each chart's data as a table. Navigation, links, the footer and the grade card
    itself are left out, so a grade never reads an earlier grade; so is every conclusion of this system (user's rule,
    Oct 31, 2026: hard data only, nothing to anchor the grade): badges (Compounder, ...), the score panel (scores,
    ranks, its sentence), the portfolio tag, the "Cheap (Value ...)" note, each criterion's percentile and weight,
    and the scoring clauses of the definitions."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    tables = _chart_tables(soup)
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for tag in soup.select(".grade-card, .external-links, .page-nav, .company-website, .chart-instruction, "
                           ".badges, .score-panel, .tag, .why-here, .tile-pct, .holders-card"):  # fmt: skip
        tag.decompose()
    for li in soup.select(".notes li"):
        if li.get_text(" ", strip=True).startswith(CONCLUSION_NOTES):
            li.decompose()
    main = soup.find("main") or soup.body or soup
    lines, seen = [], set()
    for el in main.find_all(["h1", "h2", "h3", "p", "dt", "li", "div"]):
        classes = el.get("class") or []
        text = el.get_text(" ", strip=True)
        if el.name in ("h1", "h2", "h3"):
            lines.append(f"\n{'#' * (int(el.name[1]) + 1)} {text}")
        elif el.name == "p" and not el.find_parent(["dd", "li"]):
            lines.append(text)
        elif el.name == "dt":
            dd = el.find_next_sibling("dd")
            why = el.parent.get("data-help") if el.parent is not None else None
            lines.append(f"- {text}: {dd.get_text(' ', strip=True) if dd else ''}" + (f" ({why})" if why else ""))
        elif el.name == "li":
            why = el.get("data-help")
            lines.append(f"- {text}" + (f" ({why})" if why else ""))
        elif el.name == "div" and "tile" in classes:
            value, label = el.find(class_="tile-value"), el.find(class_="tile-label")
            if value and label:
                why = el.get("data-help")
                lines.append(f"- {label.get_text(' ', strip=True)}: {value.get_text(' ', strip=True)}"
                             + (f" ({why})" if why else ""))  # fmt: skip
    lines = [_neutral(line) for line in lines]
    body = "\n".join(line for line in lines if line.strip() and not (line in seen or seen.add(line)))
    return body.strip() + ("\n\n" + "\n\n".join(tables) if tables else "")


# --- Prompts and calls --------------------------------------------------------------------------------------------


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def prompt_hash() -> str:
    """What the grades were produced with: the grading prompt, model, passes and page (stored; another hash makes a
    grade due). The synthesis has its own (`synthesis_hash`), so rewording it never redoes the costly passes."""
    digest = hashlib.sha1(_prompt("grade_prompt.md").encode())
    digest.update(repr((GRADE_MODEL, THINKING, PASSES, SECOND_PASS_ONLY_IF_GATE, PAGE_VERSION)).encode())
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
                                             execution_date=f"{today:%Y-%m-%d}", company_page=page)  # fmt: skip


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

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


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


def run(tickers: list[str], ranked: pd.DataFrame, pages_dir: Path, path: Path, workers: int = WORKERS) -> pd.DataFrame:
    """Grade `tickers` from their pages in `pages_dir` (as the last report wrote them), saving each grade as it comes
    so an interrupted run keeps what it paid for. Returns the new grades; a company that fails is logged and skipped."""
    from fq.report.links import slug

    client = _client()
    today = pd.Timestamp.now().normalize()
    rows = ranked.set_index("Ticker")
    lock, new = threading.Lock(), []

    def one(ticker: str) -> dict | None:
        page_file = pages_dir / f"{slug(ticker)}.html"
        if not page_file.exists():
            log.warning("%s: no company page at %s, not graded", ticker, page_file)
            return None
        row = rows.loc[ticker] if ticker in rows.index else pd.Series(dtype=object)
        name = row.get("shortName") if isinstance(row.get("shortName"), str) else ticker
        result = grade_company(client, ticker, name, page_markdown(page_file.read_text(encoding="utf-8")), today)
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
