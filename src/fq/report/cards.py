"""The company page's grade pill and cards: moat and management, and who owns it (no markup here)."""

import json
import re

import pandas as pd

from fq.grading import MIN_BOTH, MIN_ONE, SHORTLIST_MIN_TOP, passes_shortlist
from fq.report.format import DASH

GRADE_PILL = ("Moat and management, 1 to 5, graded by an AI model (Gemini) from its own knowledge and this site's "
              "figures. Shown, never scored.")  # fmt: skip


def _grade_text(v) -> str:
    """A grade as written: 4.0, 3.5, 4.25 (the average of two half-point passes)."""
    return f"{v:.1f}" if float(v) * 2 == round(float(v) * 2) else f"{v:.2f}"


def grade_pill(row: pd.Series) -> dict:
    """The moat and management tag of a highlight-list row: both grades, marked low when below the list's bar
    (`grading.passes`), or `not graded`."""
    moat, mgmt = row.get("Grade_Moat"), row.get("Grade_Mgmt")
    if pd.isna(moat) or pd.isna(mgmt):
        return {"label": "not graded", "why": "Not graded on moat and management yet; it keeps its place meanwhile.",
                "cls": "pill-grade-none"}  # fmt: skip
    low = row.get("Grade_Pass") is False or row.get("Grade_Pass") == False  # noqa: E712
    rule = (
        row.get("Grade_Bar")
        if isinstance(row.get("Grade_Bar"), str)
        else (f"both at least {MIN_BOTH:g}, or either at least {MIN_ONE:g}")
    )
    bar = f" Below this list's bar ({rule}): shaded at the bottom."
    why = GRADE_PILL + (bar if low else "")
    return {"label": f"moat {_grade_text(moat)} · mgmt {_grade_text(mgmt)}", "why": why,
            "cls": "pill-grade-low" if low else "pill-grade"}  # fmt: skip


def _plain(text: str) -> str:
    """Markdown emphasis and code marks taken out (the page shows text, never the model's markup)."""
    return re.sub(r"[*`]+", "", text).strip()


def grade_card(row: pd.Series) -> dict | None:
    """The company page's moat and management card, for a company a highlight list shows and that is graded: both
    grades, each pass, whether it clears the lists' bar, and the synthesis as (heading, paragraphs) sections."""
    moat, mgmt = row.get("Grade_Moat"), row.get("Grade_Mgmt")
    if row.get("Grade_Listed") != True or pd.isna(moat) or pd.isna(mgmt):  # noqa: E712
        return None
    sections: list[tuple[str, list[str]]] = []
    for line in str(row.get("Grade_Synthesis") or "").splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("#"):
            sections.append((_plain(line.lstrip("#")), []))
        elif sections:
            sections[-1][1].append(_plain(line))
        else:
            sections.append(("", [_plain(line)]))
    low = row.get("Grade_Pass") is False or row.get("Grade_Pass") == False  # noqa: E712
    verdict = "Below the lists' bar: shown at the bottom, shaded" if low else "Clears the lists' bar"
    if row.get("Grade_Shortlisted") == True:  # noqa: E712
        strict = passes_shortlist(moat, mgmt)
        verdict += (
            (
                "; Boring businesses' stricter bar too"
                if strict
                else f"; below Boring businesses' stricter bar (both at least {MIN_BOTH:g} and either at least "
                f"{SHORTLIST_MIN_TOP:g}), so shaded at the bottom there"
            )
            if not low
            else ""
        )
    return {"moat": _grade_text(moat), "mgmt": _grade_text(mgmt), "low": low,
            "verdict": verdict,
            "detail": f"Graded {row.get('Grade_Date') or 'N/A'}.",
            "sections": [s for s in sections if s[1]]}  # fmt: skip


def _holder_pct(pct: float) -> str:
    return f"{pct:.1f}%" if pct >= 1 else f"{pct:.2f}%"


def _holder_change(h: dict) -> tuple[str, str]:
    """(text, class) of a holder's change: an institution's or fund's change since its previous filing, an insider's
    latest transaction with its date."""
    if h.get("latest"):
        return (f"{h['latest']}, {h['latest_date']}" if h.get("latest_date") else h["latest"]), ""
    change = h.get("change")
    if change is None:
        return DASH, ""
    if change > 999:  # from next to nothing: a position just opened
        return "New position", "pos"
    if change == 0:
        return "Unchanged", ""
    return f"{change:+.1f}%", "pos" if change > 0 else "neg" if change < 0 else ""


def holders_card(row: pd.Series) -> dict | None:
    """The company page's "Who owns it" card from `Holders` (`fetch.top_holders`): the largest holders with their
    type, % of the shares, change and date, plus the institutions' total. None when the holders were never fetched
    (`Holders_At` missing); `rows` empty when Yahoo has no records for the company. Shown, never scored."""
    at = row.get("Holders_At")
    if not isinstance(at, str) or not at:
        return None
    try:
        holders = json.loads(row.get("Holders") or "[]")
    except (TypeError, ValueError):
        holders = []
    rows = []
    for h in holders:
        change, cls = _holder_change(h)
        rows.append({"name": h.get("name"), "type": h.get("type"), "position": h.get("position") or DASH,
                     "pct": _holder_pct(float(h.get("pct") or 0)), "change": change, "cls": cls,
                     "date": h.get("date") or DASH})  # fmt: skip
    inst, count = pd.to_numeric(row.get("heldPercentInstitutions"), errors="coerce"), row.get("Institutions_Count")
    line = None
    if pd.notna(inst):
        line = f"Institutions hold {inst * 100:.1f}% of the shares"
        if pd.notna(pd.to_numeric(count, errors="coerce")) and count > 0:
            line += f" ({count:,.0f} institutions)"
    types = [h.get("type") for h in holders]
    # Outside the US and UK Yahoo often names only US-registered funds: the company's anchor holders are not there.
    # Insiders' indirect holdings may be one block counted under each name that shares it (HCA's Frist brothers).
    return {"rows": rows, "line": line, "date": at, "funds_only": bool(types) and set(types) == {"Fund"},
            "shared": types.count("Insider") > 1}  # fmt: skip
