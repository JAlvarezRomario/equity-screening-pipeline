"""The tags and pills beside a ticker: one vocabulary for the front page and the company page (no markup here)."""

import pandas as pd

from fq.settings import Settings


def tags(row: pd.Series) -> list[dict]:
    """Every tag a company carries, in one vocabulary for both pages (user's request, Oct 31, 2026: the front page
    and the company page say the same thing): `word` is the pill beside its ticker in the index tables (after
    `mark`), and capitalised the badge on its page; `why` the tooltip of both; `cls` the pill's highlight. Lists are
    named as their tabs (an asset play is "below book"). `pill=False` only for the ones the index says otherwise."""
    s, out = Settings(), []
    flag = lambda col: row.get(col) == True  # noqa: E731, E712 - handles NaN / numpy bool

    def add(word: str, why: str, cls: str = "", mark: str = "") -> None:
        out.append({"word": word, "why": why, "cls": cls, "mark": mark})

    if flag("Compounder"):
        add("compounder", "High returns on capital that are steady or rising, revenue and owner "
            "earnings per share compounding smoothly, shares not meaningfully rising, and recent numbers held.",
            "pill-comp", "★ ")  # fmt: skip
    if flag("Cannibal"):
        add("cannibal", f"Retiring more than {s.cannibal_min_share_reduction:.0f}% of its shares a year, paid from its "
            f"own free cash flow and spare cash, with positive ROIC and Value >= {s.cannibal_min_value:.0f}.",
            "pill-cannibal")  # fmt: skip
    reduction = pd.to_numeric(row.get("Share_Reduction_Rate"), errors="coerce")
    if flag("Compounder") and not flag("Cannibal") and reduction >= BUYBACK_PILL_MIN:
        add("buyback", f"A compounder retiring {reduction:.1f}% of its shares a year (at least {BUYBACK_PILL_MIN:g}%).",
            "pill-cannibal")  # fmt: skip
    if flag("Asset_Play"):
        add("below book", "Tangible book per share (net debt counted against it) at or above the share price, or "
            "growing fast enough to pass it within 3 years, with positive owner earnings and ROIC, bounded debt and "
            "no dilution: the Below book list's test.")  # fmt: skip
    ncav = pd.to_numeric(row.get("NCAV_to_Price"), errors="coerce")
    if ncav >= NET_NET_PILL_MIN:
        add("net-net", f"Net current assets (current assets less every liability, preferred and minorities) are "
            f"{ncav:.0f}% of the share price: the fixed assets come at no cost.")  # fmt: skip
    if flag("Financial_Mode") and isinstance(row.get("Business_Model"), str):
        out.append({"word": row["Business_Model"], "why": FINANCIAL_PILL, "cls": "", "mark": "", "badge": False})
    if flag("Cyclical"):
        add("cyclical", "Profits change sign or swing a lot, or a cyclical industry: Value uses mid-cycle earnings.")
    if flag("Peak_Margin"):
        add("peak margin", f"A cyclical whose latest operating margin is above {s.peak_margin_ratio:g}x its usual "
            "level: scored, but out of the highlight lists.")  # fmt: skip
    if flag("Distress"):
        add("distress", f"Altman Z'' below {s.distress_z:g} with negative free cash flow or interest cover below "
            f"{s.distress_interest_cover:g}x: scored, but out of the highlight lists.")  # fmt: skip
    if flag("Revenue_Decline"):
        trend = pd.to_numeric(row.get("Revenue_Trend_Real"), errors="coerce")
        detail = f" (trend {trend:+.1f}%/yr)" if pd.notna(trend) else ""
        add("revenue ↓", f"Real USD revenue is shrinking{detail}: a smooth trend of {s.revenue_decline_rate:g}%/yr or "
            "worse, or a fall in every reported year. Scored, but out of the highlight lists.")  # fmt: skip
    if flag("Accounting_Flag"):
        reasons = row.get("Accounting_Reasons")
        add("accruals", "Earnings ahead of cash" + (f": {reasons}" if isinstance(reasons, str) else "")
            + ". Scored, but out of the highlight lists.")  # fmt: skip
    if flag("Data_Suspect"):
        reasons = row.get("Suspect_Reasons")
        add(
            "check data",
            (f"{reasons}. " if isinstance(reasons, str) else "") + "Scored, but out of the highlight lists.",
        )
    if flag("Insufficient_Data"):
        add("insufficient data", "Not enough data to score Value or Quality: not ranked.")
    if isinstance(row.get("Model_Review"), str):
        add("model to confirm", f"{row['Model_Review']}. Scored, but out of the highlight lists until the business "
            "model is confirmed.")  # fmt: skip
    if row.get("Dividend_WHT") == 0:
        add("0% WHT", WHT0_PILL, "pill-wht0", "◆ ")
    insiders = pd.to_numeric(row.get("heldPercentInsiders"), errors="coerce") * 100
    if insiders >= OWNER_LED_MIN:
        add("owner-led", f"Insiders hold {insiders:.0f}% of the shares (Yahoo): management's own money is in it. A "
            "controlling family or parent company counts too, and can also mistreat minorities.")  # fmt: skip
    return out


def company_flags(row: pd.Series) -> list[tuple[str, str]]:
    """(badge, why) for the company page: every tag of `tags` but the yardstick (its page names the model), each
    the index pill's word capitalised."""
    return [(t["word"][0].upper() + t["word"][1:], t["why"]) for t in tags(row) if t.get("badge", True)]


BUYBACK_PILL_MIN = 2.0  # %/yr: a compounder that also shrinks its share count (the former Compounder-cannibals tab)
NET_NET_PILL_MIN = 100.0  # NCAV / price, %: the "net-net" pill (Graham)
WHT0_PILL = ("No withholding tax on dividends paid to non-resident individuals in its country (statutory rate, "
             "lists/dividend_wht.txt).")  # fmt: skip
OWNER_LED_MIN = 20.0  # % of shares held by insiders for the "owner-led" pill (the benchmark's median is about 13%)
FINANCIAL_PILL = ("Measured as a financial: net income, return on equity and book value; Value against its own "
                  "kind only.")  # fmt: skip


def list_pills(row: pd.Series) -> list[dict]:
    """Small tags beside a ticker in the index tables: every tag of `tags`, with its mark and highlight."""
    return [{"label": t["mark"] + t["word"], "why": t["why"], "cls": t["cls"]} for t in tags(row)]
