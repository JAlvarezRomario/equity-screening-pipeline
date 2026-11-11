"""The tags and pills beside a ticker: one vocabulary for the front page and the company page (no markup here)."""

import pandas as pd

from fq.settings import Settings


def tags(row: pd.Series) -> list[dict]:
    """Every tag a company carries, in one vocabulary for both pages (user's request, Oct 31, 2026: the front page
    and the company page say the same thing): `word` is the pill beside its ticker in the index tables (after
    `mark`), and capitalised the badge on its page; `why` the tooltip of both; `cls` the pill's highlight. Lists are
    named as their tabs (an asset play is "below book"). `pill=False` only for the ones the index says otherwise."""
    # Imported here: screens imports format, which imports this module.
    from fq.metrics.capital import NCAV_EXCLUDED_INDUSTRIES
    from fq.universe import industry_key

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
        add("below book", "Tangible book per share (net debt counted against it; a holding's book with its "
            f"goodwill) at or above the share price, the book earning at least {s.asset_play_min_roic:g}% (normalised "
            "ROIC), with positive owner earnings, bounded debt and no dilution: the Balance sheet plays list's test.")  # fmt: skip
    net_cash = pd.to_numeric(row.get("Net_Cash_to_Cap"), errors="coerce")
    if net_cash >= s.net_cash_min_pct and not flag("Financial_Mode"):
        out.append(
            {
                "word": "net cash",
                "why": f"Cash less all debt, on the shareholders' share, is {net_cash:.0f}% of "
                f"the market value (at least {s.net_cash_min_pct:g}%): a buyer gets the cash with the business.",
                "cls": "",
                "mark": "",
                "badge": False,
            }
        )  # the company page has its own net cash badge
    ncav = pd.to_numeric(row.get("NCAV_to_Price"), errors="coerce")
    industry = row.get("industry")
    land_inventory = isinstance(industry, str) and industry_key(industry) in NCAV_EXCLUDED_INDUSTRIES
    if ncav >= NET_NET_PILL_MIN and not flag("Financial_Mode") and not land_inventory:
        add("net-net", f"Net current assets (current assets less every liability, preferred and minorities) are "
            f"{ncav:.0f}% of the share price: the fixed assets come at no cost.")  # fmt: skip
    if flag("Financial_Mode") and isinstance(row.get("Business_Model"), str):
        out.append({"word": row["Business_Model"], "why": FINANCIAL_PILL, "cls": "", "mark": "", "badge": False})
    if row.get("Rank_Group") == "real_estate":
        out.append({"word": "REIT", "why": REIT_PILL, "cls": "", "mark": "", "badge": False})
    if row.get("Rank_Group") == "utility":
        out.append({"word": "utility", "why": UTILITY_PILL, "cls": "", "mark": "", "badge": False})
    if row.get("Rank_Group") == "unprofitable":
        add("loss-making", "Owner earnings or return on capital not positive in a normal year: scored, but not in the "
            "rankings, which hold profitable operating companies only.")  # fmt: skip
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
    if flag("Deep_Discount") and flag("Check_Book_Pass"):
        add("deep discount to book", f"Priced under {DEEP_PILL_MAX:g}x both book and tangible book per share, and an "
            "AI check of the book (assets carried high, hidden liabilities) found it holds: the Deep discount list's "
            "test, before the management grade.")  # fmt: skip
    record = short_record(row)
    if record:
        add("short record", f"{record}: on no list until the record is longer (a new issue's seller picks the date and "
            "the figures). Scored and ranked as usual.")  # fmt: skip
    fell = pd.to_numeric(row.get("Momentum_6M"), errors="coerce")
    if fell < FALLEN_PILL_MIN:  # no badge: the company page shows the 6-month change as a chip
        out.append({"word": f"down {-fell:.0f}% in 6m", "why": "The share fell this much over 6 months: shown, never "
                    "tested. Read why before buying; the figures may not show it yet.", "cls": "", "mark": "",
                    "badge": False})  # fmt: skip
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


def short_record(row: pd.Series) -> str:
    """Why one company fails the lists' record test (`screens.has_record`), or "" when it passes."""
    if "Has_Record" in row.index:  # stored once per report (`highlights.prepare`)
        if row["Has_Record"] is not False and row["Has_Record"] != False:  # noqa: E712 - NaN or numpy bool
            return ""
        years, listed = int(row["Record_Years"]), pd.to_numeric(row["Listed_Years"], errors="coerce")
    else:
        from fq.report import screens as sc  # screens imports format, which imports this module

        frame = row.to_frame().T
        if bool(sc.has_record(frame).iloc[0]):
            return ""
        years, listed = int(sc.roic_years(frame).iloc[0]), sc.listed_years(frame).iloc[0]
    parts = [f"{years} annual report{'s' if years != 1 else ''} with a return on capital"]
    if pd.notna(listed):
        parts.append(f"share counts or dividends over {listed:.0f} year{'s' if listed != 1 else ''}")
    return " and ".join(parts)


DEEP_PILL_MAX = 0.66  # as `screens.DEEP_DISCOUNT_MAX_PRICE_TO_BOOK` (screens imports format, which imports this module)
# A share down more than this % over 6 months carries a pill (Nov 10, 2026: shown, never tested, since the lists'
# price floor went).
FALLEN_PILL_MIN = -20.0
BUYBACK_PILL_MIN = 2.0  # %/yr: a compounder that also shrinks its share count (the former Compounder-cannibals tab)
NET_NET_PILL_MIN = 100.0  # NCAV / price, %: the "net-net" pill (Graham)
WHT0_PILL = ("No withholding tax on dividends paid to non-resident individuals in its country (statutory rate, "
             "lists/dividend_wht.txt).")  # fmt: skip
OWNER_LED_MIN = 20.0  # % of shares held by insiders for the "owner-led" pill (the benchmark's median is about 13%)
FINANCIAL_PILL = ("Measured as a financial: net income, return on equity and book value; judged against its own "
                  "kind only, not in the rankings.")  # fmt: skip
REIT_PILL = ("An equity REIT: judged against other REITs in the Real estate list, not in the rankings (its book is "
             "depreciated cost under US rules, fair value under IFRS).")  # fmt: skip
UTILITY_PILL = ("A utility: judged against other utilities in the Utilities list, not in the rankings (regulated or "
                "contracted returns on assets financed with debt).")  # fmt: skip


def list_pills(row: pd.Series) -> list[dict]:
    """Small tags beside a ticker in the index tables: every tag of `tags`, with its mark and highlight."""
    return [{"label": t["mark"] + t["word"], "why": t["why"], "cls": t["cls"]} for t in tags(row)]
