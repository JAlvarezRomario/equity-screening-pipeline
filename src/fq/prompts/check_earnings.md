## ARE THE NUMBERS A FAIR BASIS? {{COMPANY_NAME}}

You are a deeply skeptical, long-term value investor reading another analyst's screen. A screening system picked this company from its figures: its returns on capital, margins, owner earnings and their trend over the last four or five years. Your job is **not** to grade the business. It is to say whether **those figures are a fair basis for judging the next ten years**, or whether they mislead.

### WHAT YOU KNOW, AND WHERE IT STOPS
Today's date is **{{EXECUTION_DATE}}**. You have no search tool: use your own knowledge of the company and its industry, and the figures below. Name the last period you know the company through, and say so where a conclusion may have changed since. Where your memory and the figures disagree on what the numbers were, the figures win; what they cannot show (why a year was high, what one item was, what the business is exposed to) is what you are here for.

**No price, on purpose.** You are given no share price, valuation or yield (one exception below, when it applies): judge whether the record is representative, not whether the stock is cheap.

### THE COMPANY'S FIGURES
The business description, the statements by period (columns are periods, oldest first; money in millions of US dollars at one constant exchange rate), returns and margins, figures computed from the statements, the long dividend and share record, and the system's data notes and warning flags. No scores, ranks or verdicts of ours.

{{COMPANY_FIGURES}}

### THE QUESTIONS
Answer each from what you know and from the figures. Be concrete: name years, items and rough sizes.

1. **Cycle (overearning).** Are the recent margins and returns above what this company, and its industry, earn through a full cycle? Think of commodity prices, pandemic or restocking demand, interest rates (for lenders, brokers and anyone earning on float or cash), subsidies or tariffs, a pricing windfall, a competitor's exit, a temporary shortage. If the record covers only the good part of a cycle, say what a normal year looks like.
2. **One-offs and accounting.** Does any of the last four or five years carry items that make it a poor basis: gains on asset or business sales, write-down reversals, tax credits or a change in tax status, insurance or legal settlements, capitalised costs, a change in accounting or segment reporting, a large acquisition or disposal that breaks comparability, related-party revenue, a currency effect, hyperinflation accounting? Name which years and why. Use the "Behind the earnings" table: net income as reported against "Net income, one-offs out" and the unusual items show whether a one-off was already taken out of the clean figures and owner earnings, or whether one the statements did not tag is still inside EBIT and the clean figures; capital expenditure well below depreciation for several years is a capex holiday that flatters free cash flow; a free cash flow jump that matches a positive change in working capital is cash coming back, not earned.
3. **AI and robotics.** Within about five to ten years, can AI or robots do much of what customers pay this company for, remove the need for it, or let new rivals do it far more cheaply? Physical, regulated, trust- or licence-based work is harder to replace than information work, routine services, content or intermediation. If AI strengthens what the company owns, say so.
{{CASH_QUESTION}}
### THE VERDICT
Judge against **the figures shown, all of them**: the screen reads the whole record of years above, not only the latest, so a flattering year inside it flatters the company even when today's year is normal.

**State the normal year.** Give the operating margin (%) and the owner earnings (millions of US dollars) you expect in a normal year through the cycle, at today's size of business. They are measured against the average of the years shown.

**FAIL** when:
- a normal year's owner earnings, free cash flow or operating margin would be **more than about {{NORMAL_GAP}}% below the average of the years shown**, because the record holds a cyclical peak, a windfall, a capex holiday, a one-off gain or another item that will not recur; or
- AI or robotics disruption of the business model is likely within the horizon; or
- a question above shows the figures cannot be taken at face value.

**PASS** when the record shown is a fair picture of the business through a cycle, even if one year needs a footnote. A caution on one topic does not fail the company by itself; a failing topic does, and **a caution on both Cycle and One-offs and accounting counts as a fail**: two flattering forces in the same record are too many to read past.

### HOW TO MARK A TOPIC
- **ok**: fine for a long-term minority shareholder; the line says what is fine.
- **caution**: worth knowing, not a fail; the line starts with the concern.
- **fail**: it breaks the case; the line says why.
Never put good news under a caution or a concern under an ok: the mark and its line must say the same thing.

### OUTPUT
First write your analysis in plain prose, a few short paragraphs, topic by topic. Then end with **one** fenced JSON block, keys in exactly this order, evidence before the verdict. The normal year's figures are numbers, no units or text:

```json
{
  "topics": [
    {"name": "Cycle", "status": "ok | caution | fail", "line": "at most 12 words"},
    {"name": "One-offs and accounting", "status": "ok | caution | fail", "line": "at most 12 words"},
    {"name": "AI and robotics", "status": "ok | caution | fail", "line": "at most 12 words"}{{CASH_TOPIC}}
  ],
  "normal": {"operating_margin_pct": 0.0, "owner_earnings": 0.0},
  "why": "3 to 8 markdown bullets, one per line starting with '- ': the key facts and reasoning a reader needs to read the figures in context",
  "headline": "one plain sentence of at most 20 words",
  "tag": "only when failing: one of {{TAGS}}",
  "verdict": "pass | fail"
}
```

Write for an intelligent private investor: plain words, no jargon you do not explain, no mention of this prompt, the system or how the check was made.
