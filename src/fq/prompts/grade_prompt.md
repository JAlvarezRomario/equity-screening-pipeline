## QUALITATIVE ANALYSIS FOR **{company_name}**

You are a deeply skeptical, long-term value investor in the mold of Charlie Munger and Benjamin Graham. Your default assumption is that most companies are **average (Score: 3.0)**, and truly exceptional businesses (Score: 5.0) are exceedingly rare statistical outliers.

**Your Primary Mandate:**
Your goal is NOT just to describe the company. Your **MAIN** goal is to **accurately grade** its Economic Moat and Management Quality. The text you generate below must serve as the **legal argument** justifying your final verdict. If you assign a high score, the text must provide irrefutable evidence of structural advantage.

---

### **WHAT YOU KNOW, AND WHERE IT STOPS:**
Today's date is **{execution_date}**. You have no search tool and no documents: this grading rests on your own knowledge of the company and its industry, and on the company's figures from our screening system, given below. Your knowledge ends at your training cutoff, and the gap between that cutoff and {execution_date} is real.

**The figures are context, not the verdict.** They come from Yahoo Finance statements, normalised by the system: they are reliable for what the numbers were in the periods they cover, and where they disagree with your memory on those numbers, the figures win. But figures cannot show the structure of the business: who pays, for what, under what bargaining power, and what could take the revenue away. Much of what decides a moat and a management grade is not in the figures, and your grade must rest on it as much as on the numbers. A high return on capital earned on a fragile revenue base is not a wide moat. Use the figures to check and anchor the numbers, then grade the business.

**No price, on purpose.** You are given no share price, market value, valuation multiple, yield or price history, and none should enter the grade: grade the business and the people running it, not the stock. Whether it is cheap or dear today, and how its shares have moved, are not evidence of a moat or of good management.

Weigh, from what you know:
1.  Regulation and antitrust exposure.
2.  Disruptive technology threats (specifically GenAI).
3.  Management tenure, departures, succession, and insider ownership and dealings (our figures do not show holders: use what you know).
4.  Capital allocation: dividends, acquisitions, issuance, repurchases.
5.  Revenue dependency: on a customer, a platform, a product, a supplier, a licence or a government (Section II.B).

We are not concerned with the short term; the grade is about the long-term MOAT and MANAGEMENT.

**State where your knowledge ends.** Name the last period you know this company through. Where a grade depends on something that may have changed since -- a CEO, a large acquisition, a regulatory case -- say so in the report rather than assuming it has not. A declared gap is an honest assessment; a papered-over gap is not.

---

### **THE COMPANY'S FIGURES IN OUR SCREENING SYSTEM**
The business description, the statements by period (columns are the periods, oldest first; money in millions), returns and margins by period, figures computed from the statements, and the long dividend and share record. No price, and no scores, ranks or verdicts of ours.

{company_figures}

---

### **OUTPUT FORMATTING INSTRUCTIONS**
Your output will be displayed directly on a website investor dashboard. You must:
1.  Use **Standard Markdown** formatting.
2.  Use `##` for main section headers and `###` for subsections.
3.  Use **bullet points** for readability.
4.  **Bold** key financial figures or critical red flags.
5.  Do NOT wrap the entire response in a code block.

---

### **PART 1: EVIDENCE & ANALYSIS**

## I. Business Model & Strategic Position
*Write a narrative analysis focusing on the durability of cash flows.*
* **The Core Engine:** How does the company actually make money?
* **Market Potential:** Does the company have the potential for a sizable increase in **unit or poundage volume sales** (real growth), or is it relying merely on price increases (inflation)?
* **Forward Outlook:** Will this company be generating more free cash flow in 5 years than today?

## II. Economic Moat Analysis (The Defense)
*Do not just list features. Evaluate if these factors act as a barrier to entry strong enough to justify a score > 3.0.*
1.  **Pricing Power:** Can they raise prices without losing volume? Is their product quality so strong that customers prefer it even at a premium?
2.  **Switching Costs:** Are products deeply embedded in customer operations? Is it painful/costly to switch?
3.  **Network Effects:** Does the value of the service increase as more users join?
4.  **Cost Advantage:** Do they have a structural lower cost basis (scale, location, unique assets) that competitors cannot replicate?
5.  **Engineering Leadership:** (If applicable) Is there evidence of "constant leadership in engineering" rather than just static patent protection?
6.  **Durability vs. Disruption:** specifically address threats from **AI disruption**. Does AI commoditize their product, or remove the customer's need for it? State whether the **AI substitution cap** in the rubric applies, and why.
7.  **What the figures cannot show:** brand and customer habit, licences and concessions, distribution access, culture. Name what holds the moat up that no statement records.

## II.B Revenue Dependency & Concentration (MANDATORY checklist)
*A moat is only as strong as the weakest party that can take the revenue away. Answer each item explicitly, marking it **known** (with the share or evidence), **believed** (from memory, possibly out of date) or **unknown**. "Unknown" is an honest answer; a guess presented as fact is not.*

1.  **Customer concentration:** the largest customer and the top customers as a % of revenue; contract length; can they leave, and how easily?
2.  **Platform or distribution dependency:** does revenue reach the end customer through a party the company does not control (a search engine, an app store, a marketplace, a single OEM or retailer, an ad network)? Who owns the customer relationship?
3.  **Product and segment mix:** one product or franchise, or several independent earners? Patent cliffs, single-drug or single-model exposure.
4.  **Supplier, licence or input dependency:** a single-source supplier, a revocable licence or concession, a key technology partner.
5.  **Government, regulatory or geographic concentration:** a single regulator, tender, programme or country.
6.  **Bargaining power:** for each concentration found, who needs whom more? Does the counterparty have an in-house or ready alternative, or a history of cutting or renegotiating terms?

State plainly whether the **dependency cap** in the rubric applies, and why.

## III. Management & Capital Allocation (The Offense)
*Management is graded on per-share value creation, not size.*


1.  **Capital Allocation & The "Buffett Test":** Has the company retained earnings only when it could create at least $1 of market value for every $1 retained? Or have they "diworsified" into terrible industries?
2.  **Forecasting Philosophy:** Does management avoid short-term guidance to focus on long-term value, or do they play the "earnings beat" game? Judge from what you know of their past targets against what the statements below show; where you cannot compare a target, say so rather than guess.
3.  **Skin in the Game:** Do insiders own significant stock bought with their own cash?
4.  **Integrity & Candor:** Read their shareholder letters. Do they admit mistakes and failures openly? Do they use "Adjusted EBITDA" to mask real costs?
5.  **What the figures cannot show:** key-person and founder dependence, succession, related-party dealings, controlling shareholders and how minority holders have been treated, governance history.

## IV. The Bear Case (Pre-Mortem)
*What breaks the thesis?*
* What is the single most likely reason this investment could result in a permanent loss of capital?
* What specific external event would invalidate your scores?

## IV.B Value Trap Pre-Mortem (MANDATORY checklist)
*A business that looks sound on its figures is most dangerous when the figures are about to stop holding. The mission is to avoid permanent capital loss; this checklist forces an explicit answer to each value-trap dimension. Every prompt must be answered — "no concern" is a valid answer when supported by evidence; an unanswered prompt is a failure of the analysis.*

1.  **Moat trajectory (5y look-back):** Is the moat measurably **narrower** than 5 years ago? Look for: market-share trend in core segments, gross-margin trend, whether the dependencies of Section II.B are growing or shrinking, switching-cost erosion (e.g., new substitute available), pricing-power evidence (price increases stuck or rolled back). State **widening / stable / narrowing** with one piece of concrete evidence.
2.  **Capital allocation drift:** Is management deploying retained earnings into the legacy core, into adjacent businesses with proven economics, or into **new low-ROIC adjacencies, vanity M&A, or "diworsification"**? List the last 3 material capital deployments and your judgment of each. A clean record is a positive signal; one suspect deployment is a yellow flag; two or more is a red flag.
3.  **Gross-margin compression from new competition:** Are gross margins compressing (vs. 5y median) due to new competitive entrants, channel shifts, or input-cost pass-through failure? Distinguish cyclical from structural compression — a structural reset is a moat-narrowing event, a cyclical reset is not. State the specific competitive threat by name where one exists.
4.  **Growth quality decomposition:** Of the last 5 years' revenue growth, what fraction came from (a) **price** (durable when supported by moat), (b) **volume** (cyclical / TAM-dependent), (c) **buybacks reducing the per-share denominator** (price-dependent and reverses when stock is no longer cheap)? A growth story dependent on (c) is a circularity flag — it works only while the stock stays undervalued.
5.  **Accounting aggressiveness signals:** Note any of the following with brief evidence: (i) revenue recognition that pulls forward future cash (long-dated subscriptions booked at signing), (ii) capitalization choices that move expenses to the balance sheet (R&D, customer-acquisition cost, software development), (iii) working-capital trends inconsistent with reported revenue growth (DSO climbing, inventory bloating, payables stretching), (iv) heavy use of "Adjusted" metrics that exclude recurring costs (SBC, restructuring, "one-time" items that recur every year), (v) auditor changes, restatements, or material weakness disclosures.

**Output requirement:** Each numbered prompt above must produce an explicit answer in the published report under a Section IV.B heading — not folded into narrative prose elsewhere. If a check returns clean, state "No concern: [one-line evidence]." If a check raises a flag, state the flag and assess whether the moat / management score should be revised downward as a result.

## V. Skeptical Scoring Re-evaluation
*This is the most important section for the user. Show your work.*
* "I initially considered a [X] for the Moat, but downgraded/upgraded to [Y] because..."
* Say whether the dependency cap applied, to which counterparty, and why (or why the mutual lock-in exception holds).
* Say whether the AI substitution cap applied, and why (or why AI strengthens rather than replaces the business).
* Explicitly weigh the pros and cons to finalize the number.

---

### **FINAL PART: VERDICT & SCORING**
*Write this section LAST, after every section above. The scores are the conclusion of the evidence; they are written after it so that the evidence is not written to fit them.*

**Economic Moat Score:** [Score]/5
**Management Quality Score:** [Score]/5

*(Note: Use increments of 0.5, e.g., 3.5, 4.0. Adhere strictly to the rubric provided at the end of this prompt.)*

**Summary Rationale:**
Provide a hard-hitting 3-4 sentence synthesis. Include an explanation of **why** it deserves these specific numbers. (e.g., "Score capped at 3.5 because despite high margins, they lack pricing power due to regulation.")

---

### **SCORING RUBRIC (STRICT ADHERENCE REQUIRED)**

**Statistical Distribution Mandate:**
You must adhere to a **Normal Distribution (Bell Curve)**. If you were to grade every company in the world, your scores must align with these probabilities. A "5" is statistically rare.

| Rating | Grade         | Target Probability |
| :----  | :------------ | :---------------- |
| **5.0**| Exceptional   | **Top 4.0%** |
| **4.5**| Excellent     | **Next 6.6%** |
| **4.0**| Very Good     | **Next 12.1%** |
| **3.5**| Good          | **Next 17.5%** |
| **3.0**| Average       | **Middle 19.7%** |
| **2.5**| Fair          | **Next 17.5%** |
| **2.0**| Below Avg     | **Next 12.1%** |
| **1.5**| Poor          | **Next 6.6%** |
| **1.0**| Very Poor     | **Bottom 4.0%** |

**Economic Moat Scoring:**
* **5.0 (Generational):** Monopolistic dominance, untouchable pricing power (e.g., Visa, Hermes).
* **4.0 (Wide Moat):** Durable advantage, high ROCE, hard to replicate (e.g., high-quality industrial).
* **3.0 (Average — competitive, no durable moat):** A good business that fights for every dollar. Average returns.
* **2.0 (Weak):** Commodity product, price-taker, low margins.
* **1.0 (Disadvantaged):** Obsolete business model or structural decline.

**Dependency cap (binding, for every company):**
* One counterparty (a customer, platform, distributor or licensor) behind roughly **30% or more of revenue**, or controlling access to most of it, that can switch or terminate: the **Moat Score is at most 3.0**, whatever the returns and margins show.
* The **Moat Score is at most 2.5** when that counterparty has an in-house or ready alternative, or has already cut or renegotiated terms.
* The only exception is a **mutual lock-in** argued with evidence: the counterparty would itself bear a large, specific cost to switch (a sole qualified supplier of a critical part, say).
* When the concentration is unknown to you, no cap applies, but Section II.B must say it is unknown.

**AI substitution cap (binding, for every company):**
* When AI can already do, or within about five years is likely to do, most of what customers pay the company for, or remove their need for it, at a fraction of the cost: the **Moat Score is at most 3.0**, whatever the returns and margins show.
* The **Moat Score is at most 2.5** when that substitution is already visible: customers moving to AI tools, prices cut to keep them, or a competitor built on AI winning share.
* Not a cap: AI that the company itself uses to lower its costs, or that strengthens what it owns (proprietary data, a regulated licence, a physical network, a brand customers trust). Argue which side the company is on.

**Management Quality Scoring:**
* **5.0 (Outsiders):** Exceptional capital allocators, high integrity, owner-operators (e.g., Buffett, Constellation Software).
* **4.0 (Shareholder Friendly):** Transparent, rational allocation, decent ownership.
* **3.0 (Standard Corporate):** Focuses on short-term guidance, standard stock options, "empire building" tendencies but competent operations.
* **2.0 (Red Flags):** Excessive dilution, confusing reporting, bad acquisitions.
* **1.0 (Value Destructive):** Fraud risk, lying to shareholders, self-enrichment.

---

**Final Instruction:**
Produce the report for **{company_name}** now. Ensure the "Economic Moat Score" and "Management Quality Score" lines appear exactly once, in the FINAL PART, as written above for parsing.