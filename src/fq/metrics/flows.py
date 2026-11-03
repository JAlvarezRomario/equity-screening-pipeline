"""Flows per period and what they pay for: FCF, ROIC, owner capex, acquisitions and financing."""

import numpy as np
import pandas as pd

from fq.metrics.capital import _common_book_and_tangible
from fq.metrics.common import _col, _dates, _depreciation, _frame, _liquidity, _slot, _spare_cash, _ttm_span
from fq.settings import Settings


def _period_metrics(df: pd.DataFrame, period: str, cfg: Settings) -> None:
    """Add per-period yields, margins and ROIC columns in place."""
    ev = df["enterpriseValue"].replace(0, np.nan)
    mc = df["marketCap"].replace(0, np.nan)
    group = df["Equity_Value"]  # consolidated flows / group = the shareholders' part / market cap
    c = lambda name: _col(df, f"{name}_{period}")  # noqa: E731

    # EBIT: operating income, else reported EBIT, else pretax income + interest.
    ebit = c("Operating Income").fillna(c("EBIT")).fillna(c("Pretax Income") + c("Interest Expense").fillna(0))
    df[f"EBIT_{period}"] = ebit
    sbc = c("Stock Based Compensation").fillna(0)

    df[f"EBIT_Yield_{period}"] = ebit * 100 / ev  # whole-company view, display only (Value is per share)
    # Losses, tax credits and one-offs give meaningless effective rates: use the notional rate.
    pretax = c("Pretax Income")
    tax_rate = c("Tax Provision") / pretax.where(pretax > 0)
    lo, hi = cfg.tax_rate_bounds
    tax_rate = tax_rate.where(tax_rate.between(lo, hi), cfg.notional_tax_rate)
    df[f"Tax_Rate_{period}"] = tax_rate
    # One-offs (disposal gains, impairments, mark-to-market gains) are no earning power: Value reads profit
    # without Yahoo's `Total Unusual Items` (pre-tax), else its `Normalized Income`. When Yahoo tags a gross
    # non-operating item (e.g. CME's clearinghouse collateral income) without its pass-through cost, cap the
    # subtracted gain so clean pretax profit never drops below operating income less interest.
    unusual = c("Total Unusual Items")
    op_inc = c("Operating Income")
    non_op_cap = (pretax - (op_inc - c("Interest Expense").abs().fillna(0))).clip(lower=0)
    unusual = unusual.where(df["Financial_Mode"] | ~(unusual > 0) | ~(op_inc > 0) | (unusual <= non_op_cap), non_op_cap)
    net_income = (c("Net Income") - unusual * (1 - tax_rate)).fillna(c("Normalized Income")).fillna(c("Net Income"))
    df[f"Net_Income_Clean_{period}"] = net_income
    pretax_clean = pretax - unusual.fillna(0)
    # SBC is a real cost. In financial and float mode operating cash flow mixes in loan, deposit or
    # reserve flows, so net income stands in for the cash available to shareholders.
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    fcf = c("Free Cash Flow") - sbc
    # Client-cash platforms (e.g. WISE.L) may book customer wallet inflows inside operating cash flow under IFRS:
    # cap FCF at clean net income + D&A - capex - SBC when OCF exceeds 2x (clean net income + D&A).
    da = c("Depreciation And Amortization").fillna(_depreciation(df, period)).fillna(0)
    client_cap = net_income + da - c("Capital Expenditure").abs().fillna(0) - sbc
    client_inflated = (
        (df["Business_Model"] == "client_cash")
        & (c("Operating Cash Flow") > 2.0 * (net_income + da))
        & (net_income > 0)
    )
    fcf = fcf.where(~client_inflated, np.fmin(fcf, client_cap))
    # Asset-heavy (REITs, utilities): only capex up to depreciation keeps the assets; the rest buys growth
    # (`Growth_Capex`, left out of earnings power).
    capex = c("Capital Expenditure").abs()
    maintenance = np.fmin(capex, cfg.maintenance_capex_to_da * _depreciation(df, period))
    heavy = (df["Business_Model"] == "asset_heavy") & maintenance.notna() & c("Operating Cash Flow").notna()
    df[f"Growth_Capex_{period}"] = (capex - maintenance).where(heavy)
    # IFRS: lease principal is a financing flow (`_leases`); interest paid may sit under financing too,
    # and then it is still in FCF but belongs to the lenders.
    financing = c("Lease_Cost").fillna(0) + c("Interest Paid Cff").abs().fillna(0)
    # All capex charged: Quality's cash tests (`_stability`) stay strict whatever the capex bought.
    df[f"FCF_Reported_{period}"] = (fcf - financing).where(~earnings_based, c("Net Income"))
    fcf = fcf.where(~heavy, c("Operating Cash Flow") - maintenance - sbc) - financing
    df[f"FCF_Adj_{period}"] = fcf.where(~earnings_based, net_income)
    df[f"FCFy_{period}"] = df[f"FCF_Adj_{period}"] * 100 / df["FCF_Base"]
    df[f"NIy_{period}"] = c("Net Income") * 100 / mc
    df[f"CAPEXy_{period}"] = c("Capital Expenditure") * 100 / mc
    df[f"OCFy_{period}"] = c("Operating Cash Flow") * 100 / group

    # Buybacks: share-count reduction when available, else cash buyback net of issuance and SBC.
    prev = cfg.latest if period == "TTM" else f"Y{_slot(period) + 1}"
    cur_sh, prev_sh = c("Ordinary Shares Number"), _col(df, f"Ordinary Shares Number_{prev}")
    if period == "TTM":
        # TTM ends a few months after Y0: compare with the count a year before the TTM end, read off the
        # line from Y1 to Y0, or without Y1 annualise the change since Y0 (cash flows are 12 months too).
        span = ((_dates(df, "TTM") - _dates(df, cfg.latest)).dt.days / 365.25).clip(0.25, 1.0)
        y1_sh = _col(df, f"Ordinary Shares Number_{cfg.prior}")
        year_ago = y1_sh + span * (prev_sh - y1_sh)
        annualised = cur_sh / (cur_sh / prev_sh.where(prev_sh > 0)) ** (1 / span)
        prev_sh = prev_sh.where(span.isna(), year_ago.where(y1_sh > 0, annualised))
    # Per share: the share of the company a holder gains from buybacks, plus dividends / market cap.
    by_shares = (prev_sh - cur_sh) / prev_sh.replace(0, np.nan) * 100
    by_cash = (-c("Repurchase Of Capital Stock").fillna(0) - c("Issuance Of Capital Stock").fillna(0) - sbc) * 100 / mc
    use_shares = cur_sh.notna() & prev_sh.notna() & (prev_sh > 0)
    buyback = np.maximum(0, np.where(use_shares, by_shares, by_cash))
    dividends = -c("Cash Dividends Paid").fillna(0) * 100 / mc
    df[f"DirectCashtoInvestors_Yield_{period}"] = dividends + buyback

    revenue = c("Total Revenue").replace(0, np.nan)
    df[f"profitMargins_{period}"] = c("Net Income") * 100 / revenue
    df[f"operatingMargins_{period}"] = c("Operating Income") * 100 / revenue
    assets = c("Total Assets")
    # On the assets the business uses: spare cash is valued at face in Value, so it must not dilute Quality too.
    # Not meaningful for lenders and insurers (assets are the loan book / investments).
    operating_assets = assets - _spare_cash(_liquidity(df, period).where(~df["Float_Mode"], 0.0), revenue, cfg)
    gross = c("Gross Profit") * 100 / operating_assets.where(operating_assets > 0)
    df[f"Gross_Profitability_{period}"] = gross.mask(df["Financial_Mode"])
    # Balance-sheet businesses: return on assets (blank for the others, so they never shape its distribution).
    df[f"ROA_{period}"] = (c("Net Income") * 100 / assets.where(assets > 0)).where(df["Financial_Mode"])
    # Pre-tax income per share / price, every model: after interest and without one-offs, the shareholders' part.
    df[f"Pretax_Yield_{period}"] = pretax_clean * 100 / group

    if f"Stockholders Equity_{period}" in df.columns:
        # All the capital owners and lenders left in the business, cash included (as ROIC.ai): cash not
        # invested is a choice the return should show; in Value cash counts through its interest (`_financing`).
        # Leases stay out (their cost is in EBIT, `_leases`). Missing debt / minority rows count as none reported.
        invested = (
            c("Stockholders Equity") + c("Financial_Debt").fillna(0) + c("Minority Interest").fillna(0)
        )  # fmt: skip
        # IFRS EBIT is before lease interest: with the leases out of capital, their interest leaves NOPAT too
        # (the lease liability at the company's own rate on its debt), as under US GAAP where rent is in EBIT.
        debt = c("Total Debt")
        leases = (debt - c("Financial_Debt")).clip(lower=0).where(df["Accounting_Standard"] == "ifrs", 0.0)
        rate = (c("Interest Expense").abs() / debt.where(debt > 0)).clip(0, 0.15).fillna(0)
        df[f"NOPAT_{period}"] = (ebit - (leases * rate).fillna(0)) * (1 - tax_rate)
        df[f"Invested_Capital_{period}"] = invested  # closing balance; `_roic` averages it with the opening one
        # What the business runs on, whatever financed it (`_roic` checks a capped return against it): plant
        # (net of the leases inside it), goodwill and non-cash working capital. Blank without plant.
        working = (
            c("Current Assets") - _liquidity(df, period)
            - (c("Current Liabilities") - c("Current Debt And Capital Lease Obligation").fillna(0))
        )  # fmt: skip
        df[f"Operating_Capital_{period}"] = (
            c("Net PPE") - (debt - c("Financial_Debt")).clip(lower=0).fillna(0)
            + c("Goodwill And Other Intangible Assets").fillna(0) + working.fillna(0)
        )  # fmt: skip

    # Book value and NAV (tangible book) per share as % of TODAY's price (share counts are split-adjusted).
    shares = c("Ordinary Shares Number").fillna(c("Share Issued"))
    if period in ("TTM", cfg.latest):
        shares = shares.fillna(df["Manual_Shares"])
    book, tangible = _common_book_and_tangible(c)
    per_price = lambda total: total / shares.where(shares > 0) / df["currentPrice"].where(df["currentPrice"] > 0) * 100  # noqa: E731
    df[f"BVPS_Pct_{period}"] = per_price(book)
    df[f"NAVPS_Pct_{period}"] = per_price(tangible)

    # No revenue means the period is unusable: blank every derived yield.
    no_rev = _col(df, f"Total Revenue_{period}", 0).fillna(0) == 0
    for name in ("DirectCashtoInvestors_Yield", "FCFy", "EBIT_Yield", "CAPEXy", "Gross_Profitability",
                 "ROA", "Pretax_Yield"):  # fmt: skip
        col = f"{name}_{period}"
        if col in df.columns:
            df[col] = df[col].mask(no_rev)


def _opening(df: pd.DataFrame, name: str, period: str, cfg: Settings) -> pd.Series:
    """`name` (a balance) at the start of `period`: the annual report a year earlier; for TTM the value a
    year before the TTM end, read off the line from Y1 to Y0 (Y0 itself without Y1)."""
    if period != "TTM":
        return _col(df, f"{name}_Y{_slot(period) + 1}")
    span = (_dates(df, "TTM") - _dates(df, cfg.latest)).dt.days / 365.25
    y0, y1 = _col(df, f"{name}_{cfg.latest}"), _col(df, f"{name}_{cfg.prior}")
    return (y1 + span.clip(0, 1) * (y0 - y1)).fillna(y0)


def _roic(df: pd.DataFrame, cfg: Settings) -> None:
    """`ROIC_<p>` = NOPAT / average invested capital (opening + closing / 2, else closing), as ROIC.ai.

    Profit is earned over the period on the capital in use during it: a year-end balance overstates the
    base of a fast grower. Profit on no capital is the best case (the cap), a loss on no capital the worst
    (a blank would be scored as unknown, not as bad). Financial mode: return on average equity.
    """
    for period in cfg.fiscal_years:
        if f"Invested_Capital_{period}" not in df.columns:
            continue

        def average(name: str, period: str = period) -> pd.Series:
            closing = _col(df, f"{name}_{period}")
            return ((closing + _opening(df, name, period, cfg)) / 2).fillna(closing)

        capital, nopat = average("Invested_Capital"), _col(df, f"NOPAT_{period}")
        # A return at the cap must come from a business that runs on little, not from equity wiped out by past
        # losses or write-offs (RR.L, 6666.HK): then the plant, goodwill and working capital it uses are the base.
        at_cap = (capital <= 0) | (nopat * 100 >= cfg.roic_cap * capital)
        operating = average("Operating_Capital")
        capital = capital.where(~(at_cap & (operating > capital)), operating)
        none = capital <= 0
        roic = (
            (nopat * 100 / capital.where(capital > 0))
            .mask(none & (nopat > 0), cfg.roic_cap)
            .mask(none & (nopat <= 0), -cfg.roic_cap)
        )
        equity, net_income = average("Stockholders Equity"), _col(df, f"Net Income_{period}")
        roe = (net_income * 100 / equity.where(equity > 0)).mask((equity <= 0) & net_income.notna(), -cfg.roic_cap)
        no_rev = _col(df, f"Total Revenue_{period}", 0).fillna(0) == 0
        df[f"ROIC_{period}"] = roic.where(~df["Financial_Mode"], roe).clip(upper=cfg.roic_cap).mask(no_rev)
        # The same without goodwill and intangibles (shown, a candidate): the operating moat apart from what was
        # paid for acquisitions; the gap to ROIC is M&A discipline. Operating companies only.
        bare = capital - average("Goodwill And Other Intangible Assets").fillna(0).clip(lower=0)
        ex_goodwill = (nopat * 100 / bare.where(bare > 0)).mask((bare <= 0) & (nopat > 0), cfg.roic_cap)
        ex_goodwill = ex_goodwill.mask((bare <= 0) & (nopat <= 0), -cfg.roic_cap).clip(upper=cfg.roic_cap)
        df[f"ROIC_Ex_Goodwill_{period}"] = ex_goodwill.mask(df["Financial_Mode"] | no_rev)


def _acquisitions(df: pd.DataFrame, cfg: Settings) -> None:
    """`Acquisition_Yield`: average yearly net cash paid for businesses, per share (% of market cap).

    FCF (operating cash flow - capex) leaves acquisitions out, though the revenue they buy shows as growth: shown
    (company page, a note when large), not scored. A cash-flow statement without the row means no acquisitions
    that year; a net seller counts as 0. Not applied in financial mode (FCF there is net income).
    """
    years = cfg.annual_years[: cfg.acquisition_years]
    if not any(f"Net Business Purchase And Sale_{p}" in df.columns for p in years):
        df["Acquisition_Yield"] = np.nan  # cache fetched before acquisitions were collected
        return
    spend = -_frame(df, "Net Business Purchase And Sale", years)
    spend = spend.where(spend.notna() | _frame(df, "Operating Cash Flow", years).isna(), 0.0)
    df["Acquisition_Yield"] = (spend.mean(axis=1).clip(lower=0) * 100 / df["Equity_Value"]).mask(df["Financial_Mode"])


def _us_inflation(df: pd.DataFrame, period: str, cfg: Settings) -> pd.Series:
    """US inflation (fraction) over the year `period` covers (`metrics.inflation`), `cfg.inflation` when its date is
    unknown: the price share of that year's USD revenue growth (8% in 2022 was price, not volume)."""
    return _col(df, f"Inflation_US_{period}").fillna(cfg.inflation)


def _organic_growth_capex(df: pd.DataFrame, cfg: Settings) -> None:
    """Owner earnings (Buffett, 1986): capex that grows the business is not a cost of keeping it.

    Greenwald's split for `cfg.organic_capex_models`: the growth a year needed = capital intensity
    (median net PPE / revenue) x the revenue it added in real terms (USD of the time, like `_growth`, so inflation
    in a falling currency is not growth, deflated by that year's US inflation, `_us_inflation`: price rises buy no
    volume, and replacing plant at today's prices is upkeep), never more than capex above depreciation (maintenance is at least depreciation). `Growth_Capex` is added back to `FCF_Adj`;
    `FCF_Reported` keeps all capex. Only growing (`Growth_Used` > 0),
    non-cyclical companies: an upswing's capacity is not lasting growth. TTM and the oldest year with
    revenue have no older year to compare with: TTM takes Y0's growth share of capex, the oldest year the
    share over the years measured (counting none would make its margin a step the normalisation reads).
    `Growth_Capex_Owner_<p>` keeps the split for every non-cyclical company of those models, growing or
    not: earnings power (`EPV_Yield`) charges maintenance capex only.
    """
    eligible = df["Business_Model"].isin(cfg.organic_capex_models) & ~df["Cyclical"]
    organic = eligible & (df["Growth_Used"] > 0)
    years = cfg.annual_years
    revenue = _frame(df, "Total Revenue", years)
    ppe = _frame(df, "Net PPE", years)
    intensity = (ppe / revenue.where(revenue > 0)).where(ppe >= 0).median(axis=1)
    usd = revenue * _frame(df, "FX_Hist", years).fillna(1.0)
    usd = usd.where(usd > 0)
    capex = _frame(df, "Capital Expenditure", years).abs()
    growth = pd.DataFrame(np.nan, index=df.index, columns=years)
    for new, old in zip(years[:-1], years[1:], strict=True):
        added = (usd[new] / usd[old] / (1 + _us_inflation(df, new, cfg)) - 1).clip(lower=0) * revenue[new]
        above = (capex[new] - _depreciation(df, new)).clip(lower=0)
        growth[new] = np.minimum(intensity * added, above).where(added.notna())
    measured = growth.notna() & capex.notna()
    share = growth.where(measured).sum(axis=1) / capex.where(measured).sum(axis=1).replace(0, np.nan)
    oldest = usd.notna() & usd.shift(-1, axis=1).isna()  # no older revenue to compare with
    growth = growth.mask(oldest, capex.mul(share, axis=0)).fillna(0.0)
    latest = growth[cfg.latest] / capex[cfg.latest].replace(0, np.nan)
    growth["TTM"] = (latest * _col(df, "Capital Expenditure_TTM").abs()).fillna(0.0)
    for p in cfg.fiscal_years:
        df[f"Growth_Capex_{p}"] = df[f"Growth_Capex_{p}"].mask(organic, growth[p])
        df[f"Growth_Capex_Owner_{p}"] = growth[p].where(eligible, 0.0)
    for period in cfg.fiscal_years:
        df[f"FCF_Adj_{period}"] += df[f"Growth_Capex_{period}"].where(organic, 0.0)
        df[f"FCFy_{period}"] = df[f"FCF_Adj_{period}"] * 100 / df["FCF_Base"]
        df[f"FCFy_{period}"] = df[f"FCFy_{period}"].mask(_col(df, f"Total Revenue_{period}", 0).fillna(0) == 0)
    df["Organic_Capex"] = organic


def _growth_working_capital(df: pd.DataFrame, cfg: Settings) -> None:
    """`Growth_WC_Owner_<p>`: the working capital real growth tied up (+) or freed (-), as growth capex for owners.

    Earnings power is at zero real growth (`_organic_growth_capex`), and operating cash flow holds the working
    capital each year's growth moved: a grower with receivables and stock reads poorer than its earnings power, one
    funded by suppliers or customers (a retailer paid before it pays, subscriptions, advances) richer, and a real
    shrinker's release is no earning power either. Greenwald's split: intensity (median non-cash working capital /
    revenue: current assets less cash, less current liabilities other than debt, so deferred revenue and advances
    count) x the revenue added beyond that year's US inflation (USD of the time, `_us_inflation`), signed, and never beyond what the
    cash-flow statement's `Change In Working Capital` moved that year in the same direction (no row: no
    adjustment). Inflation's share stays charged: it keeps volume, it does not add any. The oldest year and TTM
    take the ratio to revenue measured over the other years and Y0, as growth capex does. Same companies as
    `_organic_growth_capex` (cyclicals' working capital swings with the cycle, which normalisation already sees).
    Only earning power and the owner-earnings bars use it (`valuation._value`); FCF_Adj keeps the cash as it moved.
    """
    eligible = (
        df["Business_Model"].isin(cfg.organic_capex_models) & (df["Business_Model"] != "client_cash") & ~df["Cyclical"]
    )
    years = cfg.annual_years
    revenue = _frame(df, "Total Revenue", years)
    current_debt = _frame(df, "Current Debt And Capital Lease Obligation", years).fillna(0)
    operating_liabilities = _frame(df, "Current Liabilities", years) - current_debt
    non_cash = _frame(df, "Current Assets", years) - pd.DataFrame({p: _liquidity(df, p) for p in years})
    intensity = ((non_cash - operating_liabilities) / revenue.where(revenue > 0)).median(axis=1)
    usd = revenue * _frame(df, "FX_Hist", years).fillna(1.0)
    usd = usd.where(usd > 0)

    def capped(growth: pd.Series, period: str) -> pd.Series:
        moved = -_col(df, f"Change In Working Capital_{period}")  # + = cash tied up
        out = np.where(growth > 0, np.fmin(growth, moved.clip(lower=0)), np.fmax(growth, moved.clip(upper=0)))
        return pd.Series(out, index=df.index).where(moved.notna() & growth.notna())

    wc = pd.DataFrame(np.nan, index=df.index, columns=years)
    for new, old in zip(years[:-1], years[1:], strict=True):
        real = usd[new] / usd[old] / (1 + _us_inflation(df, new, cfg)) - 1
        wc[new] = capped(intensity * real * revenue[new], new)
    measured = wc.notna() & revenue.notna()
    share = wc.where(measured).sum(axis=1) / revenue.where(measured).sum(axis=1).replace(0, np.nan)
    oldest = usd.notna() & usd.shift(-1, axis=1).isna()  # no older revenue to compare with
    for p in years:
        wc[p] = wc[p].mask(oldest[p], capped(share * revenue[p], p))
    latest = wc[cfg.latest] / revenue[cfg.latest].where(revenue[cfg.latest] > 0)
    wc["TTM"] = capped(latest * _col(df, "Total Revenue_TTM"), "TTM")
    for p in cfg.fiscal_years:
        df[f"Growth_WC_Owner_{p}"] = wc[p].fillna(0.0).where(eligible, 0.0)


def _capital_retention(df: pd.DataFrame, cfg: Settings) -> None:
    """`Retention_Owner_<p>`: the earnings banks, insurers, holdings and health insurers keep to hold their balance
    sheet flat in real terms (common book x the local inflation: `cfg.inflation` compounded with the currency's
    inflation beyond US inflation, `Excess_Inflation_<p>`, `metrics.inflation`), the counterpart of the inflation
    share of working capital an operating company keeps charged (`_growth_working_capital`). A lira bank must keep
    about a third of its book a year to stay the same size; a franc bank less than a dollar one. Yahoo serves no
    risk-weighted assets, so book value measures the capital needed. Earnings power and the owner-earnings bars
    only. `Retention_Rate` (newest period, %) is shown in the owner-earnings note."""
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    for p in cfg.fiscal_years:
        book, _ = _common_book_and_tangible(lambda name, p=p: _col(df, f"{name}_{p}"))
        rate = ((1 + cfg.inflation) * (1 + _col(df, f"Excess_Inflation_{p}").fillna(0.0)) - 1).clip(lower=0)
        df[f"Retention_Owner_{p}"] = (rate * book.clip(lower=0)).fillna(0.0).where(earnings_based, 0.0)
        if p == "TTM":
            df["Retention_Rate"] = rate * 100


def _year_on_balance(df: pd.DataFrame, concept: str, balance, cfg: Settings) -> tuple[pd.Series, pd.Series]:
    """(a year's `concept` (TTM, else Y0) at the balance as it stands at that year's end, the yearly rate %).

    Interest over a year was paid on the average balance; today's cost is that rate on today's balance, so
    debt raised or repaid during the year counts in full. Without the older balance the flow stays as it is.
    """
    ttm, y0 = _col(df, f"{concept}_TTM").abs(), _col(df, f"{concept}_{cfg.latest}").abs()
    use_ttm = ttm.notna()
    flow = ttm.where(use_ttm, y0)
    span = _ttm_span(df, cfg).fillna(0.0)
    b_ttm, b0, b1 = balance("TTM").replace(0, np.nan), balance(cfg.latest), balance(cfg.prior)
    closing = b_ttm.where(use_ttm, b0).fillna(b0)
    opening = (b1 + span * (b0 - b1)).where(use_ttm, b1).fillna(b0)
    average = ((closing + opening) / 2).where(lambda a: a > 0)
    ratio = (closing / average).clip(0.25, 4.0).fillna(1.0)
    return flow * ratio, flow * 100 / average


def _financing(df: pd.DataFrame, cfg: Settings) -> None:
    """Flows before financing, and today's financing cost (operating, client cash, asset-heavy).

    `FCF_PreFin_<p>` = owner FCF before interest (paid and, where FCF holds it, received), so normalising
    it does not carry past capital structures or past rates; Value divides it by `Value_Base` (debt and cash
    at face, `capital._value_base`), which reads what was taken out here: `Debt_Rate` (interest added back /
    average financial debt, leases out) and `Cash_Interest_Out`. Display only: `Net_Financing_Yield` (after
    tax, % of `FCF_Base`), the interest on today's debt less the interest on today's cash, both at the last
    year's rates. Financial and float mode keep net income: their interest is the business.

    `Leverage_Adjustment` (pp, Modigliani-Miller, display only): a share bought with the company's debt behind
    it must earn more than one without, by (`asset_required_return` - after-tax rate) x net debt / equity
    value; net cash lowers the hurdle the same way (cash at its after-tax rate). Net debt / equity is clipped
    to [-1, 3]. Not ranked (user's choice, Oct 2026): leverage is judged in Quality (`NetDebt_to_EBIT`), and
    the adjustment magnified balance sheets the data misreads (captive lenders, client money as cash).
    """
    operating = ~(df["Financial_Mode"] | (df["Business_Model"] == "float"))
    tax = _col(df, "Tax_Rate_TTM").fillna(_col(df, f"Tax_Rate_{cfg.latest}")).fillna(cfg.notional_tax_rate)
    expense, debt_rate = _year_on_balance(df, "Interest Expense", lambda p: _col(df, f"Total Debt_{p}"), cfg)
    income, cash_rate = _year_on_balance(df, "Interest Income", lambda p: _liquidity(df, p), cfg)
    # Restated only where today's figure is known: a company that stopped reporting interest (AAPL since
    # 2024) would otherwise get its past interest back and pay none today. Client cash earns interest as its
    # business, so it stays in.
    restate_expense = operating & expense.notna()
    restate_income = operating & ~df["Float_Mode"] & income.notna()
    ifrs = df["Accounting_Standard"] == "ifrs"
    financial_share = {}
    for p in cfg.fiscal_years:
        # IFRS interest expense holds the leases' interest, but the leases left debt (`capital._leases`) and only
        # their principal is charged (`Lease_Cost`): only the borrowings' share of the interest goes back, as NOPAT
        # takes the leases' interest out (`_period_metrics`), else it is neither paid nor claimed (BOWL.L).
        debt = _col(df, f"Total Debt_{p}")
        share = (_col(df, f"Financial_Debt_{p}") / debt.where(debt > 0)).clip(0, 1)
        financial_share[p] = share.where(ifrs & _col(df, f"Lease_Cost_{p}").notna(), 1.0).fillna(1.0)
        paid = _col(df, f"Interest Expense_{p}").abs().fillna(0).where(restate_expense, 0.0) * financial_share[p]
        earned = _col(df, f"Interest Income_{p}").abs().fillna(0).where(restate_income, 0.0)
        # Interest paid is in FCF (operating, or `Interest Paid Cff`); interest received only when it is not
        # under investing (IFRS). Pre-tax income holds both.
        in_ocf = _col(df, f"Interest Received Cfi_{p}").fillna(0) == 0
        after_tax = 1 - _col(df, f"Tax_Rate_{p}").fillna(cfg.notional_tax_rate)
        df[f"FCF_PreFin_{p}"] = _col(df, f"FCF_Adj_{p}") + (paid - earned.where(in_ocf, 0.0)) * after_tax
        df[f"Pretax_PreFin_Yield_{p}"] = _col(df, f"Pretax_Yield_{p}") + (paid - earned) * 100 / df["Equity_Value"]
        # What the owner keeps: the same profit after the period's tax rate (bounded, `_period_metrics`), so a
        # high-tax country's pre-tax yield does not look as cheap as a low-tax one's.
        df[f"Aftertax_PreFin_Yield_{p}"] = df[f"Pretax_PreFin_Yield_{p}"] * after_tax
    net = (expense.where(restate_expense, 0.0) - income.where(restate_income, 0.0)).where(operating)
    df["Net_Financing_Yield"] = net * (1 - tax) * 100 / df["FCF_Base"]
    _, financial_rate = _year_on_balance(df, "Interest Expense", lambda p: _col(df, f"Financial_Debt_{p}"), cfg)
    now = financial_share["TTM"].where(_col(df, "Interest Expense_TTM").notna(), financial_share[cfg.latest])
    df["Debt_Rate"] = (financial_rate * now).where(restate_expense)
    df["Cash_Interest_Out"] = restate_income

    net_debt = df["Net_Debt"]
    rate = debt_rate.clip(1, 15).fillna(5.0).where(net_debt > 0, cash_rate.clip(0, 8).fillna(3.0))
    gearing = (net_debt / df["Equity_Value"]).clip(-1, 3)
    adjustment = (cfg.asset_required_return - rate * (1 - tax)) * gearing
    df["Leverage_Adjustment"] = adjustment.where(~df["Financial_Mode"])


def _dividend_yield(df: pd.DataFrame, cfg: Settings) -> None:
    """`Dividend_Yield` from cash actually paid (TTM, else the latest year): Yahoo's own field changed units in 2025.
    A cash-flow statement without a dividend row pays none."""
    paid = _col(df, "Cash Dividends Paid_TTM").fillna(_col(df, f"Cash Dividends Paid_{cfg.latest}"))
    has_cash_flow = _col(df, "Operating Cash Flow_TTM").notna() | _col(df, f"Operating Cash Flow_{cfg.latest}").notna()
    paid = paid.where(paid.notna() | ~has_cash_flow, 0.0)
    df["Dividend_Yield"] = (-paid).clip(lower=0) * 100 / df["marketCap"].where(df["marketCap"] > 0) + 0.0  # no -0.0
