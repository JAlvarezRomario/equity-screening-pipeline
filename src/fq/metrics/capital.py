"""The capital side: leases, EV, book value, the shareholders' base and the claims Value counts at face."""

import numpy as np
import pandas as pd

from fq.metrics.common import _col, _first_valid, _frame, _latest, _liquidity, _spare_cash
from fq.settings import Settings
from fq.universe import accounting_standard, model_overrides

EV_FLOOR_SHARE = 0.1  # EV never below this share of market cap (net cash near or above the price)


def _leases(df: pd.DataFrame, cfg: Settings) -> None:
    """Debt without lease liabilities (`Financial_Debt_<p>`) and, under IFRS, a year's lease cash (`Lease_Cost_<p>`).

    Yahoo's Total Debt holds every lease liability (ITX.MC: all of it). Under US GAAP rent is already inside
    operating income and operating cash flow, so the liability as debt charged leases twice (retailers,
    airlines). Under IFRS the principal is a financing flow that FCF leaves out: the liability due within
    12 months stands in for a year's lease cash. Leases leave debt only when Total Debt holds them
    (debt >= leases) and, under IFRS, when that current part is known (else they stay debt, charged once).
    US-GAAP finance leases (principal in financing) are treated as operating: small for most companies.
    `Lease_Adjusted` describes the period whose debt enters EV.
    """
    df["Accounting_Standard"] = df["Ticker"].map(accounting_standard)
    ifrs = df["Accounting_Standard"] == "ifrs"

    def leases(p: str) -> tuple[pd.Series, pd.Series]:
        lease = _col(df, f"Capital Lease Obligations_{p}").where(lambda s: s > 0)
        return lease, _col(df, f"Current Capital Lease Obligation_{p}").where(lambda s: s >= 0)

    # Many companies break leases out only in the annual report. A TTM quarter without the lease row
    # takes Y0's leases (and, under IFRS, Y0's current share of them), else its leases would stay debt
    # in EV (which reads TTM first) while the annual FCF behind the normalised yield paid the lease cash.
    lease_y0, current_y0 = leases(cfg.latest)
    current_share = current_y0 / lease_y0
    used = {}
    for p in cfg.fiscal_years:
        debt = _col(df, f"Total Debt_{p}")
        lease, current = leases(p)
        if p == "TTM":
            lease = lease.where(_col(df, f"Capital Lease Obligations_{p}").notna(), lease_y0)  # 0 reported stays 0
            current = current.fillna(lease * current_share)
        out = lease.notna() & (debt >= lease) & (~ifrs | current.notna())
        financial = debt - lease.where(out, 0.0)
        if p == "TTM":
            # A quarter whose Total Debt is only its leases while the annual report had real borrowings did not
            # report the borrowings (TEF.MC, HOT.DE, BNR.DE; TEF: 8.9bn = leases, against 36bn of financial debt in
            # Y0): unknown, not none, so the latest annual borrowings stand in, in Total Debt too, so EV, today's
            # interest cost and the debt counted at face all see them.
            y0 = _col(df, f"Total Debt_{cfg.latest}") - lease_y0.fillna(0)
            missing = out & (financial < 0.05 * y0) & (y0 > 0)
            financial = financial.mask(missing, y0)
            df[f"Total Debt_{p}"] = debt.mask(missing, y0 + lease)
        df[f"Financial_Debt_{p}"] = financial
        df[f"Lease_Cost_{p}"] = current.where(ifrs & out)  # charged once: as debt or as cash, never both
        used[p] = out.where(debt.notna())
    # Reported for the period whose debt enters EV (`_rebuild_enterprise_value`: TTM, else Y0).
    df["Lease_Adjusted"] = used["TTM"].fillna(used[cfg.latest]).fillna(False).astype(bool)


def _client_money_signs(df: pd.DataFrame, cfg: Settings) -> tuple[pd.Series, pd.Series]:
    """(cash over 45% of assets and above equity + financial debt, revenue under 10% of assets with assets over
    8x equity): balance sheets whose cash may belong to clients or policyholders (`_model_review`), so Value
    does not count it as spare (`_rebuild_enterprise_value`). Only operating, non-financial rows."""
    assets = _latest(df, "Total Assets", cfg).where(lambda s: s > 0)
    equity = _latest(df, "Stockholders Equity", cfg)
    revenue = _col(df, "Total Revenue_TTM").fillna(_col(df, f"Total Revenue_{cfg.latest}"))
    liquidity = df["Manual_Liquidity"]
    client_cash = ((liquidity > 0.45 * assets) & (df["Business_Model"] == "operating")
                   & (liquidity > equity.clip(lower=0) + _latest(df, "Financial_Debt", cfg).fillna(0)))  # fmt: skip
    balance = (revenue / assets < 0.10) & (assets > 8 * equity.where(equity > 0)) & ~df["Financial_Mode"]
    return client_cash.fillna(False), balance.fillna(False)


def _rebuild_enterprise_value(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    pref = ["TTM", cfg.latest, cfg.prior]
    shares = _latest(df, "Ordinary Shares Number", cfg).fillna(_latest(df, "Share Issued", cfg))
    for key in ("sharesOutstanding", "impliedSharesOutstanding"):  # Yahoo summary data as last resort
        if key in df.columns:
            shares = shares.fillna(pd.to_numeric(df[key], errors="coerce"))
    df["Manual_Shares"] = shares
    manual_cap = df["currentPrice"] * shares
    df["Effective_MarketCap"] = manual_cap.where(manual_cap.notna() & (manual_cap != 0), df["marketCap"])

    debt = _first_valid(df, [f"Financial_Debt_{p}" for p in pref[:2]]).fillna(_col(df, f"Financial_Debt_{cfg.prior}"))
    equity = _latest(df, "Stockholders Equity", cfg)
    # A missing debt row means no debt reported (e.g. Nintendo), same convention as for EV below.
    df["Calculated_Debt_to_Equity"] = debt.fillna(0) / equity.replace(0, np.nan) * 100

    df["Manual_Liquidity"] = (
        _liquidity(df, "TTM")
        .replace(0, np.nan)
        .fillna(_liquidity(df, cfg.latest))
        .fillna(_liquidity(df, cfg.prior))
        .fillna(0)
    )
    minority = (
        _first_valid(df, [f"Minority Interest_{p}" for p in pref[:2]])
        .fillna(_col(df, f"Minority Interest_{cfg.prior}", 0))
        .fillna(0)  # most companies have no minority-interest row at all
    )
    # Preferred shares rank ahead of common and their dividends are paid out of the earnings and cash flows
    # yielded on EV (they are also inside Stockholders Equity, the ROIC capital base).
    # Both names per period, newest first: an older period's preferred may since have converted (COHR).
    preferred = _first_valid(
        df, [f"{name}_{p}" for p in pref for name in ("Preferred Stock Equity", "Preferred Stock")]
    )
    preferred = preferred.fillna(0)
    # Float mode (health insurers): the cash backs policyholder claims, so it is not spare.
    liquidity = df["Manual_Liquidity"].where(~df["Float_Mode"], 0.0)
    df["Net_Debt"] = debt.fillna(0) + minority + preferred - liquidity
    # Value (`_shareholder_base`): debt and preferred at face, less only the cash the business does not need.
    # Cash that may be clients' (`_client_money_signs`: HKEX margin deposits) is not counted until
    # `model_overrides.txt` says what the company is.
    spare = _spare_cash(liquidity, _first_valid(df, [f"Total Revenue_{p}" for p in pref[:2]]), cfg)
    doubtful = np.logical_or(*_client_money_signs(df, cfg)) & ~df["Ticker"].str.upper().isin(set(model_overrides()))
    df["Spare_Cash_Doubtful"] = doubtful & (spare > 0)
    df["Excess_Cash"] = spare.mask(doubtful, 0.0)
    df["Value_Debt"], df["Value_Preferred"] = debt.fillna(0), preferred  # at face where their cost left the flows
    df["Manual_EV"] = df["Effective_MarketCap"] + df["Net_Debt"]
    df["marketCap"] = df["Effective_MarketCap"]
    # Net cash close to or above the market cap: a floor instead of falling back to the market cap, which
    # would make the cheapest companies look expensive (`EV_Floored`, set with Value's floor in `_value_base`).
    df["enterpriseValue"] = np.fmax(df["Manual_EV"], df["marketCap"] * EV_FLOOR_SHARE).fillna(df["marketCap"])
    # Financial mode: debt and cash are the business itself (deposits, float), not claims on it.
    # Minorities and preferred still claim part of the consolidated earnings (holding companies such as Brookfield).
    fin = df["Financial_Mode"]
    df.loc[fin, "enterpriseValue"] = df.loc[fin, "marketCap"] + minority[fin] + preferred[fin]
    df.loc[fin, "Net_Debt"] = np.nan
    df.loc[fin, ["Excess_Cash", "Value_Debt", "Value_Preferred"]] = np.nan
    return df


def _common_book_and_tangible(c) -> tuple[pd.Series, pd.Series]:
    """Book value and minority-adjusted tangible book that belong to common shareholders.

    Book = common equity (preferred shares belong to preferred holders). For tangible book only
    the parent's share of goodwill/intangibles is deducted: consolidated intangibles sit partly in
    subsidiaries owned by minority holders (Yahoo deducts 100%, which makes holding companies such
    as Brookfield look negative). Without minority interests this equals Yahoo's tangible book.
    """
    parent = c("Stockholders Equity")
    common = c("Common Stock Equity").fillna(parent - c("Preferred Stock Equity").fillna(0))
    total = c("Total Equity Gross Minority Interest").fillna(parent + c("Minority Interest").fillna(0))
    parent_share = (parent / total.where(total > 0)).clip(0, 1).fillna(1.0)
    # Yahoo's tangible book = common equity - all intangibles, so it also reveals intangibles.
    intangibles = c("Goodwill And Other Intangible Assets").fillna(common - c("Tangible Book Value"))
    return common, common - parent_share * intangibles


def _shareholder_base(df: pd.DataFrame, cfg: Settings) -> None:
    """Denominators of the per-share yields (user's rule: Value is per share).

    `Parent_Share`: the parent's part of consolidated profit (net income / net income including minority
    interests, median of the years both are profits); minority holders own the rest of every consolidated
    flow (cash flow, pre-tax income). Book equity is no guide: buybacks shrink it (Colgate's minorities get
    ~6% of profit but would read ~40% of its book). `Equity_Value` = market cap / parent share: the whole
    group at the share price, so a consolidated flow over it is the shareholders' part over market cap.
    Net income is already the parent's (`FCF_Base` = market cap where FCF is net income). Unknown = 1.
    """
    years = cfg.fiscal_years
    parent = _frame(df, "Net Income", years)
    total = _frame(df, "Net Income Including Noncontrolling Interests", years)
    share = (parent / total).where((parent > 0) & (total > 0)).clip(upper=1.0).median(axis=1)
    df["Parent_Share"] = share.fillna(1.0)
    mc = df["marketCap"].where(df["marketCap"] > 0)
    df["Equity_Value"] = mc / df["Parent_Share"]
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    df["FCF_Base"] = df["Equity_Value"].where(~earnings_based, mc)


def _value_base(df: pd.DataFrame, cfg: Settings) -> None:
    """`Value_Base`, what Value's yields divide by (Greenwald: debt and cash at face; `_to_value_base`).

    The group at the share price plus debt and preferred, less spare cash (`Net_Claims`), so a flow before
    interest over it is the shareholders' part over their share of the price of the operations. Each claim
    counts once: at face only as far as its cost or income was taken out of the flows (`_financing`: `Debt_Rate`,
    `Cash_Interest_Out`), else through the flows as reported.
    - Debt: `Debt_At_Face` = debt x min(1, its implied interest rate / `debt_cost_floor`). The median company
      pays ~4%; a rate far below it means most of the debt's cost is elsewhere (a captive lender's or a
      lessor's in cost of sales, consolidated funds' inside fund results: BMW 0.6%, MBG 0.4%, BEN 0.6%), and
      no interest row (AER) means it is all still in the flows.
    - Cash: `Cash_At_Face` = spare cash (`Excess_Cash`) when its interest income was taken out, else none.
    Net cash near or above the group value floors the base at `EV_FLOOR_SHARE` of it (`EV_Floored`, also set
    when the display EV was floored). Financial and float mode keep `FCF_Base`: their debt and cash are the
    business.
    """
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    rate = df["Debt_Rate"]
    df["Debt_At_Face"] = df["Value_Debt"] * (rate / cfg.debt_cost_floor).clip(0, 1).fillna(0)
    df["Cash_At_Face"] = df["Excess_Cash"].where(df["Cash_Interest_Out"], 0.0)
    df["Net_Claims"] = (df["Debt_At_Face"] + df["Value_Preferred"] - df["Cash_At_Face"]).where(~earnings_based)
    # More debt with its cost inside operating costs than the group is worth: the debt is the business (a captive
    # lender, a lessor, consolidated funds), so the yields are a lender's, levered (`Model_Review`, suspect).
    hidden = df["Value_Debt"] - df["Debt_At_Face"]
    overridden = df["Ticker"].str.upper().isin(set(model_overrides()))
    operating = df["Business_Model"].isin(("operating", "client_cash"))  # asset-heavy debt is plant, scored apart
    df["Debt_Cost_Hidden"] = (hidden > df["Equity_Value"]) & operating & ~overridden
    at_face = df["Equity_Value"] + df["Net_Claims"].fillna(0)
    floor = df["Equity_Value"] * EV_FLOOR_SHARE
    ev_floored = (df["Manual_EV"] < df["marketCap"] * EV_FLOOR_SHARE) & ~df["Financial_Mode"]
    df["EV_Floored"] = ev_floored | ((at_face < floor) & ~earnings_based)
    df["Value_Base"] = np.fmax(at_face, floor).where(~earnings_based, df["FCF_Base"])


def _to_value_base(df: pd.DataFrame, yields: pd.Series) -> pd.Series:
    """A yield on `Equity_Value` (consolidated flow / group) as one on `Value_Base`; unchanged where they are the
    same (financial and float mode, `FCF_Base`)."""
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    return yields * (df["Equity_Value"] / df["Value_Base"]).where(~earnings_based, 1.0)


def _book_ratios(df: pd.DataFrame, cfg: Settings) -> None:
    """Book value, tangible book and net current asset value (`NCAV_to_Price`, Graham's net-net test; not for banks,
    insurers and holdings, whose current assets mean nothing) per share as % of today's price (newest period),
    shareholder net cash / debt as % of market cap (`Net_Cash_to_Cap`), enterprise-delevered tangible NAV
    (`Effective_NAV_to_Price`), and whether the NAV chart shows."""
    for name, prefix in (("NAV_to_Price", "NAVPS_Pct"), ("BV_to_Price", "BVPS_Pct")):
        df[name] = _latest(df, prefix, cfg)
    df["Show_NAV_Chart"] = df[["NAV_to_Price", "BV_to_Price"]].max(axis=1) >= cfg.nav_chart_threshold_pct
    fin = df["Financial_Mode"] == True  # noqa: E712
    df["NCAV_to_Price"] = _latest(df, "NCAVPS_Pct", cfg).where(~fin)
    # Shareholder-basis spare cash minus financial debt and preferred stock (`Minority Interest` is already out of
    # common equity, and `Excess_Cash` leaves out operating till-cash and client/float money).
    fin_net_debt = df["Value_Debt"].fillna(0) + df["Value_Preferred"].fillna(0) - df["Excess_Cash"].fillna(0)
    eq_val = df["Equity_Value"].where(df["Equity_Value"] > 0)
    df["Net_Cash_to_Cap"] = (-fin_net_debt * 100.0 / eq_val).where(~fin)
    # Primary NAV proxy is minority-adjusted tangible book (`NAV_to_Price`), falling back to common book only when no
    # intangibles or tangible book row is reported. When in net debt (d = -Net_Cash_to_Cap > 0), de-lever to the
    # enterprise level so senior debt cannot artificially inflate NAV/Price above 100%.
    raw_nav = df["NAV_to_Price"].fillna(df["BV_to_Price"])
    net_debt_pct = (-df["Net_Cash_to_Cap"]).clip(lower=0).fillna(0)
    unlevered_nav = (raw_nav + net_debt_pct) / (100.0 + net_debt_pct) * 100.0
    eff_nav = np.where(fin | (df["Net_Cash_to_Cap"] >= 0), raw_nav, np.fmin(raw_nav, unlevered_nav))
    df["Effective_NAV_to_Price"] = pd.Series(eff_nav, index=df.index, dtype=float).where(raw_nav.notna())


def _unrealized(df: pd.DataFrame, period: str) -> pd.Series:
    """Unrealised gains (+) / losses (-) on securities held in equity: IFRS `Unrealized Gain Loss`, else, under
    US GAAP, the whole AOCI (mostly securities and hedges for a lender). IFRS AOCI is left alone: it also holds
    currency translation (BBVA: -20bn), a real loss for a group in falling currencies."""
    us = df["Ticker"].map(accounting_standard) != "ifrs"
    aoci = _col(df, f"Gains Losses Not Affecting Retained Earnings_{period}").where(us)
    return _col(df, f"Unrealized Gain Loss_{period}").fillna(aoci).fillna(0)


def _book_per_share(df: pd.DataFrame, period: str) -> pd.Series:
    """Common equity per share in the reporting currency (blank when either is missing or <= 0), without
    unrealised securities gains and losses: a bond book marked down in 2022 comes back as it matures, so book
    growth and compounding measure what was earned and kept, not where rates moved."""
    c = lambda name: _col(df, f"{name}_{period}")  # noqa: E731
    common = c("Common Stock Equity").fillna(c("Stockholders Equity") - c("Preferred Stock Equity").fillna(0))
    unrealized = _unrealized(df, period)
    # More than half of equity is no bond mark-down but a misfiled row (STNE: 13.7bn on 14.6bn): left alone.
    common = common - unrealized.where(unrealized.abs() <= 0.5 * common.abs(), 0.0)
    shares = c("Ordinary Shares Number").fillna(c("Share Issued"))
    return (common / shares.where(shares > 0)).where(common > 0)
