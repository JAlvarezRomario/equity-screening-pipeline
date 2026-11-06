"""Flags that are not scores: suspect data and business models to review."""

import logging

import numpy as np
import pandas as pd

from fq.metrics.capital import EV_FLOOR_SHARE, _client_money_signs
from fq.metrics.common import _col, _dates, _distinct, _frame, _latest, _text, adjusted_r2, local_inflation
from fq.settings import Settings
from fq.universe import model_overrides

log = logging.getLogger(__name__)


def _data_suspect(df: pd.DataFrame, cfg: Settings) -> None:
    """Flag figures too good to be true or unreliable; they stay ranked but leave the headline tables."""
    stale = _text(df, "Stale_Annual") == True  # noqa: E712 - handles NaN / numpy bool
    checks = [
        (_col(df, "FCFy_EV_Norm") > cfg.suspect_fcf_yield, f"normalised FCF yield above {cfg.suspect_fcf_yield:.0f}%"),
        # What Value scores (earning power with its trend), not only the reported cash it starts from.
        (_col(df, "Owner_Yield") > cfg.suspect_fcf_yield, f"owner earnings yield above {cfg.suspect_fcf_yield:.0f}%"),
        (  # a high yield that rests mostly on add-backs (growth capex, working capital) rather than reported cash
            (_col(df, "EPV_Yield") > cfg.suspect_growth_yield)
            & (_col(df, "EPV_Yield") > 2 * _col(df, "FCFy_EV_Norm"))
            & (_col(df, "FCFy_EV_Norm") > 0),
            "earning power more than twice the reported FCF yield (add-backs worth checking)",
        ),
        (  # a lender's 'EBIT' includes its interest income and is not scored (pre-tax yield is)
            (_col(df, "Pretax_Yield_Avg") > cfg.suspect_pretax_yield) & ~df["Financial_Mode"],
            f"pre-tax yield above {cfg.suspect_pretax_yield:.0f}%",
        ),
        (  # unresolved only: when Yahoo's TTM was closer to the annual report, fetch already used it
            (_col(df, "TTM_Check_Gap") > cfg.suspect_ttm_gap) & (_text(df, "TTM_Source") == "quarters"),
            "quarterly sum and Yahoo's TTM disagree",
        ),
        (stale, "latest annual report is old"),
        (
            _text(df, "EV_Floored") == True,  # noqa: E712
            f"net cash near or above the market cap (value base floored at {EV_FLOOR_SHARE:.0%} of it)",
        ),
        (
            _text(df, "Debt_Cost_Hidden") == True,  # noqa: E712
            "most of the debt's cost is not reported as interest (captive lender, lessor? see the model review)",
        ),
        (
            (_col(df, "Growth_Used") >= cfg.growth_cap) & (_col(df, "FCFy_EV_Norm") > cfg.suspect_growth_yield),
            "growth at the cap with a high FCF yield (one-off?)",
        ),
    ]
    reasons: list[list[str]] = [[] for _ in range(len(df))]
    for mask, text in checks:
        for i in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            reasons[i].append(text)
    df["Suspect_Reasons"] = ["; ".join(r) or None for r in reasons]
    df["Data_Suspect"] = df["Suspect_Reasons"].notna()


def _peak_margin(df: pd.DataFrame, cfg: Settings) -> None:
    """Flag cyclicals earning far above their own weaker years: still scored, out of the top lists.

    The mid-cycle margin (lower of median and mean) cannot see a boom that fills most of the five years
    Yahoo serves (SRB.L: 3.7, 4.7, 35.9, 43.6% with the gold price). `Peak_Margin_Ratio` = latest operating
    margin (TTM, else Y0) / the `peak_margin_quantile` annual margin, at least `peak_margin_floor` points.
    """
    margin = pd.DataFrame(
        {
            p: _col(df, f"EBIT_{p}") * 100 / _col(df, f"Total Revenue_{p}").where(lambda r: r > 0)
            for p in cfg.fiscal_years
        },
        index=df.index,
    )
    annual = margin[cfg.annual_years]
    anchor = annual.quantile(cfg.peak_margin_quantile, axis=1).clip(lower=cfg.peak_margin_floor)
    enough = (annual.notna().sum(axis=1) >= 3) & ~df["Financial_Mode"]
    ratio = (margin["TTM"].fillna(margin[cfg.latest]) / anchor).where(enough)
    df["Peak_Margin_Ratio"] = ratio
    df["Peak_Margin"] = (_text(df, "Cyclical") == True) & (ratio > cfg.peak_margin_ratio)  # noqa: E712


def _distress(df: pd.DataFrame, cfg: Settings) -> None:
    """Flag operating companies that look close to failure: still scored, out of the top lists and Dislocation.

    Altman's Z'' (the non-manufacturer version: no sales / assets term, so heavy industry is not marked down)
    on the newest balance sheet, below `distress_z`, AND a cash test: negative reported FCF (newest period) or
    EBIT below `distress_interest_cover` x interest. The cash test keeps companies whose equity buybacks made
    negative (low Z'', healthy cash). Operating model only: other models' balance sheets are the business.
    """
    c = lambda name: _latest(df, name, cfg)  # noqa: E731
    assets = c("Total Assets").where(lambda a: a > 0)
    liabilities = c("Total Liabilities Net Minority Interest").where(lambda v: v > 0)
    z = (
        6.56 * (c("Current Assets") - c("Current Liabilities")) / assets
        + 3.26 * c("Retained Earnings") / assets
        + 6.72 * c("EBIT") / assets
        + 1.05 * c("Stockholders Equity") / liabilities
    )
    operating = df["Business_Model"] == "operating"
    df["Altman_Z"] = z.where(operating)
    cash_strain = (c("FCF_Reported") < 0) | (_col(df, "Interest_Coverage") < cfg.distress_interest_cover)
    df["Distress"] = (df["Altman_Z"] < cfg.distress_z) & cash_strain


def _revenue_decline(df: pd.DataFrame, cfg: Settings) -> None:
    """Flag a business whose sales shrink in real terms: still scored, out of the highlight lists.

    Real revenue: in the statements' currency at today's constant rate, net of that money's inflation
    (`common.local_inflation`: US inflation, `Inflation_US_Mean` over the trend and each year's own rate for a fall,
    compounded with `Excess_Inflation`; 0 for restated statements), so a falling currency or a price rise neither
    makes nor hides a decline. No exchange rate of the past (Nov 6, 2026: the yen's fall flagged ~200 Japanese
    companies growing in yen, a strong euro and pound hid real declines, and IAS 29 years already in today's money
    read as a collapse). Two ways to be clear: a log-linear trend over TTM and the
    annual reports at their real period ends (Y0 left out where TTM covers most of it, `_distinct`; at least
    `trend_min_points`) at or below `revenue_decline_rate` %/yr with an adjusted R² of at least
    `revenue_decline_min_r2` (noise around a flat line is not a decline), or a fall in every stored annual year over
    at least `revenue_decline_min_falls` steps averaging the same rate or worse (a flat line net of inflation is not
    clear either). Cyclicals only by the second (a fall from a peak is the cycle). `Revenue_Trend_Real` (%/yr) and `Revenue_Trend_R2` are shown. Not
    for banks, insurers and holdings (their revenue says little; book value per share is their growth).
    """
    periods = cfg.fiscal_years
    usd = _distinct(df, "Total Revenue", periods, cfg).where(lambda v: v > 0)
    y0 = _dates(df, cfg.latest)
    times = pd.DataFrame({p: (_dates(df, p) - y0).dt.days / 365.25 for p in periods}, index=df.index)
    us_mean = local_inflation(df, cfg)
    rate, fit = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    for i, (values, t) in enumerate(zip(usd.to_numpy(), times.to_numpy(), strict=True)):
        ok = np.isfinite(values) & np.isfinite(t)
        n = int(ok.sum())
        if n < cfg.trend_min_points or np.ptp(t[ok]) == 0:
            continue
        slope, intercept = np.polyfit(t[ok], np.log(values[ok]), 1)
        residual = np.log(values[ok]) - (slope * t[ok] + intercept)
        total = np.log(values[ok]) - np.log(values[ok]).mean()
        r2 = 1 - (residual @ residual) / (total @ total) if total @ total > 0 else 0.0
        rate.iloc[i] = (np.exp(slope) / (1 + us_mean.iloc[i]) - 1) * 100
        fit.iloc[i] = adjusted_r2(r2, n)
    years = cfg.annual_years  # newest first
    annual = _frame(df, "Total Revenue", years)
    pairs = list(zip(years[:-1], years[1:], strict=True))
    real = pd.DataFrame(  # each year's real change, log
        {new: np.log(annual[new] / annual[old] / (1 + local_inflation(df, cfg, new))) for new, old in pairs},
        index=df.index,
    ).where(lambda v: np.isfinite(v))
    n_steps = real.notna().sum(axis=1)
    mean_fall = (np.exp(real.mean(axis=1)) - 1) * 100  # the average step, so a flat line deflated is not enough
    every_fall = (n_steps >= cfg.revenue_decline_min_falls) & real.lt(0).sum(axis=1).eq(n_steps)
    every_fall &= mean_fall <= cfg.revenue_decline_rate
    df["Revenue_Trend_Real"] = rate
    df["Revenue_Trend_R2"] = fit
    # A cyclical's sales fall from a peak (refiners, chemicals after 2022): the trend is the cycle, so it counts only
    # when every year fell, as the owner trend does (`valuation._owner_yield`).
    trend = (rate <= cfg.revenue_decline_rate) & (fit >= cfg.revenue_decline_min_r2) & (_text(df, "Cyclical") != True)  # noqa: E712
    df["Revenue_Decline"] = (trend | every_fall) & ~df["Financial_Mode"]


def _accounting_flag(df: pd.DataFrame, cfg: Settings) -> None:
    """Flag earnings that run ahead of cash: still scored, out of the highlight lists (`ranking.UNTRUSTED_FLAGS`).

    Median accruals ((net income - operating cash flow) / assets, `quality.Accruals`) above `accounting_max_accruals`,
    or Beneish's M-score above `accounting_max_beneish` with accruals above `accounting_beneish_accruals` (Beneish
    alone marks too many honest growers; Oct 30, 2026: 71 of 4,753 trusted companies at 10%, 48 more by both). Missing
    figures pass (missing is not bad). `Accounting_Reasons` names the test.
    """
    accruals, m = _col(df, "Accruals"), _col(df, "Beneish_M")
    high = accruals > cfg.accounting_max_accruals
    both = (m > cfg.accounting_max_beneish) & (accruals > cfg.accounting_beneish_accruals)
    reasons = np.where(
        high,
        f"accruals above {cfg.accounting_max_accruals:.0f}% of assets (earnings ahead of cash)",
        np.where(both, f"Beneish M-score above {cfg.accounting_max_beneish} with accruals above "
                 f"{cfg.accounting_beneish_accruals:.0f}% of assets", None),
    )  # fmt: skip
    df["Accounting_Reasons"] = pd.Series(reasons, index=df.index, dtype=object)
    df["Accounting_Flag"] = (high | both).fillna(False).astype(bool)


def _model_review(df: pd.DataFrame, cfg: Settings) -> None:
    """`Model_Review`: signs in a company's own statements that another business model may fit better than
    the one Yahoo's industry gave it. Reported only (log and company page): the model changes only when the
    user adds the company to `lists/model_overrides.txt`. Companies already there are not flagged.

    - client_cash: cash over 45% of assets and above equity + financial debt together (more cash than owners
      and lenders provided: client money, PDD, UPWK);
    - holding: in most profitable years operating income under half of pre-tax income and the rest over 3%
      of assets (profit from stakes, H02.SI; interest on a loss-maker's cash stays below that);
    - bank, insurer or client_cash: revenue under 10% of assets with assets over 8x equity (a balance sheet that
      is the business: a lender, a life insurer such as LGEN, or a clearing house such as LSEG's LCH);
    - holding (investment trust): book value 70-150% of the price, equity over 75% of assets and fixed assets
      under 5% of them (a fund priced near its net assets: SMT, FCIT, 3i, Pershing Square; Power Assets);
    - asset_heavy: fixed assets over 60% of assets with revenue under 25% of them (REIT / utility-like);
    - bank or asset_heavy: debt whose cost is not reported as interest exceeds the group's market value
      (`Debt_Cost_Hidden`, `_value_base`: a captive lender, a lessor such as AER);
    - insurer: an operating company that reports policyholder benefits and claims (MNG, LGEN filed as asset
      managers);
    - operating or holding: a bank only by Yahoo's industry whose balance sheet is not a lender's, assets under 3x
      equity (a fee business, a crypto treasury, or an equity-funded lender?).
    Until the user confirms a model, the company stays out of the highlight lists (`ranking.trusted`).
    """
    assets = _latest(df, "Total Assets", cfg).where(lambda s: s > 0)
    equity = _latest(df, "Stockholders Equity", cfg)
    turnover = _col(df, "Total Revenue_TTM").fillna(_col(df, f"Total Revenue_{cfg.latest}")) / assets
    years = cfg.annual_years
    pretax, operating = _frame(df, "Pretax Income", years), _frame(df, "Operating Income", years)
    total_assets = _frame(df, "Total Assets", years)
    outside = ((operating < 0.5 * pretax) & (pretax - operating > 0.03 * total_assets)).where(pretax > 0)
    model, financial = df["Business_Model"], df["Financial_Mode"]
    client_cash, balance_business = _client_money_signs(df, cfg)
    signs = [
        (client_cash, "client_cash", "cash is over 45% of assets and more than equity and debt together (client money?)"),
        ((outside.sum(axis=1) > outside.count(axis=1) / 2) & (outside.count(axis=1) >= 2) & ~financial,
         "holding", "operating income is under half of pre-tax income in most years (profit from stakes?)"),
        (balance_business, "bank, insurer or client_cash", "revenue is under 10% of assets with assets over 8x "
         "equity (lender, insurer or clearing?)"),
        (df["Debt_Cost_Hidden"], "bank or asset_heavy", "debt above the market value with most of its cost outside "
         "interest expense (captive lender, lessor or consolidated funds?)"),
        ((_latest(df, "Net Policyholder Benefits And Claims", cfg).abs() > 0) & model.isin(("operating", "client_cash")),
         "insurer", "reports policyholder benefits and claims (an insurer filed elsewhere?)"),
        (df["BV_to_Price"].between(70, 150) & (equity > 0.75 * assets) & (_latest(df, "Net PPE", cfg).fillna(0) < 0.05 * assets)
         & ~financial, "holding", "priced near book value, mostly equity-funded, almost no fixed assets "
         "(an investment trust or holding?)"),
        ((_latest(df, "Net PPE", cfg) > 0.6 * assets) & (turnover < 0.25) & model.isin(("operating", "client_cash", "float")),
         "asset_heavy", "fixed assets are over 60% of assets with revenue under 25% of them (REIT / utility-like?)"),
        # No deposit lender is funded a third or more by equity, whatever its revenue (PURR, a crypto treasury with
        # revenue under 1% of assets, read Quality #1 through the bank restatement in `quality._return_quality`).
        ((model == "bank") & (df.get("Archetype_Source") == "industry") & (assets < 3 * equity.where(equity > 0)),
         "operating or holding", "a bank by industry only: assets under 3x equity (a fee business, a crypto "
         "treasury, or an equity-funded lender?)"),
    ]  # fmt: skip
    overridden = df["Ticker"].str.upper().isin(set(model_overrides()))
    notes: list[list[str]] = [[] for _ in range(len(df))]
    for mask, suggested, why in signs:
        for i in np.flatnonzero((mask.fillna(False) & ~overridden).to_numpy(dtype=bool)):
            notes[i].append(f"{suggested}: {why}")
    df["Model_Review"] = ["; ".join(n) or None for n in notes]
    flagged = df.loc[df["Model_Review"].notna(), "Ticker"]
    if len(flagged):
        log.info("Business model to review for %d companies (see Model_Review; confirm in model_overrides.txt): %s",
                 len(flagged), ", ".join(flagged))  # fmt: skip
