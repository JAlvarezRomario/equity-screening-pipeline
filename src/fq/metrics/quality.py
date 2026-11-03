"""Quality inputs: stability, leverage, capital allocation, cyclicality and recent change."""

import numpy as np
import pandas as pd

from fq.metrics.capital import _book_per_share
from fq.metrics.common import (
    TTM_SHARED_SPAN,
    _col,
    _dates,
    _depreciation,
    _distinct,
    _frame,
    _latest,
    _log_slope_growth,
    _slot,
    _text,
    _ttm_span,
    adjusted_r2,
)
from fq.settings import Settings
from fq.universe import is_cyclical_industry


def _annual_share_reduction(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Average yearly % reduction in share count, from the oldest period with data to the newest.

    The span is measured from the stored period-end dates (slot distance if dates are missing),
    so it does not matter how many years Yahoo currently serves.
    """
    ttm = _col(df, "Ordinary Shares Number_TTM")
    end = ttm.fillna(_col(df, f"Ordinary Shares Number_{cfg.latest}"))
    end_date = _dates(df, "TTM").where(ttm.notna(), _dates(df, cfg.latest))
    start = pd.Series(np.nan, index=df.index)
    span = pd.Series(np.nan, index=df.index)
    for period in cfg.annual_years[1:]:  # oldest last, so it wins
        shares = _col(df, f"Ordinary Shares Number_{period}")
        elapsed = (end_date - _dates(df, period)).dt.days / 365.25
        elapsed = elapsed.fillna(float(_slot(period)))
        start = shares.where(shares > 0, start)
        span = span.where(~(shares > 0), elapsed)
    ratio = end / start
    rate = (1 - ratio.pow(1 / span.where(span > 0))) * 100
    return rate.where(np.isfinite(rate))


def _leverage(df: pd.DataFrame, cfg: Settings) -> None:
    """Net debt / EBIT (median EBIT of TTM and the last three years) and interest coverage.

    Unlike debt / equity it stays meaningful when buybacks have made book equity negative. Net cash
    counts as none (0, best): a bigger idle pile is no better business, and ROIC already counts the cash
    as capital. Net debt without operating profit gets 99 (worst).
    """
    ebit = _distinct(df, "EBIT", cfg.avg_years, cfg).median(axis=1)
    net_debt = df["Net_Debt"]
    ratio = (net_debt / ebit.where(ebit > 0)).clip(lower=0)
    loss = ebit.notna() & (ebit <= 0)
    df["NetDebt_to_EBIT"] = ratio.mask(loss & (net_debt > 0), 99.0).mask(loss & (net_debt <= 0), 0.0)

    cover = pd.Series(np.nan, index=df.index)
    for p in ("TTM", cfg.latest):
        interest = _col(df, f"Interest Expense_{p}").abs()
        cover = cover.fillna(_col(df, f"EBIT_{p}") / interest.where(interest > 0))
    df["Interest_Coverage"] = cover.mask(df["Financial_Mode"])


def _stability(df: pd.DataFrame, cfg: Settings) -> None:
    """ROIC spread in percentage points, share of periods with positive FCF, and FCF conversion.

    Measured on levels rather than % deviations, so values near zero or changing sign do not explode.
    """
    years = cfg.fiscal_years
    roic = _distinct(df, "ROIC", years, cfg)
    df["ROIC_StdDev"] = roic.std(axis=1, ddof=0).where(roic.notna().sum(axis=1) >= 3)
    # Scored: the spread around the ROIC's own trend line (a steady climb is a trend, credited in
    # `_return_quality`, not unsteadiness), against the level (at least `roic_variability_floor` points), since in
    # points the best businesses look the least steady (a 60% ROIC moving 8 points against an 8% one moving 1).
    df["ROIC_Slope"], spread, df["ROIC_Fit_R2"] = _roic_fit(df, cfg)
    level = roic.mean(axis=1).abs().clip(lower=cfg.roic_variability_floor)
    df["ROIC_Variability"] = spread / level

    fcf = _distinct(df, "FCF_Reported", years, cfg)  # all capex charged, growth capex included
    n = fcf.notna().sum(axis=1)
    df["FCF_Positive_Share"] = (fcf > 0).sum(axis=1) * 100 / n.where(n >= 3)

    ni = _distinct(df, "Net Income", years, cfg)
    # A loss with negative FCF is the worst conversion, not an unknown one (a loss with positive FCF stays
    # unknown: write-downs make net income meaningless there).
    conversion = (fcf / ni.where(ni > 0)).clip(0, 2).mask((ni <= 0) & (fcf <= 0), 0.0)
    enough = conversion.notna().sum(axis=1) >= 2
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    df["FCF_Conversion"] = conversion.median(axis=1).where(enough & ~earnings_based)  # NI vs NI: no signal

    # Shown percentiles (scored Oct 20-26, 2026): pricing power (moat) and recurring revenue (resilience).
    revenue = _distinct(df, "Total Revenue", years, cfg)
    margin = _distinct(df, "Gross Profit", years, cfg) * 100 / revenue.where(revenue > 0)
    df["Gross_Margin_StdDev"] = margin.std(axis=1, ddof=0).where(
        (margin.notna().sum(axis=1) >= 3) & ~df["Financial_Mode"]
    )
    # One exchange rate for every period, so the ratio is the change in the reporting currency.
    annual = _frame(df, "Total Revenue", cfg.annual_years)
    annual = annual.where(annual > 0)
    pairs = zip(annual.columns[:-1], annual.columns[1:], strict=True)  # (newer, older) neighbours
    changes = pd.DataFrame({new: (annual[new] / annual[old] - 1) * 100 for new, old in pairs}, index=df.index)
    # The latest fall counts too: a TTM ending well after Y0 (it covers mostly other quarters, `_distinct`) against
    # Y0 (QFIN: TTM revenue -17% on Y0 while every annual report rose).
    ttm = _col(df, "Total Revenue_TTM").where(_col(df, "Total Revenue_TTM") > 0)
    later = (_ttm_span(df, cfg) >= TTM_SHARED_SPAN).fillna(False)
    changes["TTM"] = ((ttm / annual[cfg.latest] - 1) * 100).where(later)
    df["Revenue_Worst_Change"] = changes.min(axis=1).where(changes.notna().sum(axis=1) >= 2)

    # Candidates (reported, not ranked): the capital cycle. Fast asset growth (Cooper-Gulen-Schill) and
    # profit running ahead of operating cash (Sloan's accruals). Not for balance-sheet businesses, whose
    # assets and operating cash flow move with deposits and reserves.
    assets = _frame(df, "Total Assets", [cfg.latest, cfg.prior])
    growth = (assets[cfg.latest] / assets[cfg.prior].where(assets[cfg.prior] > 0) - 1) * 100
    df["Asset_Growth"] = growth.mask(earnings_based)
    total = _distinct(df, "Total Assets", years, cfg)
    accruals = (ni - _distinct(df, "Operating Cash Flow", years, cfg)) * 100 / total.where(total > 0)
    df["Accruals"] = accruals.median(axis=1).where((accruals.notna().sum(axis=1) >= 2) & ~earnings_based)


def _roic_fit(df: pd.DataFrame, cfg: Settings) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Least-squares line through each company's ROIC (`_distinct`, by period-end date): the slope in pp per
    year (0 under 3 values), the spread of the values around the line in pp (blank under 3 values) and the line's
    adjusted R² (`common.adjusted_r2`: what chance alone gives is taken out), which weighs the slope in
    `_return_quality` (0 under `trend_min_points` values: three points line up by chance too often)."""
    periods = cfg.fiscal_years
    roic = _distinct(df, "ROIC", periods, cfg)
    y0 = _dates(df, cfg.latest)
    times = pd.DataFrame(
        {p: ((y0 - _dates(df, p)).dt.days / 365.25).fillna(float(_slot(p)) if p != "TTM" else np.nan) for p in periods}
    )
    slopes, spreads, r2s = [], [], []
    for idx, row in roic.iterrows():
        keep = row.notna() & times.loc[idx].notna()
        if keep.sum() < 3:
            slopes.append(0.0)
            spreads.append(np.nan)
            r2s.append(0.0)
            continue
        x = -times.loc[idx, keep].to_numpy(dtype=float)
        y = row[keep].to_numpy(dtype=float)
        ss_xx = float(np.sum((x - x.mean()) ** 2))
        slope = float(np.sum((x - x.mean()) * (y - y.mean())) / ss_xx) if ss_xx > 0 else 0.0
        residuals = y - y.mean() - slope * (x - x.mean())
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = min(1.0, max(0.0, 1.0 - float(np.sum(residuals**2)) / ss_tot)) if ss_tot > 0 else 0.0
        enough = keep.sum() >= cfg.trend_min_points
        slopes.append(slope)
        spreads.append(float(np.sqrt(np.mean(residuals**2))))
        r2s.append(adjusted_r2(r2, int(keep.sum())) if enough else 0.0)
    return pd.Series(slopes, index=df.index), pd.Series(spreads, index=df.index), pd.Series(r2s, index=df.index)


def _beneish(df: pd.DataFrame, cfg: Settings) -> None:
    """Candidate (reported, not ranked): Beneish's 8-variable M-score, the latest annual report against the one
    before. Above -1.78 the original study found most manipulators. Every input is needed (a gap is a gap, not
    a guess), and balance-sheet businesses are left out: receivables, current assets and leverage mean
    something else for them.
    """
    c = lambda name, p: _col(df, f"{name}_{p}")  # noqa: E731
    t, s = cfg.latest, cfg.prior

    def ratio(num, den, p):
        d = den(p) if callable(den) else c(den, p)
        return (num(p) if callable(num) else c(num, p)) / d.where(d > 0)

    sales = lambda p: c("Total Revenue", p)  # noqa: E731
    gross = lambda p: c("Gross Profit", p).fillna(sales(p) - c("Cost Of Revenue", p))  # noqa: E731
    hard = lambda p: c("Current Assets", p) + c("Net PPE", p)  # noqa: E731
    dep = lambda p: _depreciation(df, p)  # noqa: E731
    dsri = ratio("Receivables", sales, t) / ratio("Receivables", sales, s)
    gmi = ratio(gross, sales, s) / ratio(gross, sales, t)
    aqi = (1 - ratio(hard, "Total Assets", t)) / (1 - ratio(hard, "Total Assets", s))
    sgi = sales(t) / sales(s).where(sales(s) > 0)
    depi = ratio(dep, lambda p: dep(p) + c("Net PPE", p), s) / ratio(dep, lambda p: dep(p) + c("Net PPE", p), t)
    sgai = ratio("Selling General And Administration", sales, t) / ratio("Selling General And Administration", sales, s)
    lvgi = ratio("Total Liabilities Net Minority Interest", "Total Assets", t) / ratio(
        "Total Liabilities Net Minority Interest", "Total Assets", s
    )
    tata = (c("Net Income", t) - c("Operating Cash Flow", t)) / c("Total Assets", t).where(c("Total Assets", t) > 0)
    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai
         + 4.679 * tata - 0.327 * lvgi)  # fmt: skip
    earnings_based = df["Financial_Mode"] | (df["Business_Model"] == "float")
    df["Beneish_M"] = m.where(np.isfinite(m) & ~earnings_based)


def _management(df: pd.DataFrame, cfg: Settings) -> None:
    """Shown percentiles (scored Oct 20-26, 2026): return on newly invested capital (growth that earns its keep) and stock-based pay
    (managers paid in the owners' shares).

    `Incremental_ROIC` is the Buffett test on fundamentals: extra NOPAT per unit of extra invested
    capital between the oldest annual report with both and Y0. Acquisitions count, since goodwill is
    in invested capital. Blank when capital barely grew (nothing to judge) and in financial mode.
    """
    nopat_new, cap_new = _col(df, f"NOPAT_{cfg.latest}"), _col(df, f"Invested_Capital_{cfg.latest}")
    nopat_old = pd.Series(np.nan, index=df.index)
    cap_old = pd.Series(np.nan, index=df.index)
    for period in cfg.annual_years[1:]:  # oldest last, so it wins
        nopat, cap = _col(df, f"NOPAT_{period}"), _col(df, f"Invested_Capital_{period}")
        both = nopat.notna() & (cap > 0)
        nopat_old, cap_old = nopat.where(both, nopat_old), cap.where(both, cap_old)
    added = cap_new - cap_old
    enough = (cap_old > 0) & (added >= cfg.incremental_min_growth * cap_old)
    incremental = ((nopat_new - nopat_old) * 100 / added.where(enough)).clip(-50, cfg.roic_cap)
    df["Incremental_ROIC"] = incremental.mask(df["Financial_Mode"])

    revenue = _distinct(df, "Total Revenue", cfg.avg_years, cfg)
    sbc = _distinct(df, "Stock Based Compensation", cfg.avg_years, cfg)
    ocf = _distinct(df, "Operating Cash Flow", cfg.avg_years, cfg)
    sbc = sbc.where(sbc.notna() | ocf.isna(), 0.0)
    df["SBC_to_Revenue"] = (sbc * 100 / revenue.where(revenue > 0)).median(axis=1)


def _cyclical(df: pd.DataFrame, cfg: Settings) -> None:
    """Flag operating profit that changes sign or a margin that swings widely (cycle peaks look cheap),
    or an industry in `lists/cyclical_industries.txt` (five years may hold no trough).

    Uses the margin rather than EBIT itself, so a fast but steady grower is not flagged.
    `Cyclical_Industry` records the industry route, for the report note.
    """
    ebit = _distinct(df, "EBIT", cfg.fiscal_years, cfg)
    # A lender's 'EBIT' (pre-tax + interest) moves with rates, not with the business: net income instead.
    ebit = ebit.where(~df["Financial_Mode"], _distinct(df, "Net Income", cfg.fiscal_years, cfg), axis=0)
    rev = _distinct(df, "Total Revenue", cfg.fiscal_years, cfg)
    margin = ebit / rev.where(rev > 0)
    sign_change = (ebit.max(axis=1) > 0) & (ebit.min(axis=1) < 0)
    cv = margin.std(axis=1, ddof=0) / margin.mean(axis=1).abs().replace(0, np.nan)
    df["Cyclical_Industry"] = _text(df, "industry").map(is_cyclical_industry).astype(bool)
    statistical = (margin.notna().sum(axis=1) >= 3) & (sign_change | (cv > cfg.cyclical_margin_cv))
    df["Cyclical"] = statistical | df["Cyclical_Industry"]


def _balance_sheet(df: pd.DataFrame, cfg: Settings) -> None:
    """Financial-mode Quality inputs: book compounding and equity / assets.

    `Book_Compounding` = yearly growth of book value per share (reporting currency) + dividends as % of
    book (median of TTM and the last three years): how fast the owners' stake compounds, Buffett's
    yardstick for an insurer. `Equity_to_Assets` (common equity, %) is the leverage measure.
    """
    years = {_slot(p): p for p in cfg.annual_years}
    growth = _log_slope_growth(pd.DataFrame({k: _book_per_share(df, p) for k, p in years.items()}))
    common = _frame(df, "Common Stock Equity", cfg.avg_years).fillna(_frame(df, "Stockholders Equity", cfg.avg_years))
    paid = -_distinct(df, "Cash Dividends Paid", cfg.avg_years, cfg)
    payout = (paid.clip(lower=0) * 100 / common.where(common > 0)).median(axis=1).fillna(0)
    fin = df["Financial_Mode"]
    df["Book_Compounding"] = (growth + payout).where(fin)
    equity = _latest(df, "Common Stock Equity", cfg).fillna(_latest(df, "Stockholders Equity", cfg))
    assets = _latest(df, "Total Assets", cfg)
    df["Equity_to_Assets"] = (equity * 100 / assets.where(assets > 0)).where(fin)


def _recent_change(df: pd.DataFrame, cfg: Settings) -> None:
    """Recent change (%) in revenue and EBIT: TTM vs the latest annual report, else that report vs the
    one before. A price fall with these intact looks like mood; with them falling, like a reason.

    TTM and the annual report share the quarters before the new ones, so their difference is divided
    by the share of a year between the two period ends: a quarter that fell 36% on the year moves the
    TTM only ~9%. The result estimates the year-on-year change of the new quarters (in a seasonal
    business right in sign, approximate in size).
    """
    span = ((_dates(df, "TTM") - _dates(df, cfg.latest)).dt.days / 365.25).clip(0.25, 1.0).fillna(1.0)
    for concept, name in (("Total Revenue", "Revenue_Change_Recent"), ("EBIT", "EBIT_Change_Recent")):
        ttm, y0, y1 = (
            _col(df, f"{concept}_TTM"),
            _col(df, f"{concept}_{cfg.latest}"),
            _col(df, f"{concept}_{cfg.prior}"),
        )
        now, before = ttm.where(ttm.notna(), y0), y0.where(ttm.notna(), y1)
        years = span.where(ttm.notna(), 1.0)
        df[name] = (now - before) * 100 / before.abs().where(before.abs() > 0) / years


def _roic_trend(df: pd.DataFrame, cfg: Settings) -> None:
    """`ROIC_Trend`: mean ROIC of Y0 and TTM less that of Y2 and Y3, in percentage points (a % change explodes
    when the old ROIC is near zero)."""
    older = [f"ROIC_{y}" for y in cfg.annual_years[2:4] if f"ROIC_{y}" in df.columns]
    newer = [f"ROIC_{y}" for y in (cfg.latest, "TTM") if f"ROIC_{y}" in df.columns]
    df["ROIC_Trend"] = df[newer].mean(axis=1) - df[older].mean(axis=1) if older and newer else np.nan


def _return_quality(df: pd.DataFrame, cfg: Settings) -> None:
    """Return_Quality: single Quality measure based on ROIC (ROE for financials).
    Adjusts normalised level for its OLS trend (pp/yr x R², projected 3 years beyond the level's own date, capped
    at +/-10 pp), shrinks positive returns by variability around that trend (unsteadiness s = 1/(1+V)),
    and applies a leverage haircut for high debt or low bank capital.
    """
    cyclical = df["Cyclical"].fillna(False)

    # Trend adjustment (`ROIC_Slope` x `ROIC_Fit_R2`, `_stability`: noise around a flat line is no trend), from the
    # level's date (`ROIC_Norm_Age`: a median sits ~1.5 years back, the latest value at 0) to 3 years ahead, capped at
    # +/- roic_adj_cap pp (Oct 28, 2026). Cyclicals none: the level is already mid-cycle, so a fall since the peak
    # is the cycle, not news, unless ROIC fell every year. An over-earner capped at its median gets no rise back.
    horizon = cfg.roic_trend_years + _col(df, "ROIC_Norm_Age").fillna(0.0)
    adj = (horizon * df["ROIC_Slope"] * _col(df, "ROIC_Fit_R2").fillna(0.0)).clip(-cfg.roic_adj_cap, cfg.roic_adj_cap)
    declining = df.get("ROIC_Norm_Declining", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    adj = adj.where(~cyclical, adj.clip(upper=0.0).where(declining, 0.0))
    capped = df.get("ROIC_Norm_Capped", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    adj = adj.where(~capped, adj.clip(upper=0.0))

    level = df["ROIC_Norm"]
    eff_return = level + adj

    # Steadiness s = 1 / (1 + V); under 3 years of ROIC the steadiness is unknown, not perfect: a fixed typical
    # spread (`missing_roic_variability`, the same in every run, so scores stay benchmark-invariant).
    v = df["ROIC_Variability"].clip(lower=0).fillna(cfg.missing_roic_variability)
    s = 1.0 / (1.0 + v)

    raw = eff_return.where(eff_return <= 0, eff_return * s)

    # Leverage haircut
    haircuts = pd.Series(1.0, index=df.index)
    is_fin = df["Financial_Mode"].fillna(False)

    # Operating leverage: NetDebt/EBIT
    nd_ebit = df["NetDebt_to_EBIT"].fillna(0.0)
    op_severe = nd_ebit > cfg.leverage_haircut_operating_severe
    op_moderate = (nd_ebit > cfg.leverage_haircut_operating_moderate) & ~op_severe
    haircuts = haircuts.mask(~is_fin & op_severe, 0.50).mask(~is_fin & op_moderate, 0.75)

    # Banks: ROE at a standard leverage (ROE x Equity/Assets / `bank_standard_equity`, about ROA x 12.5), so a
    # thinner equity cushion does not read as a better business (Oct 28, 2026); this replaces their haircut.
    bank = (df["Business_Model"] == "bank") & is_fin
    eq_assets = df["Equity_to_Assets"]
    haircuts = haircuts.mask(bank & eq_assets.notna(), eq_assets / cfg.bank_standard_equity)

    # Insurers and holdings: Equity/Assets cliffs
    eq_assets = eq_assets.fillna(100.0)
    fin_severe = eq_assets < cfg.leverage_haircut_bank_severe
    fin_moderate = (eq_assets < cfg.leverage_haircut_bank_moderate) & ~fin_severe
    other_fin = is_fin & ~bank
    haircuts = haircuts.mask(other_fin & fin_severe, 0.50).mask(other_fin & fin_moderate, 0.75)

    df["Quality_Haircut"] = haircuts
    df["Return_Quality"] = raw.where(raw <= 0, raw * haircuts)
