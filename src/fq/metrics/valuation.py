"""Growth and Value's inputs (`_value`)."""

import numpy as np
import pandas as pd

from fq.metrics.capital import _book_per_share, _to_value_base
from fq.metrics.common import (
    TTM_SHARED_SPAN,
    _col,
    _dates,
    _distinct,
    _frame,
    _log_slope_growth,
    _slot,
    _ttm_span,
    adjusted_r2,
    local_inflation,
    us_terms,
)
from fq.metrics.normalise import _pretax_power, earnings_collapse, normalized_fcf_yield, run_rate
from fq.settings import GAINS_INSIDE_MODELS, Settings


def _value(df: pd.DataFrame, cfg: Settings) -> None:
    """Value's inputs: normalised flows before financing, on `Value_Base` (debt and cash at face).

    `EPV_Yield` (earnings power, Greenwald): owner FCF at zero real growth, so only maintenance capex is charged
    (`Growth_Capex_Owner`; growers already have it added back in FCF_Adj), the working capital real growth moved
    is taken out (`Growth_WC_Owner`), financials keep book x inflation (`Retention_Owner`), and no
    margin-expansion premium (that is growth). `Aftertax_Yield_Avg`: clean pre-tax income before interest, after the company's tax rate
    (`_pretax_power`; `Pretax_Yield_Avg` before tax, shown and flagged). Display only: `FCFy_Norm` (per share at
    today's financing). The implied return (earnings power plus growth) was removed on Oct 21, 2026 (user's choice):
    growth is Quality's question.
    """
    years, premium = cfg.fiscal_years, cfg.margin_expansion_premium
    # Insurers and holdings: net income holds investment gains Yahoo does not separate, so a good market year looks
    # like earning power; they take the mid-cycle view (the lower of median and mean) as cyclicals do (Value only).
    gains_inside = df["Business_Model"].isin(GAINS_INSIDE_MODELS)
    cyclical = df["Cyclical"] | gains_inside
    # Profit fallen and staying down (`earnings_collapse`): earning power is at most the TTM's after-tax clean profit
    # before financing, or its half-year run-rate when lower (earnings, not one TTM's cash flow, which working
    # capital can turn negative).
    df["Earnings_Collapse"] = earnings_collapse(df, cfg)
    df["Earnings_Run_Rate"] = run_rate(df, cfg, after_tax=True)  # display: the cap from the latest two quarters
    cap = np.fmin(_col(df, "Aftertax_PreFin_Yield_TTM"), df["Earnings_Run_Rate"]).where(df["Earnings_Collapse"])
    financing = df["Net_Financing_Yield"].fillna(0)
    df["FCFy_PreFin_Norm"] = normalized_fcf_yield(df, years, premium, cyclical, "FCF_PreFin", cap=cap)
    df["FCFy_Norm"] = df["FCFy_PreFin_Norm"] - financing
    df["FCFy_EV_Norm"] = _to_value_base(df, df["FCFy_PreFin_Norm"])
    # Zero real growth: growth capex and the working capital growth moved out, the capital a financial keeps
    # against inflation in (`flows._growth_working_capital`, `flows._capital_retention`).
    at_zero_growth = {
        p: _col(df, f"Growth_Capex_Owner_{p}").fillna(0).where(~df["Organic_Capex"], 0.0)
        + _col(df, f"Growth_WC_Owner_{p}").fillna(0)
        - _col(df, f"Retention_Owner_{p}").fillna(0)
        for p in years
    }
    for p in years:
        pre = df[f"FCF_PreFin_{p}"]
        nopat = _col(df, f"NOPAT_{p}")
        room = (nopat - pre).clip(lower=0)
        add = at_zero_growth[p]
        at_zero_growth[p] = add.where((add <= 0) | nopat.isna() | (add <= room), room)
    owner = {f"FCF_Owner_{p}": df[f"FCF_PreFin_{p}"] + at_zero_growth[p] for p in years}
    df[list(owner)] = pd.DataFrame(owner)
    epv, level_basis = normalized_fcf_yield(df, years, 1.0, cyclical, "FCF_Owner", cap=cap, with_basis=True)
    df["EPV_Yield"] = _to_value_base(df, epv)
    df["Pretax_Yield_Avg"] = _to_value_base(df, _pretax_power(df, cfg, gains_inside=cyclical))  # shown, flagged
    df["Aftertax_Yield_Avg"] = _to_value_base(df, _pretax_power(df, cfg, "Aftertax_PreFin_Yield", cyclical))
    # Display (company chart, user's choice: per share on the share price): what a share earned for its owner each
    # period after its lenders (owner FCF after interest; clean net income in financial and float mode), and the
    # earning power Value scores restated the same way: before interest on `Value_Base`, less today's interest.
    for p in years:
        df[f"Owner_Earnings_{p}"] = df[f"FCF_Adj_{p}"] + at_zero_growth[p]
    df["EPV_Per_Share_Yield"] = _per_share_yield(df)
    df["Owner_Earnings_Compounding"] = _owner_earnings_compounding(df, cfg)
    _owner_yield(df, cfg, level_basis)


def _owner_yield(df: pd.DataFrame, cfg: Settings, level_basis: pd.DataFrame) -> None:
    """Owner_Yield: what the business earns today in a normal year, at zero real growth (`EPV_Yield`), before
    interest over the price of the operations (`Value_Base`: market cap plus debt at face, less spare cash), so debt
    and cash do not move the rank. No projection (user's rule, Nov 8, 2026; was 3 years ahead on the trend): growth
    is Quality's question, counted once there. The yield one share sees after today's interest, over the share
    price, is `EPV_Per_Share_Yield` (the company page's headline, never scored).

    `Owner_Trend` (shown, and a gate of the Low beta list; it never moves the yield) is the per-share real trend
    (capped at +15%/yr, floored at -10%/yr) times how smooth it is (R²: noise around a flat line is neither),
    fitted on `FCF_Reported` (every capex and working capital charged, net income for financials).

    The trend measures the same flow as the level (Oct 28, 2026): before financing (the after-tax interest
    `flows._financing` adds back), so leverage does not magnify it, and in constant currency (today's rate, as the
    level), so a currency that already fell is not charged again; real growth is then net of the currency's own
    inflation (`common.local_inflation`: the US inflation of those years compounded with `Excess_Inflation`). Banks,
    insurers, holdings and health insurers fit it on the level's own flow, `FCF_Owner` (clean net income less the
    book kept against inflation; Nov 6, 2026: raw net income held one-offs the level leaves out). The rate is the
    slope over the
    line's value at the newest period or the level, whichever is larger (`_trend_denominator`), weighed by its
    adjusted R² (`common.adjusted_r2`). Under `trend_min_points` values there is no trend. Cyclicals and the models
    that take the mid-cycle view (`GAINS_INSIDE_MODELS`) get none (the level is already mid-cycle) unless the flow
    fell every year; a margin capped as an over-earner gets no rise back.
    """
    periods = cfg.fiscal_years
    shares = _frame(df, "Ordinary Shares Number", periods)
    shares = shares.where(shares > 0)
    # `_distinct`: a TTM that shares most of its quarters with Y0 does not count the same year twice.
    interest_back = _frame(df, "FCF_PreFin", periods) - _frame(df, "FCF_Adj", periods)
    flow = _distinct(df, "FCF_Reported", periods, cfg) + interest_back.fillna(0.0)
    earnings_based = df["Financial_Mode"].fillna(False).astype(bool) | (df["Business_Model"] == "float")
    flow = flow.where(~earnings_based, _distinct(df, "FCF_Owner", periods, cfg), axis=0)
    eps_frame = flow / shares

    book = pd.DataFrame({p: _book_per_share(df, p) for p in periods})
    rev_ps = _frame(df, "Total Revenue", periods) / shares
    scale_frame = rev_ps.where(~df["Financial_Mode"], book, axis=0)
    # The inflation of the years the trend is fitted on (`metrics.inflation`), not the 2.5% assumed ahead: 2022-26 US
    # CPI averaged ~4.3%, and the gap read as real growth. Restated (IAS 29) statements get 0: already real.
    inflation_then = local_inflation(df, cfg)

    y0 = _dates(df, cfg.latest)
    times = pd.DataFrame(
        {p: ((y0 - _dates(df, p)).dt.days / 365.25).fillna(float(_slot(p)) if p != "TTM" else np.nan) for p in periods}
    )

    # The level the rate compounds on, per today's share (`EPV_Yield` is on `Value_Base`, before interest as the line).
    shares_now = shares.bfill(axis=1).iloc[:, 0]
    level_ps = (_col(df, "EPV_Yield") / 100 * _col(df, "Value_Base") / shares_now).where(lambda s: s > 0).fillna(0.0)

    trends = []
    r2s = []
    mid_cycle = (df["Cyclical"].fillna(False) | df["Business_Model"].isin(GAINS_INSIDE_MODELS)).to_numpy()
    capped = level_basis["Capped"].fillna(False).astype(bool).to_numpy()

    for i, (idx, row) in enumerate(eps_frame.iterrows()):
        keep = row.notna() & times.loc[idx].notna()
        v = row[keep]
        t = times.loc[idx, keep]
        if len(v) < cfg.trend_min_points:
            trends.append(0.0)
            r2s.append(0.0)
            continue

        x = -t.to_numpy(dtype=float)
        y = v.to_numpy(dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx <= 0:
            trends.append(0.0)
            r2s.append(0.0)
            continue

        slope = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
        y_pred = y_mean + slope * (x - x_mean)
        ss_tot = float(np.sum((y - y_mean) ** 2))
        ss_res = float(np.sum((y - y_pred) ** 2))
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

        scales = scale_frame.loc[idx, v.index].dropna().to_numpy(dtype=float)
        mean_scale = float(np.mean(scales)) if len(scales) else 0.0
        newest = abs(float(y_mean + slope * (x.max() - x_mean)))  # the line at the newest period
        denom = _trend_denominator(newest, float(level_ps.loc[idx]), mean_scale, cfg)

        g_raw = (1 + slope / denom) / (1 + inflation_then.loc[idx]) - 1
        g = min(max(g_raw, cfg.trend_floor_down / 100.0), cfg.trend_cap_up / 100.0) * adjusted_r2(r2, len(y))

        if mid_cycle[i]:
            falling = bool((np.diff(y[np.argsort(x)]) < 0).all())
            g = min(g, 0.0) if falling else 0.0
        if capped[i]:
            g = min(g, 0.0)

        trends.append(g)
        r2s.append(r2)

    g_series = pd.Series(trends, index=df.index)
    df["Owner_Trend"] = g_series * 100.0
    df["Owner_Smoothness"] = pd.Series(r2s, index=df.index)

    # No projection (user's rule, Nov 8, 2026): the yield is today's earning power; the trend is history.
    df["Owner_Yield"] = df["EPV_Yield"]


def _per_share_yield(df: pd.DataFrame) -> pd.Series:
    """EPV_Per_Share_Yield (the company page's headline and the yellow line of its chart; never scored): what one
    share earns its owner today in a normal year, at zero growth. The flow before interest on `Value_Base`, less
    today's net interest after tax (`Net_Financing_Yield`: paid on debt, or earned on spare cash), over the market
    cap of the parent's share (`FCF_Base`). Net cash pulls it below `Owner_Yield`, net debt lifts it above: that is
    why the ranking stays on `Owner_Yield` (Greenblatt's EBIT/EV; comparability across balance sheets). Financial
    and float mode: `Value_Base` is `FCF_Base`, so only the financing differs."""
    base = _col(df, "FCF_Base")
    base = base.where(base > 0)
    return (_col(df, "EPV_Yield") * _col(df, "Value_Base") - _col(df, "Net_Financing_Yield").fillna(0) * base) / base


def _trend_denominator(newest: float, level: float, mean_scale: float, cfg: Settings) -> float:
    """What the trend's slope is a rate of: the line's newest value, or the level the rate compounds on when that is
    larger. The line is reported FCF (growth capex charged), the level earning power (growth capex out), so the same
    slope over the smaller line read as a faster rate on the larger level (IBEX: line 0.58 of the level, 11.7%/yr)."""
    return max(newest, level, cfg.trend_scale_floor * max(0.0, mean_scale), 1e-4)


def _steady_growth(
    values: pd.DataFrame,
    min_years: int = 4,
    trend_only: bool = False,
    times: pd.DataFrame | None = None,
    adjusted: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """(growth %, how it was measured) for the growth chart (user's rule, Oct 2026). Columns = years before Y0
    (or `times`: the same shape, years before Y0 per value, so a TTM can sit between them); needs `min_years`
    positive values.

    - revenue rose every year: the latest year's growth (it has earned being taken at its word);
    - and growth sped up every year: plus the latest step-up (8% after 5% counts 11%);
    - otherwise the log-slope x its R², so bumps shrink the trend toward 0 (a steady decline stays negative).
    Yearly growth between years with a gap is annualised. `trend_only`: the log-slope x R² for every row (Quality's
    `Per_Share_Compounding`: one year is too little evidence for a criterion). `adjusted`: weigh by adjusted R²
    (`common.adjusted_r2`) instead, so what chance alone gives a line through few points is not credited.
    """
    out, basis = [], []
    for idx, row in values.iterrows():
        v = row[row > 0]
        when = (times.loc[idx, v.index] if times is not None else pd.Series(v.index, index=v.index)).astype(float)
        keep = when.notna()
        v, when = v[keep], when[keep]
        if len(v) < min_years:
            out.append(np.nan)
            basis.append(None)
            continue
        order = np.argsort(-when.to_numpy())  # oldest first
        slots, revenue = when.to_numpy()[order], v.to_numpy(dtype=float)[order]
        yearly = (revenue[1:] / revenue[:-1]) ** (1 / (slots[:-1] - slots[1:])) - 1
        if (yearly > 0).all() and not trend_only:
            latest = yearly[-1] * 100
            if (np.diff(yearly) > 0).all():
                out.append(latest + (yearly[-1] - yearly[-2]) * 100)
                basis.append("accelerating every year: latest growth + its step-up")
            else:
                out.append(latest)
                basis.append("rising every year: latest growth")
            continue
        x, y = -slots, np.log(revenue)
        slope, intercept = np.polyfit(x, y, 1)
        spread = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - (intercept + slope * x)) ** 2).sum() / spread if spread > 0 else 0.0
        out.append((np.exp(slope) - 1) * 100 * (adjusted_r2(r2, len(y)) if adjusted else r2))
        basis.append("not rising every year: trend x R²")
    return pd.Series(out, index=values.index), pd.Series(basis, index=values.index, dtype=object)


def _growth(df: pd.DataFrame, cfg: Settings) -> None:
    """`Growth_Used`: revenue slope in US terms (`common.us_terms`; the `_USD` columns are US dollar equivalents, no
    exchange rate of the past), the lower of total and per share; it decides which capex buys growth
    (`flows._organic_growth_capex`) and flags implausible yields (`flags._data_suspect`).

    Per share catches dilution; taking the lower of the two stops buybacks from counting again
    (they are already in FCF, cash to investors and share reduction). The dilution stock pay causes is
    added back to per-share growth (`SBC_Dilution`: SBC / market cap, median of the years), since FCF
    already deducts SBC as a cost. Clipped, halved for cyclicals (a few good years may be a peak), 0
    when unknown.
    """
    years = {_slot(p): p for p in cfg.annual_years}
    revenue = us_terms(_frame(df, "Total Revenue", cfg.annual_years), df, cfg)
    usd = {k: revenue[p] for k, p in years.items()}
    shares = {k: _col(df, f"Ordinary Shares Number_{p}") for k, p in years.items()}
    usd_per_share = pd.DataFrame({k: usd[k] / shares[k].where(shares[k] > 0) for k in years})
    total = _log_slope_growth(pd.DataFrame(usd))
    per_share = _log_slope_growth(usd_per_share)
    df["Revenue_Growth_USD"] = total
    # Display only (growth chart): latest growth for steady risers, else the trend discounted for bumps.
    df["Revenue_Growth_Steady"], df["Revenue_Growth_Basis"] = _steady_growth(pd.DataFrame(usd))
    df["Per_Share_Compounding"] = _per_share_compounding(df, cfg)
    sbc = _distinct(df, "Stock Based Compensation", cfg.avg_years, cfg).median(axis=1)
    df["SBC_Dilution"] = (sbc.clip(lower=0) * 100 / df["marketCap"].where(df["marketCap"] > 0)).fillna(0)
    g = np.fmin(total, per_share + df["SBC_Dilution"])  # fmin ignores a missing side
    # Financial mode: revenue says little about a balance sheet; it grows as book value per share does.
    book_us = us_terms(pd.DataFrame({p: _book_per_share(df, p) for p in years.values()}, index=df.index), df, cfg)
    book = {k: book_us[p] for k, p in years.items()}
    df["Book_Growth_USD"] = _log_slope_growth(pd.DataFrame(book)).where(df["Financial_Mode"])
    g = g.where(~df["Financial_Mode"], df["Book_Growth_USD"]).clip(cfg.growth_floor, cfg.growth_cap)
    g = g.where(~df["Cyclical"] | (g <= 0), g * cfg.cyclical_growth_factor)  # halving a decline would soften it
    df["Growth_Used"] = g.fillna(0.0)


def _per_share_compounding(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Quality: what a long-term holder's share of the business grew by, buybacks and dilution included: the trend
    of revenue per share in US terms (`common.us_terms`: local growth net of the currency's inflation beyond US, no
    exchange rate of the past; Nov 6, 2026) x its adjusted R² (bumps and chance shrink it), over the annual years and
    TTM, each placed at its real period end (Y0 left out where TTM covers most of it, `_distinct`), at least
    `trend_min_points` values (Nov 6, 2026, as every scored trend since Oct 28). Financial mode: book value per share
    in US terms (`capital._book_per_share`, every balance date its own point), since a lender or insurer grows
    through its book; one yardstick, ranked market-wide (user's choice, Oct 20, 2026)."""
    periods = cfg.fiscal_years
    book = us_terms(pd.DataFrame({p: _book_per_share(df, p) for p in periods}, index=df.index), df, cfg)
    df["Book_Per_Share_Compounding"] = _compounding(df, cfg, book)
    # What the net assets earn their owner a year (user's choice, Nov 14, 2026; shown, and the test that splits the
    # two NAV lists, `report.screens.nav_compounds`; never scored): the book's trend plus the dividends paid on it.
    df["NAV_Compounding"] = df["Book_Per_Share_Compounding"] + _book_payout(df, cfg)
    per_share = _us_per_share(df, cfg, _frame(df, "Total Revenue", periods))
    return _compounding(df, cfg, per_share.where(~df["Financial_Mode"], book, axis=0))


def _book_payout(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Dividends paid as % of common equity, the median of TTM and the last years (as `quality._balance_sheet` adds
    to a financial's `Book_Compounding`); 0 when none is on file."""
    common = _frame(df, "Common Stock Equity", cfg.avg_years).fillna(_frame(df, "Stockholders Equity", cfg.avg_years))
    paid = -_distinct(df, "Cash Dividends Paid", cfg.avg_years, cfg)
    return (paid.clip(lower=0) * 100 / common.where(common > 0)).median(axis=1).fillna(0)


def _owner_earnings_compounding(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Display and the Compounder list (`ranking.compounder`), never scored: the trend of US-terms owner earnings per share
    (`Owner_Earnings_<p>`, the company chart's blue bars) x its adjusted R² (Oct 30, 2026: as the scored trends, so a
    noisy series is not credited a third of its slope), as `_per_share_compounding` measures revenue. A
    Mastercard grows earnings per share faster than revenue as margins widen; a business whose earnings stall while
    revenue grows is not compounding for its owner. Owner earnings are noisy, so not a Quality criterion (user's
    choice, Oct 21, 2026)."""
    per_share = _us_per_share(df, cfg, _frame(df, "Owner_Earnings", cfg.fiscal_years))
    return _compounding(df, cfg, per_share)


def _us_per_share(df: pd.DataFrame, cfg: Settings, values: pd.DataFrame) -> pd.DataFrame:
    """A flow per share in US terms (`common.us_terms`); Y0 left out where TTM covers most of it (`_distinct`)."""
    periods = cfg.fiscal_years
    shares = _frame(df, "Ordinary Shares Number", periods)
    per_share = us_terms(values[periods], df, cfg) / shares.where(shares > 0)
    mostly_shared = (_ttm_span(df, cfg) < TTM_SHARED_SPAN).fillna(False)
    per_share[cfg.latest] = per_share[cfg.latest].mask(mostly_shared & per_share["TTM"].notna())
    return per_share


def _compounding(df: pd.DataFrame, cfg: Settings, per_share: pd.DataFrame) -> pd.Series:
    """Log-slope x adjusted R² of per-share values over TTM and the annual years at their real period ends, at least
    `trend_min_points` of them (Nov 6, 2026: plain R² on three points credited noise about half its slope)."""
    periods = cfg.fiscal_years
    y0 = _dates(df, cfg.latest)
    times = pd.DataFrame(
        {p: ((y0 - _dates(df, p)).dt.days / 365.25).fillna(float(_slot(p)) if p != "TTM" else np.nan) for p in periods}
    )
    compounding, _ = _steady_growth(
        per_share, min_years=cfg.trend_min_points, trend_only=True, times=times, adjusted=True
    )
    return compounding


def _revenue_cagr(df: pd.DataFrame, cfg: Settings, per_share: bool = False) -> pd.Series:
    """Revenue (or revenue per share) CAGR in US terms (`common.us_terms`) between the oldest and newest annual slots
    with a value: growth that only reflects a currency's inflation beyond US does not count, and no exchange rate of
    the past enters (Nov 6, 2026)."""
    cols = {_slot(p): p for p in cfg.annual_years if f"Total Revenue_{p}" in df.columns}
    if len(cols) < 2:
        return pd.Series(np.nan, index=df.index)
    revenue_us = us_terms(_frame(df, "Total Revenue", list(cols.values())), df, cfg)

    def value(p: str) -> pd.Series:
        rev = revenue_us[p]
        if per_share:
            shares = _col(df, f"Ordinary Shares Number_{p}")
            rev = rev / shares.where(shares > 0)
        return rev.where(rev > 0)

    revs = pd.DataFrame({k: value(p) for k, p in cols.items()})

    def cagr(row: pd.Series) -> float:
        valid = row.dropna()  # index = slot; larger = older
        if len(valid) < 2:
            return np.nan
        newest, oldest = valid.index.min(), valid.index.max()
        n = oldest - newest
        return ((valid[newest] / valid[oldest]) ** (1 / n) - 1) * 100 if n > 0 else np.nan

    return revs.apply(cagr, axis=1)
