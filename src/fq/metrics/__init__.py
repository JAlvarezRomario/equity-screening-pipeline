"""Derived metrics, one step per module (raw Yahoo frame in, one wide frame out).

`cleaning` blanks implausible figures; `capital` rebuilds EV, leases, book value and the claims Value counts at
face; `flows` derives the per-period flows and what they pay for; `normalise` finds normal levels in a short
history; `quality` and `valuation` build the scored inputs; `flags` marks what to check. `calculate_metrics`
runs them in order: each step reads only columns that earlier steps wrote.
"""

import logging

import numpy as np
import pandas as pd

from fq.metrics.archetype import detect_archetypes
from fq.metrics.capital import (
    EV_FLOOR_SHARE,
    _book_ratios,
    _leases,
    _rebuild_enterprise_value,
    _shareholder_base,
    _value_base,
)
from fq.metrics.cleaning import clean_anomalies, repair_equity, validate_latest_year
from fq.metrics.common import _distinct_mean
from fq.metrics.flags import (
    _accounting_flag,
    _data_suspect,
    _distress,
    _model_review,
    _peak_margin,
    _revenue_decline,
)
from fq.metrics.flows import (
    _acquisitions,
    _capital_retention,
    _dividend_yield,
    _financing,
    _growth_working_capital,
    _organic_growth_capex,
    _period_metrics,
    _roic,
)
from fq.metrics.inflation import _excess_inflation, _real_returns, drop_unrestated_years
from fq.metrics.long_history import long_history
from fq.metrics.normalise import growth_profile, normalized_fcf_yield, normalized_metric
from fq.metrics.quality import (
    _annual_share_reduction,
    _balance_sheet,
    _beneish,
    _cyclical,
    _leverage,
    _management,
    _recent_change,
    _return_quality,
    _roic_trend,
    _stability,
)
from fq.metrics.valuation import _growth, _revenue_cagr, _value
from fq.settings import BUSINESS_MODELS, FINANCIAL_MODELS, Settings

__all__ = [
    "EV_FLOOR_SHARE", "calculate_metrics", "clean_anomalies", "growth_profile", "normalized_fcf_yield",
]  # fmt: skip

log = logging.getLogger(__name__)

# Latest-year unit errors are blanked first: a 100x annual figure would make a sound TTM look implausible.
UNIT_CHECKED = (
    "Total Revenue", "Net Income", "Operating Income", "Pretax Income", "Operating Cash Flow", "Free Cash Flow",
    "EBIT", "Capital Expenditure",
)  # fmt: skip
AVERAGED = (
    "NIy", "profitMargins", "operatingMargins", "DirectCashtoInvestors_Yield", "EBIT_Yield", "Gross_Profitability",
    "ROA",
)  # fmt: skip


def calculate_metrics(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Full metrics pipeline on the raw frame; returns a new frame."""
    df = df.copy()
    for concept in UNIT_CHECKED:
        df = validate_latest_year(df, concept, cfg)
    df = repair_equity(clean_anomalies(df, cfg), cfg)

    # Business model: how the company is measured (`settings.BUSINESS_MODELS`).
    df["Business_Model"], df["Archetype_Source"] = detect_archetypes(df, cfg)
    df["Peer_Group"] = df["Business_Model"].map(lambda m: BUSINESS_MODELS[m][0])
    df["Model_Family"] = df["Business_Model"].map(lambda m: BUSINESS_MODELS[m][1])
    df["Financial_Mode"] = df["Business_Model"].isin(FINANCIAL_MODELS)
    df["Float_Mode"] = df["Business_Model"].isin(("float", "client_cash"))  # cash backs claims or is clients'
    df = drop_unrestated_years(df, cfg)  # needs the model; before any period is read

    # Capital and per-period flows.
    _leases(df, cfg)
    df = _rebuild_enterprise_value(df, cfg)
    _shareholder_base(df, cfg)
    log.info("Rebuilt EV for %d companies", len(df))
    for period in cfg.fiscal_years:
        _period_metrics(df, period, cfg)
        df = df.copy()  # defragment: each period adds dozens of columns
    _roic(df, cfg)  # needs every period's capital (averages opening and closing)
    _excess_inflation(df, cfg)
    _real_returns(df, cfg)  # returns in US terms before anything reads them
    _cyclical(df, cfg)
    _growth(df, cfg)
    _organic_growth_capex(df, cfg)  # owner earnings: needs Growth_Used, changes FCF_Adj before it is normalised
    _growth_working_capital(df, cfg)  # earning power at zero real growth (`valuation._value`)
    _capital_retention(df, cfg)  # local inflation on the book (`Excess_Inflation`)
    _financing(df, cfg)
    _value_base(df, cfg)  # needs the interest `_financing` took out of the flows
    df = df.copy()

    # Quality inputs and the history measures shown beside them.
    # The level's age, over-earner cap and steady decline date and bound Quality's trend (`_return_quality`).
    df["ROIC_Norm"], basis = normalized_metric(df, "ROIC", cfg.fiscal_years, cyclical=df["Cyclical"], with_basis=True)
    for name in basis.columns:
        df[f"ROIC_Norm_{name}"] = basis[name]
    df["ROIC_Ex_Goodwill"] = normalized_metric(df, "ROIC_Ex_Goodwill", cfg.fiscal_years, cyclical=df["Cyclical"])
    for name in AVERAGED:
        df[f"{name}_Avg"] = _distinct_mean(df, name, cfg.avg_years, cfg)  # quarters shared by TTM and Y0 count once
    df["Revenue_Growth_CAGR"] = _revenue_cagr(df, cfg)
    # Per share so dilution counts (plain revenue when share counts are missing).
    df["Revenue_Per_Share_CAGR"] = _revenue_cagr(df, cfg, per_share=True).fillna(df["Revenue_Growth_CAGR"])
    df["Share_Reduction_Rate"] = _annual_share_reduction(df, cfg)
    _roic_trend(df, cfg)
    _stability(df, cfg)
    _management(df, cfg)
    _beneish(df, cfg)
    _leverage(df, cfg)
    _acquisitions(df, cfg)
    _balance_sheet(df, cfg)
    _return_quality(df, cfg)
    df = df.copy()

    # Value inputs (needs ROIC_Norm and the acquisitions), then what is only shown or flagged.
    _value(df, cfg)
    _recent_change(df, cfg)
    _book_ratios(df, cfg)
    _dividend_yield(df, cfg)
    _data_suspect(df, cfg)
    _peak_margin(df, cfg)
    _distress(df, cfg)  # needs Interest_Coverage and FCF_Reported
    _revenue_decline(df, cfg)  # needs Excess_Inflation and Inflation_US_<p>
    _accounting_flag(df, cfg)  # needs Accruals and Beneish_M
    _model_review(df, cfg)
    long_history(df, cfg)  # the long share-count and dividend record, shown only

    df["EV"] = df["enterpriseValue"] / 1e9
    df["Market Cap"] = df["marketCap"] / 1e9
    # Median daily traded value in USD (shown; the highlight lists' floor, `Settings.list_min_traded_usd`).
    turnover = df.get("Daily_Turnover_Shares", pd.Series(np.nan, index=df.index))
    df["Traded_Value_USD"] = pd.to_numeric(turnover, errors="coerce") * pd.to_numeric(
        df["currentPrice"], errors="coerce"
    )
    # A missing analyst count stays missing (a dash): Yahoo leaves it out for Airbus and Loews too, so it is not 0.
    df["numberOfAnalystOpinions"] = pd.to_numeric(df["numberOfAnalystOpinions"], errors="coerce")
    return df
