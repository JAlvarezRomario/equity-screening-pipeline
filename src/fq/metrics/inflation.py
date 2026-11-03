"""Returns in US terms: a 30% return in a currency losing 30% a year is no return at all (user's rule, Oct 28,
2026: what is scored and shown must compare with the dollar).

Growth and compounding are already measured in USD of the time (`valuation`, `flows`), so a falling currency
washes out there. Returns on capital are not: profit and capital are both in the local currency, so the ratio
keeps its inflation. `_excess_inflation` finds each period's inflation beyond US inflation (IMF, `fq.inflation`),
`_real_returns` divides it out of `ROIC_<p>` (ROE for financials), and `flows._capital_retention` charges banks,
insurers and holdings the local inflation on their book. Statements already restated for inflation (IAS 29) hold
real returns, so they get US inflation back instead (`_basis`): every return ends in US nominal terms.
"""

import numpy as np
import pandas as pd

from fq import inflation, universe
from fq.metrics.common import _dates, _reporting_currency
from fq.settings import HYPERINFLATION_CURRENCIES, Settings

# A fiscal year is dated by the calendar year it mostly covers (a March year end belongs to the year before).
HALF_YEAR = pd.Timedelta(days=182)


def _basis(df: pd.DataFrame) -> pd.Series:
    """`restated` (IAS 29: an operating company reporting in a hyperinflationary currency; banks and insurers
    there are exempt) or `nominal`; `lists/inflation_basis.txt` overrides."""
    currency = _reporting_currency(df)
    restated = currency.isin(HYPERINFLATION_CURRENCIES) & ~df["Financial_Mode"]
    basis = pd.Series(np.where(restated, "restated", "nominal"), index=df.index)
    overrides = universe.inflation_basis()
    return df["Ticker"].map(overrides).fillna(basis)


# A step in revenue this large between two restated years is a change of measuring unit, not growth.
RESTATEMENT_BREAK = 3.0


def drop_unrestated_years(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Leave out the years of a `restated` history older than a break in it: revenue more than `RESTATEMENT_BREAK`
    times the year before (LMKDC.IS: 23.5M in Y3, 115-177M once restated). Yahoo serves some pre-IAS 29 years
    as first reported, in money of their own time, so they made every trend and compounding rate a jump and broke
    the "fell every year" test. Every figure of those years is blanked (period ends kept) and a note says so. Runs
    once the business model is known (`_basis` needs it) and before anything reads the periods."""
    restated = _basis(df).eq("restated")
    years = cfg.annual_years
    for idx in df.index[restated]:
        revenue = [pd.to_numeric(df.at[idx, f"Total Revenue_{p}"], errors="coerce") if f"Total Revenue_{p}" in df
                   else np.nan for p in years]  # fmt: skip
        newer = next(
            (k for k in range(len(years) - 1)
             if revenue[k] > 0 and revenue[k + 1] > 0 and revenue[k] > RESTATEMENT_BREAK * revenue[k + 1]),
            None,
        )  # fmt: skip
        if newer is None:
            continue
        older = years[newer + 1 :]
        cols = [c for c in df.columns if c.rsplit("_", 1)[-1] in older and not c.startswith("PeriodEnd_")]
        df.loc[idx, cols] = np.nan
        note = f"Restated (IAS 29) history breaks before {years[newer]}: {', '.join(older)} left out."
        df.at[idx, "Anomaly_Notes"] = " ".join(filter(None, [df.at[idx, "Anomaly_Notes"], note]))
    return df


def _excess_inflation(df: pd.DataFrame, cfg: Settings) -> None:
    """`Excess_Inflation_<p>`: the statements' currency's CPI inflation beyond US inflation over the year the period
    covers, as a fraction ((1 + local) / (1 + US) - 1; negative where prices rise slower than in the US). `restated`
    statements (IAS 29) are already real, all inflation out, where every other return keeps US inflation: theirs is
    1 / (1 + US) - 1, so dividing by 1 + it adds US inflation back (a 12% real return reads 15.4% at 3% US, as a US
    company's would). 0 when the currency or the period's date is unknown. `Inflation_US_Mean`: US inflation over
    the same years, `Inflation_US_<p>` (fraction) each period's. `Inflation_Local` /
    `Inflation_US` (% per year) of the newest period are kept for the report note."""
    series = inflation.table(cfg.data_dir / "inflation.json")
    df["Inflation_Basis"] = _basis(df)
    nominal = df["Inflation_Basis"].eq("nominal")
    currency = _reporting_currency(df)
    yearly = pd.DataFrame(index=df.index)
    for period in cfg.fiscal_years:
        year = (_dates(df, period) - HALF_YEAR).dt.year
        yearly[period] = pd.Series(
            [
                np.nan if pd.isna(y) else inflation.excess(series, c, int(y)) if ok else _us_back(series, int(y))
                for c, y, ok in zip(currency, year, nominal, strict=True)
            ],
            index=df.index,
            dtype=float,
        )
    # One rate per company, the average over the years its history covers: year by year, a US spike (8% in one
    # year, 2.5% in Japan) would lift that year's return of every other currency and bend its ROIC trend, though
    # the currency, not the business, absorbed the gap. Levels move to US terms; trends stay the company's own.
    mean = np.exp(np.log1p(yearly).mean(axis=1)) - 1
    for period in cfg.fiscal_years:
        df[f"Excess_Inflation_{period}"] = mean.fillna(0.0)
    # US inflation over the same years (a fraction): (1 + it) x (1 + excess) is the local inflation those years
    # had, which deflates a trend fitted on them (`valuation._owner_yield`); blank when no period has a date.
    us = pd.DataFrame(
        {
            p: [np.nan if pd.isna(y) else (inflation.rate(series, "USD", int(y)) or np.nan) / 100 for y in years]
            for p, years in ((p, (_dates(df, p) - HALF_YEAR).dt.year) for p in cfg.fiscal_years)
        },
        index=df.index,
    )
    df["Inflation_US_Mean"] = np.exp(np.log1p(us).mean(axis=1)) - 1
    # Each year's own US rate: what part of a year's USD revenue growth was price, not volume
    # (`flows._organic_growth_capex`, `_growth_working_capital`); a mean would read a 2022 price rise as volume.
    for period in cfg.fiscal_years:
        df[f"Inflation_US_{period}"] = us[period]
    newest = (_dates(df, "TTM").fillna(_dates(df, cfg.latest)) - HALF_YEAR).dt.year
    df["Inflation_Local"] = [
        inflation.rate(series, c, int(y)) if pd.notna(y) else np.nan for c, y in zip(currency, newest, strict=True)
    ]
    df["Inflation_US"] = [inflation.rate(series, "USD", int(y)) if pd.notna(y) else np.nan for y in newest]


def _us_back(series: dict, year: int) -> float | None:
    """The `Excess_Inflation` of a real (IAS 29) return: dividing by 1 + it multiplies by 1 + US inflation."""
    us = inflation.rate(series, "USD", year)
    return None if us is None else 1 / (1 + us / 100) - 1


def _real_returns(df: pd.DataFrame, cfg: Settings) -> None:
    """`ROIC_<p>` in US terms: (1 + nominal) / (1 + `Excess_Inflation_<p>`) - 1 (Fisher), still capped at
    +/- `roic_cap`; `ROIC_Ex_Goodwill_<p>` likewise. The nominal figure stays as `ROIC_Nominal_<p>` for the report
    note."""
    for period in cfg.fiscal_years:
        if f"ROIC_{period}" not in df.columns:
            continue
        nominal = df[f"ROIC_{period}"]
        df[f"ROIC_Nominal_{period}"] = nominal
        for name in ("ROIC", "ROIC_Ex_Goodwill"):
            col = f"{name}_{period}"
            if col in df.columns:
                real = ((1 + df[col] / 100) / (1 + df[f"Excess_Inflation_{period}"]) - 1) * 100
                df[col] = real.clip(lower=-cfg.roic_cap, upper=cfg.roic_cap)
