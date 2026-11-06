import numpy as np
import pandas as pd
import pytest

from fq.settings import FIELDS_VERSION, SCHEMA_VERSION, Settings


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "output")


def make_raw(cfg: Settings, n: int = 6, seed: int = 0, latest_end: str = "2025-12-31") -> pd.DataFrame:
    """Synthetic raw frame shaped like fetch output (USD values, one row per ticker)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {
            "Ticker": f"T{i}.X" if i % 2 else f"T{i}",
            "shortName": f"Company <{i}> & Co",
            "industry": "Software - Application",
            "country": "United States" if i % 2 else "Germany",
            "currentPrice": 50.0 + i,
            "marketCap": (3 + i) * 1e9,
            "dividendYield": 1.0,
            "longBusinessSummary": "Makes <b>things</b>.",
            "numberOfAnalystOpinions": 5,
            "TTM_Identical_To_Latest": False,
            "Stale_Annual": False,
            "Schema_Version": SCHEMA_VERSION,
            "Fields_Version": FIELDS_VERSION,
            "Fetched_At": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
        growth = 1 + rng.uniform(0.02, 0.15)
        newest = pd.Timestamp(latest_end)
        for k, year in enumerate(reversed(cfg.fiscal_years)):  # oldest first, TTM last
            rev = 1e9 * growth**k
            row[f"PeriodEnd_{year}"] = (
                (newest + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
                if year == "TTM"
                else (newest - pd.DateOffset(years=int(year[1:]))).strftime("%Y-%m-%d")
            )
            row |= {
                f"Total Revenue_{year}": rev,
                f"Net Income_{year}": rev * 0.15,
                f"Operating Income_{year}": rev * 0.2,
                f"EBIT_{year}": rev * 0.2,
                f"Pretax Income_{year}": rev * 0.19,
                f"Tax Provision_{year}": rev * 0.04,
                f"Interest Expense_{year}": rev * 0.01,
                f"Free Cash Flow_{year}": rev * (0.12 + 0.01 * i),
                f"Operating Cash Flow_{year}": rev * (0.16 + 0.01 * i),
                f"Capital Expenditure_{year}": -rev * 0.04,
                f"Stock Based Compensation_{year}": rev * 0.01,
                f"Stockholders Equity_{year}": rev * 0.8,
                f"Total Debt_{year}": rev * 0.3,
                f"Cash And Cash Equivalents_{year}": rev * 0.2,
                f"Ordinary Shares Number_{year}": 1e8 * (1 - 0.01 * k),
            }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_ecb(monkeypatch):
    """No network in tests: the euro short-term rate is never downloaded (`fx.get_borrow_rate` falls back)."""
    from fq import fx

    monkeypatch.setattr(fx, "_estr", lambda: None)
    monkeypatch.setattr(fx, "_borrow_cache", {})
