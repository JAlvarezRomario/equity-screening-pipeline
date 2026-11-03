"""Returns in US terms (`fq.inflation`, `metrics.inflation`): synthetic rates, no network."""

import json

import pandas as pd
import pytest

from conftest import make_raw
from fq import fx, inflation
from fq.metrics import calculate_metrics


def _save(cfg, local: float, us: float, area: str = "TUR") -> None:
    """A saved IMF download with flat rates for every year the synthetic statements cover."""
    years = range(2015, 2031)
    series = {"USA": {str(y): us for y in years}, area: {str(y): local for y in years}}
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "inflation.json").write_text(json.dumps({"asof": "2026-10-01", "series": series}))


def _metrics(cfg, currency: str, industry: str = "Software - Application", local: float = 33.0, us: float = 3.0):
    _save(cfg, local, us)
    raw = make_raw(cfg, n=2)
    raw["financialCurrency"], raw["industry"] = currency, industry
    return calculate_metrics(raw, cfg)


def test_excess_is_fisher():
    series = {"TUR": {2025: 33.0}, "USA": {2025: 3.0}}
    assert inflation.excess(series, "TRY", 2025) == pytest.approx(1.33 / 1.03 - 1)
    assert inflation.excess(series, "TRY", 2040) == pytest.approx(1.33 / 1.03 - 1)  # nearest year
    assert inflation.excess(series, "USD", 2025) == pytest.approx(0.0)
    assert inflation.excess({}, "TRY", 2025) == pytest.approx(1.35 / 1.03 - 1)  # built-in table
    assert inflation.excess(series, "XXX", 2025) is None


def test_every_currency_with_a_rate_has_an_area():
    assert set(fx.FALLBACK_RATES) <= set(inflation.CURRENCY_AREA)
    assert set(inflation.FALLBACK) <= set(inflation.CURRENCY_AREA)


def test_usd_returns_are_unchanged(cfg):
    df = _metrics(cfg, "USD")
    pd.testing.assert_series_equal(df["ROIC_Y0"], df["ROIC_Nominal_Y0"], check_names=False)
    assert (df["Excess_Inflation_Y0"] == 0).all()


def test_nominal_bank_roe_is_put_in_us_terms_and_keeps_local_inflation(cfg):
    df = _metrics(cfg, "TRY", industry="Banks - Regional")
    assert (df["Business_Model"] == "bank").all() and (df["Inflation_Basis"] == "nominal").all()
    nominal, real = df.loc[0, "ROIC_Nominal_Y0"], df.loc[0, "ROIC_Y0"]
    assert real == pytest.approx(((1 + nominal / 100) / (1.33 / 1.03) - 1) * 100)
    assert df.loc[0, "Retention_Rate"] == pytest.approx((1.025 * 1.33 / 1.03 - 1) * 100)


def test_restated_ias29_operating_company_gets_us_inflation_back(cfg):
    """IAS 29 returns are real; a US company's keep US inflation: the same real return must read the same."""
    df = _metrics(cfg, "TRY")
    assert (df["Inflation_Basis"] == "restated").all()
    real = df.loc[0, "ROIC_Nominal_Y0"]
    assert df.loc[0, "ROIC_Y0"] == pytest.approx(((1 + real / 100) * 1.03 - 1) * 100)
    assert (df["Retention_Owner_Y0"] == 0).all()  # an operating company keeps no book for inflation


def test_one_rate_per_company_leaves_the_trend_alone(cfg):
    """A US spike in one year must not bend another currency's ROIC history."""
    _save(cfg, 2.0, 3.0, area="JPN")
    spiked = json.loads((cfg.data_dir / "inflation.json").read_text())
    spiked["series"]["USA"]["2022"] = 8.0
    (cfg.data_dir / "inflation.json").write_text(json.dumps(spiked))
    raw = make_raw(cfg, n=2)
    raw["financialCurrency"] = "JPY"
    df = calculate_metrics(raw, cfg)
    excess = pd.DataFrame({p: df[f"Excess_Inflation_{p}"] for p in cfg.fiscal_years})
    assert excess.nunique(axis=1).eq(1).all()
    assert (excess.iloc[:, 0] < 0).all()  # yen prices rose slower: returns count for more in US terms


def test_company_note_says_what_was_done(cfg):
    from fq.report.notes import inflation_note

    bank = _metrics(cfg, "TRY", industry="Banks - Regional").iloc[0]
    note = inflation_note(bank, cfg)
    assert "TRY inflation ran" in note and "above US inflation" in note and "ROE" in note and "of its book" in note
    assert "US inflation is added back" in inflation_note(_metrics(cfg, "TRY").iloc[0], cfg)
    assert inflation_note(_metrics(cfg, "USD").iloc[0], cfg) is None


def test_missing_reporting_currency_falls_back_to_the_quote(cfg):
    """Yahoo leaves `financialCurrency` out for most of Borsa Istanbul: the quote's currency stands in."""
    _save(cfg, 33.0, 3.0)
    raw = make_raw(cfg, n=2).assign(financialCurrency=None, currency="TRY")
    df = calculate_metrics(raw, cfg)
    assert (df["Inflation_Basis"] == "restated").all()
    assert (df["Excess_Inflation_Y0"] != 0).all()


def test_the_owner_trend_is_deflated_by_the_inflation_of_its_own_years(cfg):
    # The trend's growth is real growth: deflated by the inflation of the years it was fitted on (US 6% here), not
    # by the 2.5% assumed for the future, else part of a high-inflation period's rise read as real growth.
    runs = {}
    for us in (2.5, 6.0):
        _save(cfg, local=us, us=us, area="GBR")
        inflation._tables.clear()  # the saved table is cached per process
        raw = make_raw(cfg, n=2)
        raw["financialCurrency"] = "USD"
        runs[us] = calculate_metrics(raw, cfg)
    assert runs[6.0]["Inflation_US_Mean"].tolist() == pytest.approx([0.06, 0.06])
    a, b = runs[2.5], runs[6.0]
    assert (a["Owner_Trend"].abs() < 15).all()  # inside the caps: g = w x ((1 + s) / (1 + inflation) - 1)
    g25, g6 = a["Owner_Trend"] / 100, b["Owner_Trend"] / 100
    assert (g6 < g25).all()
    # Same slope s and weight w in both runs, so w = (1.025 g25 - 1.06 g6) / (1.06 - 1.025): a weight in (0, R²].
    w = (1.025 * g25 - 1.06 * g6) / (1.06 - 1.025)
    assert ((w > 0) & (w <= a["Owner_Smoothness"] + 1e-9)).all()


def test_a_restated_history_leaves_out_the_years_before_its_break(cfg):
    # LMKDC.IS: revenue 23.5M in Y3, then 115-177M in the restated (IAS 29) years. The older year is in another
    # measuring unit, so it must not enter trends, levels or compounding.
    _save(cfg, local=33.0, us=3.0)
    inflation._tables.clear()
    raw = make_raw(cfg, n=2)
    raw["financialCurrency"] = "TRY"
    for p in ("Y3", "Y4"):
        for col in ("Total Revenue", "Free Cash Flow", "Net Income"):
            raw.loc[0, f"{col}_{p}"] = raw.loc[0, f"{col}_{p}"] / 5  # a 5x step into Y2
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Inflation_Basis"] == "restated"
    assert df.loc[0, ["Total Revenue_Y3", "Total Revenue_Y4", "Free Cash Flow_Y3"]].isna().all()
    assert pd.notna(df.loc[0, "Total Revenue_Y2"])
    assert "before Y2" in str(df.loc[0, "Anomaly_Notes"])
    assert pd.notna(df.loc[1, "Total Revenue_Y3"])  # a smooth restated history keeps every year
