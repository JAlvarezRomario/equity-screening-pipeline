"""Extraction fixes: ADR share basis, TTM provenance, anomaly windows, estimates, dividend yield, FX."""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq import fx
from fq.fetch import _align_share_basis, _patch_capex_and_fcf, _statement_columns
from fq.metrics import calculate_metrics, clean_anomalies
from fq.report.format import report_periods
from fq.report.notes import data_notes
from fq.settings import Settings

QUARTERS = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]


def stmt(dates, **rows):
    return pd.DataFrame(rows, index=pd.to_datetime(dates)).T


# 1. ADRs / share classes -------------------------------------------------------------------------


def test_adr_share_counts_are_put_on_the_price_basis():
    # PDD as served by Yahoo: price per ADR, statements in ordinary shares (1 ADR = 4 shares).
    row = {"currentPrice": 80.97, "marketCap": 115.25e9,
           "Ordinary Shares Number_TTM": 5.6936e9, "Ordinary Shares Number_Y1": 5.6e9, "Share Issued_Y1": 5.7e9}  # fmt: skip
    ratio = _align_share_basis(row, ["TTM", "Y0", "Y1"], "PDD")
    assert 3.9 < ratio < 4.1
    assert abs(row["Ordinary Shares Number_TTM"] * 80.97 / 115.25e9 - 1) < 0.01
    assert row["Share Issued_Y1"] == 5.7e9 / ratio


def test_ordinary_company_is_untouched():
    row = {"currentPrice": 50.0, "marketCap": 5.2e9, "Ordinary Shares Number_TTM": 1e8}
    assert _align_share_basis(row, ["TTM"], "X") is None
    assert row["Ordinary Shares Number_TTM"] == 1e8
    assert _align_share_basis({"currentPrice": 50.0}, ["TTM"], "X") is None  # no market cap: no guess


# 2. TTM provenance and cross-check -----------------------------------------------------------------


def test_ttm_source_and_crosscheck():
    q = stmt(QUARTERS, **{"Total Revenue": [10, 10, 10, 10]})
    agree = stmt(["2026-06-30"], **{"Total Revenue": [40.4]})
    _, _, meta = _statement_columns(pd.DataFrame(), q, agree, ["Total Revenue"], Settings(), 1.0, "sum")
    assert meta["ttm_source"] == "quarters" and abs(meta["ttm_check"] - 0.01) < 1e-9

    half_yearly = stmt(["2026-06-30", "2025-12-31"], **{"Total Revenue": [20, 20]})
    out, _, meta = _statement_columns(pd.DataFrame(), half_yearly, agree, ["Total Revenue"], Settings(), 1.0, "sum")
    assert out["Total Revenue_TTM"] == 40.4 and meta["ttm_source"] == "yahoo_ttm" and meta["ttm_check"] is None


def test_half_year_reported_as_a_quarter_is_not_summed():
    # ACS.MC as served: the Dec "quarter" is the whole second half, so four "quarters" hold six months twice.
    annual = stmt(["2025-12-31"], **{"Total Revenue": [49.8]})
    q = stmt(QUARTERS, **{"Total Revenue": [13.8, 12.3, 25.7, 12.6]})
    yahoo = stmt(["2026-06-30"], **{"Total Revenue": [51.9]})
    out, _, meta = _statement_columns(annual, q, yahoo, ["Total Revenue"], Settings(), 1.0, "sum")
    assert out["Total Revenue_TTM"] == 51.9 and meta["ttm_source"] == "yahoo_ttm_checked"
    assert meta["ttm_check"] == pytest.approx(abs(51.9 - 64.4) / 64.4)

    yahoo_off = stmt(["2026-06-30"], **{"Total Revenue": [80.0]})  # Yahoo's TTM is the outlier: keep the sum
    out, _, meta = _statement_columns(annual, q, yahoo_off, ["Total Revenue"], Settings(), 1.0, "sum")
    assert out["Total Revenue_TTM"] == pytest.approx(64.4) and meta["ttm_source"] == "quarters_checked"


def test_anomalies_blank_only_the_lines_they_question(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[1, ["Ticker", "industry"]] = ["BANKX", "Banks - Regional"]
    raw.loc[:, "Operating Cash Flow_TTM"] = -0.5 * raw["Operating Cash Flow_Y0"]  # working capital / deposits
    raw.loc[0, "Interest Expense_TTM"] = 5 * raw.loc[0, "Interest Expense_Y0"]  # other costs booked as interest
    for p in ("TTM", "Y0"):
        raw.loc[:, f"Total Debt_{p}"] = 1e9  # flat debt
    out = clean_anomalies(raw, cfg)
    # Interest is checked first among the narrow rules, so company 0 keeps its cash flow this time ...
    assert (
        pd.isna(out.loc[0, "Interest Expense_TTM"])
        and out.loc[0, "Total Revenue_TTM"] == raw.loc[0, "Total Revenue_TTM"]
    )
    assert out.loc[0, "Net Income_TTM"] == raw.loc[0, "Net Income_TTM"]
    # ... and a bank's operating cash flow swing is not an anomaly at all.
    assert (
        out.loc[1, "Operating Cash Flow_TTM"] == raw.loc[1, "Operating Cash Flow_TTM"]
        and out.loc[1, "Anomaly_Notes"] is None
    )

    raw.loc[0, "Interest Expense_TTM"] = raw.loc[0, "Interest Expense_Y0"]
    out = clean_anomalies(raw, cfg)  # a cash-flow drop blanks the cash flows, not the income statement
    assert pd.isna(out.loc[0, "Operating Cash Flow_TTM"]) and pd.isna(out.loc[0, "Free Cash Flow_TTM"])
    assert out.loc[0, "Total Revenue_TTM"] == raw.loc[0, "Total Revenue_TTM"]


def test_aliases_are_recorded():
    annual = stmt(["2025-12-31"], **{"Operating Revenue": [50]})
    _, _, meta = _statement_columns(annual, pd.DataFrame(), pd.DataFrame(), ["Total Revenue"], Settings(), 1.0, "sum")
    assert meta["aliases"] == {"Total Revenue <- Operating Revenue"}


# 3. Anomaly check compares neighbouring periods ------------------------------------------------------


def _growth_row(cfg, ttm, y0, y1, gap=None):
    raw = make_raw(cfg, n=1)
    raw["Total Revenue_TTM"], raw["Total Revenue_Y0"], raw["Total Revenue_Y1"] = ttm, y0, y1
    raw["Total Debt_TTM"] = raw["Total Debt_Y0"] = raw["Total Debt_Y1"] = 1e9  # flat debt
    raw["TTM_Check_Gap"] = gap
    return clean_anomalies(raw, cfg).loc[0, "Total Revenue_TTM"]


def test_fast_grower_is_not_mistaken_for_an_error(cfg):
    # Sterling-like: TTM 3.44 vs Y0 2.49 (1.38x) vs Y1 2.12 (1.62x). Real growth, keep it.
    assert _growth_row(cfg, 3.44e9, 2.49e9, 2.12e9) == 3.44e9


def test_impossible_jump_is_blanked_unless_two_sources_agree(cfg):
    assert pd.isna(_growth_row(cfg, 6.0e9, 2.49e9, 2.12e9))
    assert _growth_row(cfg, 6.0e9, 2.49e9, 2.12e9, gap=0.0) == 6.0e9


# 4. Estimates are visible ----------------------------------------------------------------------------


def test_capex_estimates_are_reported(cfg):
    data = {"Total Revenue_Y0": 100.0, "Capital Expenditure_Y0": None,
            "Total Revenue_Y1": 90.0, "Capital Expenditure_Y1": -9.0}  # fmt: skip
    assert _patch_capex_and_fcf(data, ["Y0", "Y1"], "X") == ["Y0"]
    assert data["Capital Expenditure_Y0"] == -10.0

    df = calculate_metrics(make_raw(cfg, n=1).assign(CapEx_Estimated="Y0", TTM_Source="yahoo_ttm",
                                                      Share_Basis_Ratio=4.0, FX_Source="saved (stale) 2026-01-01"), cfg)  # fmt: skip
    notes = " ".join(data_notes(df.iloc[0], cfg, report_periods(df, cfg)))
    for text in ("CapEx estimated", "Dec-25", "Yahoo's TTM", "ADR", "stale"):
        assert text in notes, text


# 5. Own dividend yield and FX provenance --------------------------------------------------------------


def test_dividend_yield_is_computed_from_cash_paid(cfg):
    raw = make_raw(cfg, n=1)
    raw["Cash Dividends Paid_TTM"] = -100e6
    df = calculate_metrics(raw, cfg)
    assert abs(df.loc[0, "Dividend_Yield"] - 100e6 / df.loc[0, "marketCap"] * 100) < 1e-9
    raw["Cash Dividends Paid_TTM"] = np.nan
    raw["Cash Dividends Paid_Y0"] = np.nan
    assert calculate_metrics(raw, cfg).loc[0, "Dividend_Yield"] == 0  # non-payer
    raw["Operating Cash Flow_TTM"] = raw["Operating Cash Flow_Y0"] = np.nan
    assert pd.isna(calculate_metrics(raw, cfg).loc[0, "Dividend_Yield"])  # no cash-flow data: unknown


def test_fx_uses_saved_rates_when_live_fails(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(fx, "_live", boom)
    monkeypatch.setattr(fx, "_cache", {})
    path = tmp_path / "fx_rates.json"
    recent = (date.today() - timedelta(days=2)).isoformat()
    path.write_text(json.dumps({"asof": recent, "rates": {"EUR": 1.5}}), encoding="utf-8")
    assert fx.get_rates(path)["EUR"] == 1.5 and fx.status() == f"saved {recent}"

    monkeypatch.setattr(fx, "_cache", {})
    path.write_text(json.dumps({"asof": "2020-01-01", "rates": {"EUR": 1.5}}), encoding="utf-8")
    fx.get_rates(path)
    assert "stale" in fx.status()

    monkeypatch.setattr(fx, "_cache", {})
    assert fx.get_rates(tmp_path / "missing.json")["EUR"] == fx.FALLBACK_RATES["EUR"]
    assert fx.status().startswith("built-in table (stale)")


def test_fx_saves_live_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "_live", lambda: {c: 1.0 for c in fx.FALLBACK_RATES})
    monkeypatch.setattr(fx, "_cache", {})
    path = tmp_path / "fx_rates.json"
    fx.get_rates(path)
    assert json.loads(path.read_text(encoding="utf-8"))["asof"] == date.today().isoformat()
    assert fx.status().startswith("live")


def test_ev_uses_debt_and_cash_when_some_rows_are_missing(cfg):
    # PDD-like: no minority-interest row, no "Short Term Investments" row, big other investments.
    raw = make_raw(cfg, n=1)
    for p in cfg.fiscal_years:
        raw[f"Minority Interest_{p}"] = np.nan
        raw[f"Short Term Investments_{p}"] = np.nan
        raw[f"Other Short Term Investments_{p}"] = 2e9
    df = calculate_metrics(raw, cfg)
    cap = df.loc[0, "Effective_MarketCap"]
    debt, cash = raw.loc[0, "Total Debt_TTM"], raw.loc[0, "Cash And Cash Equivalents_TTM"] + 2e9
    assert abs(df.loc[0, "enterpriseValue"] - (cap + debt - cash)) < 1


def test_prices_only_refresh(cfg, monkeypatch):
    from fq import fetch

    week_old = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d %H:%M")
    raw = make_raw(cfg, n=3).assign(currency=["USD", "GBp", "EUR"], marketCap=[1e9, 2e9, 3e9], Fetched_At=week_old)
    raw["currentPrice"] = [10.0, 13.1, 11.5]  # cached USD prices
    monkeypatch.setattr(fx, "_cache", {str(cfg.data_dir / "fx_rates.json"): {"USD": 1.0, "GBP": 1.31, "EUR": 1.15}})
    stats = pd.DataFrame(
        {
            "Last_Close": [12.0, 1100.0, np.nan],
            "Momentum_6M": [20.0, -5.0, np.nan],
            "Above_52W_Low": [150.0, 110.0, np.nan],
        },
        index=raw["Ticker"],
    )  # pence for the GBp line; third ticker has no quote
    monkeypatch.setattr(fetch, "_price_history", lambda tickers: (None, None, None))
    monkeypatch.setattr(fetch, "price_stats", lambda close, low, volume: stats)

    out = fetch.refresh_prices(raw, list(raw["Ticker"]), cfg)
    assert out.loc[0, "currentPrice"] == 12.0 and abs(out.loc[0, "marketCap"] - 1.2e9) < 1
    assert abs(out.loc[1, "currentPrice"] - 11.0 * 1.31) < 1e-9  # 1100 GBp = 11 GBP
    assert out.loc[2, "currentPrice"] == 11.5 and pd.isna(out.loc[2, "Price_At"])  # kept, not guessed
    assert out.loc[0, "Total Revenue_Y0"] == raw.loc[0, "Total Revenue_Y0"]  # fundamentals untouched
    assert out.loc[0, "Momentum_6M"] == 20.0 and out.loc[1, "Above_52W_Low"] == 110.0

    # A price under a week old is kept: refreshed just now, so a second refresh changes nothing.
    stats["Last_Close"] = [99.0, 9999.0, np.nan]
    again = fetch.refresh_prices(out, list(out["Ticker"]), cfg)
    assert again.loc[0, "currentPrice"] == 12.0 and again.loc[1, "currentPrice"] == out.loc[1, "currentPrice"]
    later = fetch.refresh_prices(out, list(out["Ticker"]), cfg, now=pd.Timestamp.now() + pd.Timedelta(days=8))
    assert later.loc[0, "currentPrice"] == 99.0
    # A price stored before the traded value existed is refreshed a day later, so the liquidity floor gets its figure.
    next_day = fetch.refresh_prices(out, list(out["Ticker"]), cfg, now=pd.Timestamp.now() + pd.Timedelta(days=2))
    assert next_day.loc[0, "currentPrice"] == 99.0
    stats["Daily_Turnover_Shares"] = [5000.0, 800.0, np.nan]
    known = fetch.refresh_prices(out, list(out["Ticker"]), cfg)
    known = fetch.refresh_prices(known, list(known["Ticker"]), cfg, now=pd.Timestamp.now() + pd.Timedelta(days=2))
    assert known.loc[0, "Daily_Turnover_Shares"] == 5000.0


def test_pe_ratios_move_with_the_quote_not_the_currency():
    from fq import fetch

    df = pd.DataFrame({"Ticker": ["A", "B"], "trailingPE": [20.0, 10.0], "forwardPE": [16.0, np.nan],
                       "Price_At": [np.nan, np.nan], "Fetched_At": ["2026-10-01 09:00", np.nan]})  # fmt: skip
    days = pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-30"])
    close = pd.DataFrame({"A": [90.0, 100.0, 80.0], "B": [1.0, 1.0, 2.0]}, index=days)  # quote currency
    fetch._rescale_pe(df, 0, close, 80.0)
    assert df.loc[0, "trailingPE"] == 16.0 and df.loc[0, "forwardPE"] == 12.8  # price -20% since the fetch
    fetch._rescale_pe(df, 1, close, 2.0)
    assert df.loc[1, "trailingPE"] == 10.0  # no date its price was set: left as it is, not guessed
    fetch._rescale_pe(df, 0, None, 50.0)
    assert df.loc[0, "trailingPE"] == 16.0  # no price history: unchanged


def test_price_stats_momentum_and_52w_low():
    from fq.fetch import price_stats

    days = pd.date_range("2025-10-20", "2026-10-16", freq="B")
    close = pd.DataFrame({"A": np.linspace(50, 100, len(days)), "NEW": np.nan}, index=days)
    close.loc[days[-20] :, "NEW"] = 10.0  # listed a month ago
    low = close * 0.98
    stats = price_stats(close, low)
    six_months_ago = close["A"].loc[: days[-1] - pd.DateOffset(months=6)].iloc[-1]
    assert abs(stats.at["A", "Momentum_6M"] - (100 / six_months_ago - 1) * 100) < 1e-9
    assert abs(stats.at["A", "Above_52W_Low"] - (100 / 49.0 - 1) * 100) < 1e-9
    assert (
        pd.isna(stats.at["NEW", "Momentum_6M"]) and abs(stats.at["NEW", "Above_52W_Low"] - (1 / 0.98 - 1) * 100) < 1e-9
    )


def test_price_stats_skip_untraded_days_and_basis_breaks():
    from fq.fetch import price_stats

    days = pd.date_range("2025-10-20", "2026-10-16", freq="B")
    close = pd.DataFrame({"PENCE": 130.0, "RSPLIT": 3.0, "MOVER": 100.0}, index=days)
    volume = pd.DataFrame(1e6, index=days, columns=close.columns)
    close.loc[days[100:105], "PENCE"] = 1.3  # quoted in pounds on days nothing traded (MEGP.L)
    volume.loc[days[100:105], "PENCE"] = 0
    close.loc[days[90] :, "RSPLIT"] = 24.0  # 1-for-8 reverse split missing from Yahoo's history (SIVE.ST)
    close.loc[days[150] :, "MOVER"] = 250.0  # a big but real move stays in
    stats = price_stats(close, close, volume)
    assert stats.at["PENCE", "Above_52W_Low"] == pytest.approx(0.0)
    assert stats.at["RSPLIT", "Above_52W_Low"] == pytest.approx(0.0)
    assert stats.at["MOVER", "Above_52W_Low"] == pytest.approx(150.0)


def _annual_case(cfg, **values):
    """One company; values are {column: (Y1, Y0)}; returns (cleaned row, notes)."""
    raw = make_raw(cfg, n=1)
    for col, (y1, y0) in values.items():
        raw[f"{col}_Y1"], raw[f"{col}_Y0"] = y1, y0
    row = clean_anomalies(raw, cfg).iloc[0]
    return row, row["Anomaly_Notes"]


def test_real_annual_moves_are_kept_and_noted(cfg):
    # Nintendo FY Mar-26: revenue doubled, no debt reported at all.
    row, notes = _annual_case(cfg, **{"Total Revenue": (7.38e9, 14.65e9), "Total Debt": (np.nan, np.nan)})
    assert row["Total Revenue_Y0"] == 14.65e9 and notes is None  # missing debt is not "flat debt"
    # Lennar: operating cash flow 2.4 -> 0.22 bn. Kept, but flagged for the reader.
    row, notes = _annual_case(cfg, **{"Operating Cash Flow": (2.4e9, 0.22e9)})
    assert row["Operating Cash Flow_Y0"] == 0.22e9 and "worth checking" in notes
    # Terex: interest doubled on flat debt (acquisition debt for a full year). Kept and noted.
    row, notes = _annual_case(cfg, **{"Interest Expense": (0.09e9, 0.18e9), "Total Debt": (2.6e9, 2.6e9)})
    assert row["Interest Expense_Y0"] == 0.18e9 and "interest x2.0" in notes


def test_two_losses_are_not_growth(cfg):
    # Transocean: EBIT -0.16 -> -2.39 bn (impairment). The ratio of two losses is not "x14.9".
    row, notes = _annual_case(cfg, **{"EBIT": (-0.16e9, -2.39e9), "Operating Income": (-0.16e9, -2.39e9),
                                      "Total Debt": (7.2e9, 6.0e9)})  # fmt: skip
    assert row["EBIT_Y0"] == -2.39e9 and notes is None


def test_debt_free_company_gets_roic_and_zero_leverage(cfg):
    raw = make_raw(cfg, n=2)
    for p in cfg.fiscal_years:
        raw.loc[0, f"Total Debt_{p}"] = np.nan  # Nintendo-like: no debt row at all
    df = calculate_metrics(raw, cfg)
    assert df.loc[0, "Calculated_Debt_to_Equity"] == 0
    assert pd.notna(df.loc[0, "ROIC_Norm"]) and df.loc[0, "ROIC_Norm"] > df.loc[1, "ROIC_Norm"]  # no debt: less capital


def test_share_classes_with_unequal_rights_use_economic_shares():
    from fq.fetch import economic_share_basis
    from fq.universe import share_classes

    assert share_classes()["0087.HK"] == ("0019.HK", 5.0)
    # Swire Pacific B: Yahoo counts 0.78bn A + 2.85bn B as 3.63bn equal shares; one A = five B.
    row = {"sharesOutstanding": 2.85e9, "currentPrice": 2.0, "marketCap": 2.0 * 3.63e9,
           "Ordinary Shares Number_Y0": 3.63e9, "Ordinary Shares Number_Y1": 3.70e9}  # fmt: skip
    factor = economic_share_basis(row, ["TTM", "Y0", "Y1"], "0087.HK", 5.0, 0.78e9)
    assert row["Ordinary Shares Number_Y0"] == pytest.approx(2.85e9 + 5 * 0.78e9)
    assert row["Ordinary Shares Number_Y1"] == pytest.approx(3.70e9 * factor)
    assert row["marketCap"] == pytest.approx(2.0 * 6.75e9)
    assert economic_share_basis({"sharesOutstanding": 1e9}, ["Y0"], "X", 5.0, None) is None


def test_rate_on_uses_last_quote_before_the_date(monkeypatch):
    hist = pd.Series([0.5, 0.4, 0.25], index=pd.to_datetime(["2024-12-30", "2024-12-31", "2025-12-31"]))
    monkeypatch.setitem(fx._history, "XXX", hist)
    assert fx.rate_on("XXX", "2025-06-30") == 0.4
    assert fx.rate_on("XXX", "2023-01-01") is None and fx.rate_on("USD", "2020-01-01") == 1.0


def test_exchange_rates_for_minor_units_and_unlisted_currencies(monkeypatch):
    from fq import fx

    monkeypatch.setattr(fx, "_history", {"XYZ": pd.Series([0.5, 0.25], index=pd.to_datetime(["2026-01-01", "2026-02-01"])),
                                         "NOPE": None})  # fmt: skip
    rates = {"USD": 1.0, "GBP": 1.3}
    assert fx.usd_rate(rates, "GBp") == pytest.approx(0.013)  # pence
    assert fx.usd_rate(rates, "XYZ") == 0.25 and rates["XYZ"] == 0.25  # fetched once, then kept
    assert fx.usd_rate(rates, "NOPE") is None  # unknown: never read as dollars


def test_a_cash_flow_jump_is_set_aside_but_a_recovery_is_not(cfg):
    from fq.metrics import clean_anomalies

    raw = make_raw(cfg, n=2)
    raw["TTM_Check_Gap"] = np.nan  # unconfirmed TTM
    raw.loc[0, "Operating Cash Flow_TTM"] = raw.loc[0, "Operating Cash Flow_Y0"] * 4  # one quarter of advances
    raw.loc[1, "Operating Cash Flow_Y0"] = raw.loc[1, "Operating Cash Flow_Y1"] * 0.2  # one bad year ...
    raw.loc[1, "Operating Cash Flow_TTM"] = raw.loc[1, "Operating Cash Flow_Y1"] * 1.1  # ... then back to normal
    out = clean_anomalies(raw, cfg)
    assert pd.isna(out.loc[0, "Operating Cash Flow_TTM"]) and pd.isna(out.loc[0, "Free Cash Flow_TTM"])
    assert "customer advances" in out.loc[0, "Anomaly_Notes"]
    assert out.loc[1, "Operating Cash Flow_TTM"] == raw.loc[1, "Operating Cash Flow_TTM"]


def test_price_stats_daily_turnover_in_todays_shares():
    from fq.fetch import price_stats

    days = pd.date_range("2025-10-20", "2026-10-16", freq="B")
    close = pd.DataFrame({"A": 10.0, "B": 5.0, "NOVOL": 3.0}, index=days)
    close.loc[days[-1], "A"] = 20.0  # today's close doubled
    volume = pd.DataFrame({"A": 1_000.0, "B": 0.0, "NOVOL": np.nan}, index=days)
    volume.loc[days[::2], "B"] = 400.0  # trades every other day
    stats = price_stats(close, close, volume)
    assert stats.at["A", "Daily_Turnover_Shares"] == pytest.approx(10_000 / 20)  # median value / today's close
    assert stats.at["B", "Daily_Turnover_Shares"] == pytest.approx(400)  # untraded days left out
    assert np.isnan(stats.at["NOVOL", "Daily_Turnover_Shares"])  # no volume: unknown, not illiquid
