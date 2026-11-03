import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.filters import clean_and_filter
from fq.metrics import calculate_metrics, clean_anomalies, growth_profile
from fq.ranking import apply_dual_ranking, top_n
from fq.report.links import external_links, slug
from fq.universe import benchmark_part, industry_key, load_universe, resolve_selection


def test_settings_years_are_derived(cfg):
    assert cfg.annual_years == ["Y0", "Y1", "Y2", "Y3", "Y4"]
    assert cfg.fiscal_years[0] == "TTM" and cfg.chronological == ["Y4", "Y3", "Y2", "Y1", "Y0", "TTM"]
    assert (cfg.latest, cfg.prior, cfg.oldest) == ("Y0", "Y1", "Y4")


def test_universe_lists():
    full = load_universe("benchmark")
    assert full == sorted(set(full))
    assert {"GOOG", "AAPL", "ITX.MC", "0001.HK"} <= set(full) and len(load_universe("test")) == 8


def test_industry_key_ignores_dash_variants():
    assert industry_key("Banks—Regional") == industry_key("Banks - Regional") == "banksregional"


def test_slug_and_links():
    assert slug("BRK-B") == "BRK-B" and slug("A/B C") == "A_B_C"
    labels = [label for label, _ in external_links("0001.HK")]
    assert labels == ["Dataroma", "Yahoo Finance", "ROIC.ai"]
    assert all("0001.HK" in url for _, url in external_links("0001.HK"))


def test_growth_profile_classes():
    years = ["Y0", "Y1", "Y2", "Y3"]

    def row(vals):
        return pd.Series({f"X_{y}": v for y, v in zip(reversed(years), vals, strict=True)})  # oldest first

    assert growth_profile(row([1, 2, 3, 4]), "X", years)["status"] == "STRICT_GROWTH"
    assert growth_profile(row([4, 3, 2, 1]), "X", years)["status"] == "DECLINE"
    assert growth_profile(row([3, 1, 4, 1]), "X", years)["status"] == "CYCLICAL"
    assert growth_profile(row([1, 2, np.nan, np.nan]), "X", years)["status"] == "INSUFFICIENT_DATA"
    # SJM-like margins: a 0.09-point first step is flat, not the start of a run of rises.
    sjm = row([0.100, 0.1009, 0.138, 0.158])
    assert growth_profile(sjm, "X", years)["status"] == "STRICT_GROWTH"
    assert growth_profile(sjm, "X", years, step=0.005)["status"] != "STRICT_GROWTH"
    assert growth_profile(row([0.20, 0.17, 0.14, 0.11]), "X", years, step=0.005)["status"] == "DECLINE"


def test_anomaly_cleaning_blanks_impossible_revenue(cfg):
    raw = make_raw(cfg, n=2)
    raw["Total Debt_TTM"] = raw["Total Debt_Y1"]  # flat debt makes a revenue jump implausible
    raw.loc[0, "Total Revenue_TTM"] = raw.loc[0, "Total Revenue_Y1"] * 2.5
    cleaned = clean_anomalies(raw, cfg)
    assert pd.isna(cleaned.loc[0, "Total Revenue_TTM"])
    assert pd.notna(cleaned.loc[1, "Total Revenue_TTM"])


def test_metrics_and_ranking(cfg):
    df = calculate_metrics(make_raw(cfg), cfg)
    for col in ["FCFy_Norm", "ROIC_Norm", "Revenue_Growth_CAGR", "ROIC_StdDev", "operatingMargins_Avg", "EV"]:
        assert df[col].notna().all(), col
    # More FCF per revenue at similar EV means a higher normalised yield.
    assert df["FCFy_Norm"].corr(pd.Series(range(len(df)))) > 0

    kept, removed = clean_and_filter(df, list(df["Ticker"]))
    assert removed == [] and len(kept) == len(df)
    ranked = apply_dual_ranking(kept, cfg)
    assert list(ranked["Rank"]) == list(range(1, len(ranked) + 1))
    assert ranked["Combined_Rank_Score"].is_monotonic_increasing
    assert len(top_n(ranked, "Value_Rank", 3)) == 3


def test_filter_reasons(cfg):
    df = calculate_metrics(make_raw(cfg, n=3), cfg)
    df.loc[0, "industry"] = "Shell Companies"
    df.loc[1, "industry"] = np.nan
    kept, removed = clean_and_filter(df, list(df["Ticker"]))
    assert len(kept) == 1
    assert {t for t, _ in removed} == {df.loc[0, "Ticker"], df.loc[1, "Ticker"]}
    _, out_of_scope = clean_and_filter(df, [df.loc[2, "Ticker"]])
    assert len(out_of_scope) == 2


def test_resolve_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "portfolio.txt").write_text("jnj\nNESN.SW  # comment\n\nKO\n", encoding="utf-8")
    sel = resolve_selection("portfolio")
    assert (sel.label, sel.tickers) == ("portfolio", ["JNJ", "NESN.SW", "KO"])
    mixed = resolve_selection("portfolio, sap.de ,ko")
    assert mixed.label == "custom" and mixed.tickers == ["JNJ", "NESN.SW", "KO", "SAP.DE"]
    assert resolve_selection("aapl,msft").tickers == ["AAPL", "MSFT"]
    assert resolve_selection("benchmark").label == "benchmark" and "GOOG" in resolve_selection("benchmark").tickers
    (tmp_path / "mine.txt").write_text("AAA\nBBB\n", encoding="utf-8")
    assert resolve_selection("mine.txt").tickers == ["AAA", "BBB"]


def test_holdings_are_always_in_the_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    without = load_universe("benchmark")
    assert "5425.T" not in without  # excluded_tickers.txt
    (tmp_path / "portfolio.txt").write_text("ZZZZ.XX\n5425.T\nAAPL\n", encoding="utf-8")
    others = benchmark_part("others")
    assert {"ZZZZ.XX", "5425.T"} <= set(others) and "AAPL" not in others  # AAPL stays in its index part
    assert set(load_universe("benchmark")) == set(without) | {"ZZZZ.XX", "5425.T"}


def test_missing_portfolio_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="portfolio.txt"):
        resolve_selection("portfolio")
    with pytest.raises(ValueError):
        resolve_selection(" , ")


def test_run_labels_separate_outputs(cfg):
    from dataclasses import replace

    a, b = replace(cfg, label="portfolio"), replace(cfg, label="benchmark")
    assert a.site_dir != b.site_dir and a.ranked_path != b.ranked_path


def test_nav_flag_and_protected_industries(cfg):
    raw = make_raw(cfg, n=3)
    for y in cfg.fiscal_years:  # ticker 0: tangible book far above the share price
        raw.loc[0, f"Tangible Book Value_{y}"] = 2e10
        raw.loc[1, f"Tangible Book Value_{y}"] = 1e8
    raw.loc[2, "industry"] = "Shell Companies"
    df = calculate_metrics(raw, cfg)
    assert list(df["Show_NAV_Chart"]) == [True, False, False]
    assert df.loc[0, "NAV_to_Price"] > 100 > df.loc[1, "NAV_to_Price"]
    assert (df["BV_to_Price"] > 0).all()

    scope = list(df["Ticker"])
    kept, removed = clean_and_filter(df, scope)
    assert [t for t, _ in removed] == [df.loc[2, "Ticker"]]
    kept, removed = clean_and_filter(df, scope, protected={df.loc[2, "Ticker"]})
    assert removed == [] and kept["Industry_Excluded_Kept"].sum() == 1


def test_untrusted_rows_follow_every_trusted_row_in_the_combined_order(cfg):
    from fq.ranking import trusted

    df = calculate_metrics(make_raw(cfg), cfg)
    plain = apply_dual_ranking(df, cfg)
    best = plain.loc[plain["Rank"] == 1, "Ticker"].iloc[0]
    flagged = df.assign(Data_Suspect=df["Ticker"] == best)
    ranked = apply_dual_ranking(flagged, cfg)
    scored = ranked[ranked["Rank"].notna()]
    ok = trusted(scored)
    assert ok.any() and (~ok).any()
    assert scored.loc[ok, "Rank"].max() < scored.loc[~ok, "Rank"].min()
    row = ranked.set_index("Ticker").loc[best]
    assert row["Value_Score"] == plain.set_index("Ticker").loc[best, "Value_Score"]  # scores unchanged


def test_accounting_flag_marks_earnings_ahead_of_cash(cfg):
    from fq.metrics.flags import _accounting_flag

    df = pd.DataFrame({"Accruals": [12.0, 6.0, 6.0, np.nan, -3.0], "Beneish_M": [-3.0, -1.5, -2.5, -1.0, np.nan]})
    _accounting_flag(df, cfg)
    assert df["Accounting_Flag"].tolist() == [True, True, False, False, False]
    assert "accruals above" in df.loc[0, "Accounting_Reasons"] and "Beneish" in df.loc[1, "Accounting_Reasons"]
    assert df.loc[2:, "Accounting_Reasons"].isna().all()
