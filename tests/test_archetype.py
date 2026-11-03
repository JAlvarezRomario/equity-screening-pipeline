"""Tests for automatic archetype detection and provenance tracking."""

from conftest import make_raw
from fq.metrics.archetype import detect_archetypes


def test_archetype_override_takes_precedence(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[0, "Ticker"] = "V"
    raw.loc[0, "industry"] = "Credit Services"
    raw.loc[1, "Ticker"] = "MSFT"
    raw.loc[1, "industry"] = "Software - Infrastructure"

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "operating"
    assert sources.iloc[0] == "override"
    assert models.iloc[1] == "operating"
    assert sources.iloc[1] == "industry"


def test_archetype_claims_signature(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Ticker"] = "MNG"
    raw.loc[0, "industry"] = "Asset Management"
    for p in ("TTM", cfg.latest):
        raw.loc[0, f"Total Revenue_{p}"] = 1000.0
        raw.loc[0, f"Net Policyholder Benefits And Claims_{p}"] = 350.0  # 35% of revenue

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "insurer"
    assert sources.iloc[0] == "signature:claims"


def test_archetype_lender_signature(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Ticker"] = "LEND"
    raw.loc[0, "industry"] = "Credit Services"
    for p in ("TTM", cfg.latest):
        raw.loc[0, f"Total Assets_{p}"] = 100000.0
        raw.loc[0, f"Stockholders Equity_{p}"] = 8000.0  # Assets > 12x equity
        raw.loc[0, f"Total Revenue_{p}"] = 5000.0  # Turnover 5% < 10%
        raw.loc[0, f"Net Interest Income_{p}"] = 2000.0  # NII 40% of revenue

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "bank"
    assert sources.iloc[0] == "signature:lender"


def test_archetype_fee_network_signature(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Ticker"] = "PAYNET"
    raw.loc[0, "industry"] = "Credit Services"  # base industry model is bank
    for p in ("TTM", cfg.latest):
        raw.loc[0, f"Total Assets_{p}"] = 5000.0
        raw.loc[0, f"Stockholders Equity_{p}"] = 2500.0  # Assets = 2x equity (not a bank)
        raw.loc[0, f"Total Revenue_{p}"] = 3000.0  # Turnover 60%
        raw.loc[0, f"Net Interest Income_{p}"] = 50.0  # NII < 2% of revenue

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "operating"
    assert sources.iloc[0] == "signature:fee_network"


def test_archetype_levered_broker_is_not_a_fee_network(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Ticker"] = "BROKER"
    raw.loc[0, "industry"] = "Capital Markets"  # base industry model is bank
    for p in ("TTM", cfg.latest):
        raw.loc[0, f"Total Assets_{p}"] = 65000.0
        raw.loc[0, f"Stockholders Equity_{p}"] = 10000.0  # assets 6.5x equity: client funds and margin loans
        raw.loc[0, f"Total Revenue_{p}"] = 2275.0  # 3.5% of assets
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = 27300.0  # 42% of assets, mostly clients'
        raw.loc[0, f"Net Interest Income_{p}"] = float("nan")  # not reported

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "bank"
    assert sources.iloc[0] == "industry"


def test_archetype_client_cash_signature(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Ticker"] = "CASHCO"
    raw.loc[0, "industry"] = "Software - Infrastructure"
    for p in ("TTM", cfg.latest):
        raw.loc[0, f"Total Assets_{p}"] = 10000.0
        raw.loc[0, f"Stockholders Equity_{p}"] = 2000.0
        raw.loc[0, f"Total Debt_{p}"] = 1000.0
        raw.loc[0, f"Cash And Cash Equivalents_{p}"] = 6000.0  # > 45% assets and > equity + debt

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "client_cash"
    assert sources.iloc[0] == "signature:client_cash"


def test_archetype_plant_signature(cfg):
    raw = make_raw(cfg, n=1)
    raw.loc[0, "Ticker"] = "HEAVY"
    raw.loc[0, "industry"] = "Specialty Industrial Machinery"
    for p in ("TTM", cfg.latest):
        raw.loc[0, f"Total Assets_{p}"] = 10000.0
        raw.loc[0, f"Net PPE_{p}"] = 7000.0  # 70% of assets
        raw.loc[0, f"Total Revenue_{p}"] = 2000.0  # Turnover 20% < 25%

    models, sources = detect_archetypes(raw, cfg)
    assert models.iloc[0] == "asset_heavy"
    assert sources.iloc[0] == "signature:plant"
