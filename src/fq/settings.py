"""Central configuration. Importing this module performs no I/O."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Bump when column names or their meaning change; `fq rank` refuses caches from another version.
SCHEMA_VERSION = 7  # 7: lease liabilities fetched (leases are an operating cost, not debt)

# Every run (portfolio, custom tickers) is scored against this one global set: S&P 500 + IBEX 35 + DAX +
# EURO STOXX 50 + FTSE 100 + STI + Hang Seng (Hong Kong companies only) +
# FBM KLCI + other markets (`universe.BENCHMARK_SETS`), so all reports share one scale. `--benchmark none` ranks a
# group on its own.
DEFAULT_BENCHMARK = "benchmark"
# Labels that are reference sets themselves and are never benchmarked.
SELF_BENCHMARKED = frozenset({"benchmark"})
# Share of the benchmark that must be cached to use it (else the group is ranked on its own).
BENCHMARK_MIN_CACHED = 0.8
# How run and benchmark labels read in the reports.
DISPLAY_NAMES: dict[str, str] = {
    "sp500": "S&P 500", "ibex35": "IBEX 35", "dax": "DAX", "eurostoxx50": "EURO STOXX 50", "ftse100": "FTSE 100", "sti": "STI", "hsi_hk": "Hang Seng (HK companies)", "klci": "FBM KLCI", "obx": "OBX", "omxs30": "OMXS30", "wig20": "WIG20", "asx50": "ASX 50", "msci_world": "MSCI World", "msci_em": "MSCI ACWI emerging markets",
    "bist": "BIST 100", "india": "NIFTY 100",
    "others": "Other markets",
    "benchmark": "Global", "portfolio": "Portfolio", "custom": "This run", "test": "Test set",
}  # fmt: skip


def display_name(label: str) -> str:
    """'benchmark' -> 'Global'; unknown labels as they are."""
    return DISPLAY_NAMES.get(label, label)


# Display-name fixes for tickers where Yahoo returns a poor shortName.
NAME_OVERRIDES: dict[str, str] = {
    "LOG.MC": "Logista Integral S.A.",
    "CEV.MC": "Compañía Española de Viviendas en Alquiler S.A.",
    "PG": "The Procter & Gamble Company",
    "ADI": "Analog Devices Inc",
    "0001.HK": "CK Hutchison Holdings Limited",
    "EC4.DE": "Ecotel Communication AG",
    "MBB.DE": "MBB SE",
    "VAR.OL": "Vår Energi ASA",
    # Warsaw: Yahoo's shortName is a code ("MBK C.XWAR PLN").
    "ALE.WA": "Allegro.eu S.A.",
    "ALR.WA": "Alior Bank S.A.",
    "BDX.WA": "Budimex S.A.",
    "CDR.WA": "CD Projekt S.A.",
    "DNP.WA": "Dino Polska S.A.",
    "EBP.WA": "Erste Bank Polska S.A.",
    "KGH.WA": "KGHM Polska Miedź S.A.",
    "KRU.WA": "KRUK S.A.",
    "KTY.WA": "Grupa Kęty S.A.",
    "LPP.WA": "LPP S.A.",
    "MBK.WA": "mBank S.A.",
    "MDV.WA": "Modivo S.A.",
    "PCO.WA": "Pepco Group N.V.",
    "PEO.WA": "Bank Polska Kasa Opieki S.A.",
    "PGE.WA": "PGE Polska Grupa Energetyczna S.A.",
    "PKN.WA": "ORLEN S.A.",
    "PKO.WA": "PKO Bank Polski S.A.",
    "PZU.WA": "Powszechny Zakład Ubezpieczeń S.A.",
    "TPE.WA": "Tauron Polska Energia S.A.",
    "ZAB.WA": "Żabka Group S.A.",
    # pl_midsmall.txt (names from Yahoo's longName, accents dropped):
    "1AT.WA": "Atal S.A.",
    "ABE.WA": "AB S.A.",
    "ABS.WA": "Asseco Business Solutions S.A.",
    "AMB.WA": "Ambra S.A.",
    "ANR.WA": "Answear.com S.A.",
    "APT.WA": "Apator S.A.",
    "ARH.WA": "Archicom S.A.",
    "ASE.WA": "Asseco South Eastern Europe S.A.",
    "AST.WA": "Astarta Holding PLC",
    "ATT.WA": "Grupa Azoty S.A.",
    "BCX.WA": "Bioceltix S.A.",
    "BFT.WA": "Benefit Systems S.A.",
    "BHW.WA": "Bank Handlowy w Warszawie S.A.",
    "BLO.WA": "Bloober Team SA",
    "BNP.WA": "BNP Paribas Bank Polska S.A.",
    "BOS.WA": "Bank Ochrony Srodowiska S.A.",
    "BRS.WA": "Boryszew S.A.",
    "BST.WA": "BEST S.A.",
    "CBF.WA": "cyber_Folks S.A.",
    "CIG.WA": "CI Games SE",
    "CLD.WA": "Cloud Technologies S.A.",
    "CLN.WA": "Celon Pharma S.A.",
    "COG.WA": "Cognor Holding S.A.",
    "CRQ.WA": "Creotech Quantum S.A.",
    "DAD.WA": "Dadelo S.A.",
    "DAT.WA": "DataWalk S.A.",
    "DBC.WA": "Tire Company Debica S.A.",
    "DCR.WA": "Decora S.A.",
    "DEK.WA": "Dekpol S.A.",
    "DIA.WA": "Diagnostyka S.A.",
    "DIG.WA": "Digital Network SA",
    "DOM.WA": "Dom Development S.A.",
    "DVL.WA": "Develia S.A.",
    "ECH.WA": "Echo Investment S.A.",
    "ELT.WA": "Elektrotim S.A.",
    "ENA.WA": "ENEA S.A.",
    "ENE.WA": "Centrum Medyczne ENEL-MED S.A.",
    "ENT.WA": "Enter Air S.A.",
    "EQU.WA": "Equnico SE",
    "EUR.WA": "Eurocash S.A.",
    "GNS.WA": "Niewiadow Polska Grupa Militarna S.A.",
    "GOB.WA": "Gobarto S.A.",
    "GPP.WA": "Grupa Pracuj S.A.",
    "GPW.WA": "Gielda Papierow Wartosciowych w Warszawie S.A.",
    "GTC.WA": "Globe Trade Centre S.A.",
    "HUG.WA": "Huuuge, Inc.",
    "IMC.WA": "IMC S.A.",
    "ING.WA": "ING Bank Slaski S.A.",
    "JSW.WA": "Jastrzebska Spolka Weglowa S.A.",
    "KER.WA": "Kernel Holding S.A.",
    "KGN.WA": "Zespol Elektrocieplowni Wroclawskich KOGENERACJA S.A.",
    "LBW.WA": "Lubawa S.A.",
    "LKD.WA": "Lokum Deweloper S.A.",
    "LWB.WA": "Lubelski Wegiel Bogdanka S.A.",
    "MBR.WA": "Mo-BRUK S.A.",
    "MGT.WA": "Mangata Holding S.A.",
    "MLG.WA": "MLP Group S.A.",
    "MNC.WA": "Mennica Polska S.A.",
    "MRB.WA": "Mirbud S.A.",
    "MRC.WA": "Mercator Medical S.A.",
    "MSZ.WA": "Mostostal Zabrze S.A.",
    "NTT.WA": "NTT System S.A.",
    "NWG.WA": "Newag S.A.",
    "OBL.WA": "Orzel Bialy S.A.",
    "OND.WA": "Onde S.A.",
    "OPL.WA": "Orange Polska S.A.",
    "OPN.WA": "Oponeo.pl S.A.",
    "PCE.WA": "Grupa Azoty Zaklady Chemiczne Police S.A.",
    "PCR.WA": "PCC Rokita SA",
    "PHN.WA": "Polski Holding Nieruchomosci S.A.",
    "PKP.WA": "Pkp Cargo S.A.",
    "PLW.WA": "PlayWay S.A.",
    "PXM.WA": "Polimex-Mostostal S.A.",
    "QRS.WA": "Quercus TFI S.A.",
    "RAF.WA": "Fabryka Obrabiarek RAFAMET S.A.",
    "RBW.WA": "Rainbow Tours S.A.",
    "ROB.WA": "Robyg S.A.",
    "RVU.WA": "Ryvu Therapeutics S.A.",
    "RWL.WA": "Rawlplug S.A.",
    "S2B.WA": "Syn2bio S.A.",
    "SCP.WA": "Scope Fluidics S.A.",
    "SCW.WA": "Scanway S.A.",
    "SEL.WA": "Selena FM S.A.",
    "SGN.WA": "Sygnity S.A.",
    "SKA.WA": "Fabryka Farb i Lakierow Sniezka SA",
    "SNK.WA": "Sanok Rubber Company Spolka Akcyjna",
    "SNT.WA": "Synektik Spolka Akcyjna",
    "STP.WA": "Stalprodukt S.A.",
    "TAR.WA": "Tarczynski S.A.",
    "TEN.WA": "Ten Square Games S.A.",
    "TOA.WA": "TOYA S.A.",
    "TOR.WA": "Torpol S.A.",
    "TRK.WA": "Trakcja S.A.",
    "UNI.WA": "UNIBEP S.A.",
    "UNT.WA": "Unimot S.A.",
    "VGO.WA": "VIGO Photonics S.A.",
    "VOT.WA": "Votum S.A.",
    "VOX.WA": "Voxel S.A.",
    "VRC.WA": "Vercom S.A.",
    "VRG.WA": "VRG S.A.",
    "WAS.WA": "Wasko S.A.",
    "WPL.WA": "Wirtualna Polska Holding S.A.",
    "WWL.WA": "Wawel S.A.",
    "XTB.WA": "XTB S.A.",
    "ZAP.WA": "Zaklady Azotowe Pulawy S.A.",
    "ZEP.WA": "ZE PAK SA",
}

# Tickers where Yahoo's financialCurrency is misreported (e.g. reporting local statements tagged as "USD").
FINANCIAL_CURRENCY_OVERRIDES: dict[str, str] = {
    "HCLTECH.NS": "INR",
    "PTTEP.BK": "THB",
}

INFO_KEYS: list[str] = [
    "shortName", "dividendYield", "forwardPE", "marketCap", "currency",
    "enterpriseValue", "priceToBook", "pegRatio", "financialCurrency",
    "trailingPE", "heldPercentInsiders", "operatingCashflow",
    "operatingMargins", "returnOnEquity", "profitMargins", "industry",
    "sector", "country", "numberOfAnalystOpinions",
    "debtToEquity", "freeCashflow", "heldPercentInstitutions",
    "targetLowPrice", "previousClose", "currentPrice",
    "longBusinessSummary", "fullTimeEmployees", "beta",
    "sharesOutstanding", "impliedSharesOutstanding", "earningsTimestamp",
    "website",
]  # fmt: skip

CONCEPTS_INCOME = [
    "Total Revenue", "Net Income", "Operating Income", "EBIT",
    "Tax Provision", "Pretax Income", "Interest Expense", "Gross Profit",
    "Net Income Including Noncontrolling Interests",  # / net income = the parent's share of the group
    "Interest Income",  # with Interest Expense: today's financing cost (`metrics.flows._financing`)
    "Net Interest Income",  # archetype signature for lenders vs fee networks
    "Total Unusual Items", "Normalized Income",  # one-offs (pre-tax) and net income without them
    "Net Policyholder Benefits And Claims",  # only insurers report it (`metrics.flags._model_review`: MNG, LGEN)
    "Cost Of Revenue", "Selling General And Administration",  # Beneish M-score (`metrics.quality._beneish`)
    # Average share counts of the period, diluted (options, convertibles) and basic (Nov 4, 2026): stored, not yet
    # read by any metric; `AVERAGE_CONCEPTS`, `SHARE_COUNT_CONCEPTS`.
    "Diluted Average Shares", "Basic Average Shares",
]  # fmt: skip

CONCEPTS_BALANCE = [
    "Stockholders Equity", "Invested Capital", "Net Debt",
    "Cash And Cash Equivalents", "Short Term Investments",
    "Other Short Term Investments", "Cash Cash Equivalents And Short Term Investments",
    "Total Debt", "Ordinary Shares Number", "Share Issued",
    "Minority Interest", "Total Equity Gross Minority Interest",
    "Preferred Stock", "Preferred Stock Equity", "Net PPE",
    "Tangible Book Value", "Goodwill And Other Intangible Assets", "Common Stock Equity",
    "Total Assets",
    "Capital Lease Obligations", "Current Capital Lease Obligation",  # inside Total Debt (`metrics.capital._leases`)
    # Unrealised securities gains / losses in equity (`metrics.capital._book_per_share`): IFRS row, else US GAAP AOCI.
    "Unrealized Gain Loss", "Gains Losses Not Affecting Retained Earnings",
    # Altman Z'' (`metrics.flags._distress`) and Beneish M-score (`metrics.quality._beneish`).
    "Current Assets", "Current Liabilities", "Retained Earnings", "Total Liabilities Net Minority Interest",
    "Receivables",
    # Non-cash working capital = current assets - cash - (current liabilities - this) (`flows._growth_working_capital`).
    "Current Debt And Capital Lease Obligation",
]  # fmt: skip

CONCEPTS_CASHFLOW = [
    "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
    "Stock Based Compensation", "Repurchase Of Capital Stock", "Repayment Of Debt",
    "Issuance Of Debt", "Issuance Of Capital Stock", "Cash Dividends Paid",
    "Depreciation And Amortization", "Depreciation", "Purchase Of PPE",
    "Depreciation Amortization Depletion", "Reconciled Depreciation",
    "Net Business Purchase And Sale",  # acquisitions (negative) net of disposals
    "Interest Paid Cff",  # IFRS: interest paid shown under financing, so not yet out of free cash flow
    "Interest Received Cfi",  # IFRS: interest received shown under investing, so never in free cash flow
    "Change In Working Capital",  # caps the working capital growth tied up (`flows._growth_working_capital`)
]  # fmt: skip

# Yahoo renames or drops rows over time. The first name present (and non-null) wins.
ALIASES: dict[str, list[str]] = {
    "Total Revenue": ["Operating Revenue"],
    "Operating Income": ["Total Operating Income As Reported"],
    "Interest Expense": ["Interest Expense Non Operating"],
    "Interest Income": ["Interest Income Non Operating"],
    "Net Interest Income": ["Net Non Operating Interest Income"],
    "Stockholders Equity": ["Common Stock Equity"],
    "Tangible Book Value": ["Net Tangible Assets"],
    "Cash Dividends Paid": ["Common Stock Dividend Paid"],
    "Repurchase Of Capital Stock": ["Common Stock Payments"],
    "Issuance Of Capital Stock": ["Common Stock Issuance"],
    "Capital Expenditure": ["Net PPE Purchase And Sale"],
    "Net Business Purchase And Sale": ["Purchase Of Business"],
    "Receivables": ["Accounts Receivable"],
    "Current Debt And Capital Lease Obligation": ["Current Debt"],
    "Cost Of Revenue": ["Reconciled Cost Of Revenue"],
}
# Concepts that can be rebuilt as the sum of other rows when absent.
COMPOSITES: dict[str, list[str]] = {
    "Total Debt": ["Long Term Debt And Capital Lease Obligation", "Current Debt And Capital Lease Obligation"],
    "Capital Lease Obligations": ["Long Term Capital Lease Obligation", "Current Capital Lease Obligation"],
}
# The newest quarters kept one by one (`<concept>_Q0` newest .. `_Q{N_QUARTERS-1}`, `PeriodEnd_Q*`), so a profit
# fall that lasts can be told from one bad quarter and given its run-rate (`metrics.normalise.earnings_collapse`).
QUARTERLY_CONCEPTS = ["Total Revenue", "Pretax Income", "Net Income", "Total Unusual Items", "Interest Expense"]
N_QUARTERS = 5
QUARTERS = [f"Q{i}" for i in range(N_QUARTERS)]
# Long per-share series fetched besides the statements (`fetch.yearly_shares`, `fetch.yearly_dividends`; shown only),
# and each dividend payment (`fetch.dividend_payments`, Nov 4, 2026: the Unbroken dividends list).
HISTORY_SERIES = ("shares_full", "dividends", "dividend_payments")
# Fingerprint of what a fetch collects. Stored per company (`Fields_Version`): a company fetched with
# another set is due for a full fetch (`refresh.reason`: "new fields"), so adding a row here reaches the
# cache gradually, within `max_full_fetches` per run, instead of needing `--full`.
FIELDS_VERSION = hashlib.sha1(
    repr(
        (INFO_KEYS, CONCEPTS_INCOME, CONCEPTS_BALANCE, CONCEPTS_CASHFLOW, ALIASES, COMPOSITES, QUARTERLY_CONCEPTS,
         N_QUARTERS, HISTORY_SERIES)
    ).encode()
).hexdigest()[:8]  # fmt: skip
# If fewer than CRITICAL_MIN_COVERAGE of tickers have these, the API has probably changed.
CRITICAL_CONCEPTS = [
    "Total Revenue", "Operating Income", "Net Income", "Operating Cash Flow", "Free Cash Flow",
    "Stockholders Equity", "Total Debt", "Ordinary Shares Number",
]  # fmt: skip
CRITICAL_MIN_COVERAGE = 0.6

# Counts / ratios that must never be multiplied by an FX rate.
NON_CURRENCY_CONCEPTS = frozenset(
    {
        "Ordinary Shares Number",
        "Share Issued",
        "Diluted Average Shares",
        "Basic Average Shares",
        "Full Time Employees",
        "numberOfAnalystOpinions",
        "heldPercentInsiders",
        "heldPercentInstitutions",
    }
)
# Income-statement rows that are averages over the period, not flows: their TTM is the mean of the four quarters.
AVERAGE_CONCEPTS = frozenset({"Diluted Average Shares", "Basic Average Shares"})
# Every stored share count: split, ADR and share-class adjustments apply to all of them (`fetch._apply_splits`,
# `_align_share_basis`, `economic_share_basis`), so they stay on one basis.
SHARE_COUNT_CONCEPTS = ("Ordinary Shares Number", "Share Issued", "Diluted Average Shares", "Basic Average Shares")

# [metric, "top" (higher is better) | "bottom" (lower is better), weight]
Criterion = tuple[str, str, float]

# Value is per share (user's rule): every yield is the shareholders' part of a flow before interest / their
# share of the price paid for the operations (market cap + their share of debt and preferred - spare cash,
# `metrics.capital._shareholder_base`), so cash and debt count at face and a company is neither cheaper for borrowing
# nor dearer for holding cash (Greenwald). Leverage risk is judged once, in Quality. Payout (dividends +
# buybacks) is shown only: it is a use of the FCF already counted (share reduction is a shown percentile too).
# Value is a single owner-earnings yield before interest over `Value_Base` (user's rule, Oct 26, 2026): normalised
# owner earnings at zero real growth, over market cap plus debt, less spare cash. No projection (user's rule,
# Nov 8, 2026; was 3 years forward on the trend): growth is Quality's, counted once.
VALUE_CRITERIA: tuple[Criterion, ...] = (("Owner_Yield", "top", 1.0),)

# Quality (user's rule, Oct 26, 2026): return on capital, normalised ROIC (operating) or ROE (financials), adjusted
# for its trend (capped at +/-10 pp), shrunk for variability (unsteadiness), with a haircut for severe leverage.
# Since Oct 28, 2026 (user's choice: what a moat and management do is not measurable, what they leave a long-term
# holder is) also what one share of the business grew by (revenue per share in US terms since Nov 6, 2026, book per
# share for financials, trend x adjusted R²) and whether the share count shrank, both the same question for every
# model, so market-wide.
# `Return_Quality` alone (0.7) still meets `min_coverage`.
QUALITY_CRITERIA: tuple[Criterion, ...] = (
    ("Return_Quality", "top", 0.7),
    ("Per_Share_Compounding", "top", 0.2),
    ("Share_Reduction_Rate", "top", 0.1),
)
QUALITY_MARKET_CRITERIA: tuple[str, ...] = ("Per_Share_Compounding", "Share_Reduction_Rate")

# Criteria shown on company pages or used by highlight lists (Compounders, Asset Plays, etc.)
# scored as percentiles (Pct_*) but not part of the primary Value/Quality scores.
SHOWN_PERCENTILES: tuple[tuple[str, str], ...] = (
    ("ROIC_Norm", "top"),
    ("ROIC_Variability", "bottom"),
    ("Per_Share_Compounding", "top"),
    ("Share_Reduction_Rate", "top"),
    ("Revenue_Worst_Change", "top"),
    ("Gross_Margin_StdDev", "bottom"),
    ("Gross_Profitability_Avg", "top"),
    ("Incremental_ROIC", "top"),
    ("ROIC_Trend", "top"),
    ("NetDebt_to_EBIT", "bottom"),
    ("FCF_Conversion", "top"),
    ("SBC_to_Revenue", "bottom"),
    ("Equity_to_Assets", "top"),
    ("ROA_Avg", "top"),
    ("EPV_Yield", "top"),
    ("Aftertax_Yield_Avg", "top"),
)
SHOWN_MARKET_CRITERIA: tuple[str, ...] = (
    "ROIC_Variability",
    "Share_Reduction_Rate",
    "Revenue_Worst_Change",
    "Gross_Margin_StdDev",
    "SBC_to_Revenue",
)

# Proxies under observation: reported and given percentiles (`Cand_Pct_*`, in the snapshots and in the history
# test `history.quality_ic`) but never part of a score. One joins QUALITY_CRITERIA when the reasoning for it
# holds and it is not a copy of ROIC_Norm (< 0.7).
QUALITY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("heldPercentInsiders", "top"),  # skin in the game (a controlling family can also mistreat minorities)
    ("Asset_Growth", "bottom"),  # capital cycle: fast asset growers tend to disappoint (Cooper-Gulen-Schill)
    ("Accruals", "bottom"),  # earnings ahead of cash: (net income - operating cash flow) / assets, %
    ("Beneish_M", "bottom"),  # earnings manipulation (Beneish 8-variable M-score, latest year vs the one before)
    ("ROIC_Ex_Goodwill", "top"),  # the operating moat without acquisitions' goodwill (ROIC with it is M&A discipline)
)

FINANCIAL_VALUE_CRITERIA: tuple[Criterion, ...] = VALUE_CRITERIA
FINANCIAL_QUALITY_CRITERIA: tuple[Criterion, ...] = QUALITY_CRITERIA

# Business models (`lists/business_models.txt`, `lists/model_overrides.txt`) and how they are measured:
# model -> (peer group whose distribution its percentiles come from, family = the fallback group when
# the peer group is too small). Operating and client-cash companies share one distribution.
BUSINESS_MODELS: dict[str, tuple[str, str]] = {
    "operating": ("operating", "operating"),
    "client_cash": ("operating", "operating"),  # payroll, clearing, payments: client cash is not spare cash
    "float": ("operating", "operating"),  # health insurers: also 'FCF' = net income (reserve swings)
    "asset_heavy": ("asset_heavy", "asset_heavy"),  # REITs, utilities: growth capex, real debt
    "bank": ("bank", "financial"),
    "insurer": ("insurer", "financial"),
    "holding": ("insurer", "financial"),  # Berkshire-style: insurance float plus owned businesses
}
FINANCIAL_MODELS = frozenset({"bank", "insurer", "holding"})
# Models whose net income holds investment gains Yahoo does not separate (no net investment income row; a gains row
# for 1 of 18 insurers and holdings checked, none for Berkshire): Value takes their mid-cycle level.
GAINS_INSIDE_MODELS = frozenset({"insurer", "holding"})
# Currencies of IAS 29 (hyperinflationary) economies: listed operating companies there restate their statements in
# today's money, so their returns are already real; banks and insurers are exempt (Turkey: BRSA, SEDDK) and report
# nominal figures. `metrics.inflation._basis` reads it; `lists/inflation_basis.txt` overrides per ticker.
HYPERINFLATION_CURRENCIES = frozenset({"TRY", "ARS", "VES", "LBP", "IRR", "ETB", "SLE", "GHS", "HTG", "SSP", "ZWG"})


@dataclass(frozen=True)
class Settings:
    """Run-time settings.

    Periods are relative to each company: `TTM`, then `Y0` (its latest annual report), `Y1`
    (one year earlier), ... `Y{n_years-1}`. No calendar year appears anywhere, so the code keeps
    working whatever Yahoo currently serves; slots Yahoo does not provide are simply empty.
    """

    n_years: int = 5
    # Auto refresh (refresh.py): fundamentals older than this are re-fetched (and health warns).
    max_cache_age_days: int = 90
    report_lag_days: int = 45  # quarter end -> statements on Yahoo
    annual_report_lag_days: int = 90  # fiscal year end -> annual report on Yahoo
    earnings_settle_days: int = 2  # earnings call -> statements on Yahoo
    incomplete_retry_days: int = 7  # retry companies with missing TTM / critical data after this
    price_refresh_days: int = 7  # prices newer than this are kept (user's choice: a week moves little)
    # Full fetches per run (0 = no cap), by priority (`refresh.select`); the run's own new tickers are
    # always fetched and count toward it. ~5 s each: 60 adds at most ~5 minutes.
    max_full_fetches: int = 60
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    label: str = "benchmark"  # run name: separates cache and output per selection
    fetch_delay_s: float = 1.0
    fetch_retries: int = 3
    # Tickers fetched at once (threads, each pausing `fetch_delay_s`); a rate limit drops to one for the
    # rest of the batch. Yahoo answered 8 threads with HTTP 429 on Oct 29, 2026.
    fetch_workers: int = 3
    # Multiplier applied to the latest FCF margin when margins expand every year.
    margin_expansion_premium: float = 1.07
    # TTM clean profit before financing more than this % below the latest annual report: the normal level of every
    # Value yield is at most the TTM (`normalise.earnings_collapse`; user's choice, Oct 20, 2026).
    earnings_collapse_pct: float = 25.0
    # Show the NAV/book chart when book value OR adjusted tangible book per share is at least this % of price.
    nav_chart_threshold_pct: float = 40.0
    # ROIC tax: effective rates outside the bounds (losses, credits, one-offs) use the notional rate.
    notional_tax_rate: float = 0.21
    tax_rate_bounds: tuple[float, float] = (0.05, 0.40)
    # Operating-margin coefficient of variation above which a company is flagged Cyclical (also when
    # EBIT changes sign). The margin, not EBIT, so a fast but steady grower is not flagged.
    cyclical_margin_cv: float = 0.5
    # Peak_Margin (`metrics.flags._peak_margin`): a cyclical whose latest operating margin is above this multiple
    # of its `peak_margin_quantile` annual margin (at least `peak_margin_floor` points) leaves the top lists.
    peak_margin_ratio: float = 1.5
    peak_margin_quantile: float = 0.3
    peak_margin_floor: float = 2.0
    # Distress (`metrics.flags._distress`): Altman Z'' (non-manufacturer) below this AND negative FCF or EBIT below
    # `distress_interest_cover` x interest. Still scored, out of the top lists and Dislocation.
    distress_z: float = 1.1
    distress_interest_cover: float = 1.5
    # Revenue_Decline (`metrics.flags._revenue_decline`): real USD revenue trend at or below this (%/yr) with an
    # adjusted R² of at least `revenue_decline_min_r2`, or a fall in every stored year over at least
    # `revenue_decline_min_falls` steps averaging that rate. Still scored, out of the highlight lists (`ranking.UNTRUSTED_FLAGS`).
    revenue_decline_rate: float = -3.0
    revenue_decline_min_r2: float = 0.5
    revenue_decline_min_falls: int = 3
    # Ranking: minimum share of criterion weight a company must have data for, and winsorizing tails.
    # Above 0.7 no Value goes without earnings power (5 of 7), which alone is enough (0.71).
    min_coverage: float = 0.7
    winsor_pct: float = 0.01
    # Cash a business needs to operate (% of revenue); only the rest is spare and counts at face in Value.
    operating_cash_pct: float = 2.0
    # Debt counts at face in Value as far as its interest left the flows (`metrics.capital._value_base`): fully at an
    # implied rate of at least this %, in proportion below it (5% of companies with debt pay under 1.8%).
    debt_cost_floor: float = 1.5
    # Value percentiles: this share within the company's Yahoo sector, the rest across the market; sectors with
    # fewer reference companies use the market only. 0 = market only (user's choice, Oct 20, 2026: the same owner
    # yield is the same Value in any sector; why a sector is cheap is Quality's question). Was 0.5.
    sector_blend: float = 0.0
    sector_min_members: int = 15
    # Business models that take ranks, enter the cross-model (market) distribution, the top lists and the
    # Dislocation flag. User's choice (Oct 20, 2026): every model, one yardstick. Value = the normalised earnings
    # that belong to a share / its price; Quality = how likely they are to last and grow. A model left out is
    # scored against its peer group and shown, but unranked (`fq evaluate` keeps per-model ICs).
    ranked_models: tuple[str, ...] = (
        "operating", "client_cash", "float", "asset_heavy", "bank", "insurer", "holding",
    )  # fmt: skip
    # Scored with the market as above, but no position in the ranks (`ranking.rank_group`; user's choice, Nov 5, 2026:
    # a lender's or a holding's yardstick is not an operating company's, so the rankings hold profitable operating
    # companies only; these have lists of their own). Equity REITs leave by industry (`lists/real_estate_owners.txt`).
    rank_apart_models: tuple[str, ...] = ("bank", "insurer", "holding")
    # Asset-heavy companies: capex up to depreciation keeps the assets (maintenance, deducted from FCF);
    # the rest buys growth and is left out of earnings power.
    maintenance_capex_to_da: float = 1.0
    # Owner earnings (`metrics.flows._organic_growth_capex`): models whose capex is split Greenwald's way into
    # maintenance (charged for ever) and growth (charged while the company grows). () = all capex charged.
    organic_capex_models: tuple[str, ...] = ("operating", "client_cash")
    # Owner earnings are at zero REAL growth (both owner-earnings proposals reviewed Oct 20, 2026): revenue growth up
    # to this rate (USD) is price, not volume, so the capex and working capital it needs stay charged
    # (`flows._organic_growth_capex`, `flows._growth_working_capital`), and banks, insurers and holdings keep this
    # share of their book to hold the balance sheet flat in real terms (`flows._capital_retention`). A fixed
    # assumption; not tuned.
    inflation: float = 0.025
    # Model families whose Value is also judged against the market (the ranked models), not only their peer
    # group. Banks and insurers are not: their (former) implied return is on equity (levered) and ran 3-4 pp above
    # operating companies' (on EV) on the S&P 500 (median 8.6-9.6% vs 5.4%), so a cross-model comparison
    # would call every lender cheap. Asset-heavy medians matched (5.1%). Tried with every model ranked (Oct 20,
    # 2026): median Value banks 72, insurers 65, operating 47; 8 of the top 20 Value lenders or insurers.
    value_market_families: tuple[str, ...] = ("operating", "asset_heavy")
    # Data_Suspect thresholds: implausibly cheap, or TTM sources disagreeing.
    suspect_fcf_yield: float = 25.0
    suspect_pretax_yield: float = 40.0
    suspect_ttm_gap: float = 0.10
    suspect_growth_yield: float = 15.0  # FCF yield above this with growth at the cap: likely a one-off
    # `Accounting_Flag` (`flags._accounting_flag`, Oct 30, 2026): earnings ahead of cash, out of the highlight lists.
    accounting_max_accruals: float = 10.0  # median accruals, % of assets
    accounting_max_beneish: float = -1.78  # Beneish's threshold, only together with accruals above the next
    accounting_beneish_accruals: float = 5.0
    # Highlight lists leave out companies whose median daily traded value (`Traded_Value_USD`, 1 year) is below this,
    # in USD: a slice nobody can buy beats nothing (Oct 30, 2026). A missing figure passes.
    list_min_traded_usd: float = 100_000.0
    # `Growth_Used` (which capex buys growth; the suspect-yield flag): USD revenue slope, the lower of total and per
    # share, clipped, halved for cyclicals.
    growth_floor: float = -5.0
    growth_cap: float = 15.0
    cyclical_growth_factor: float = 0.5
    # Acquisitions (net cash, averaged over the latest `acquisition_years` annual reports), shown.
    acquisition_years: int = 4
    # Dislocation ("mood") flag, not ranked and blind to the share price path: Value of at least
    # `dislocation_min_value` (cheap), Quality of at least `dislocation_min_quality`, EBIT and revenue
    # held (recent change above the minimums, in %) and analysts did not cut next year's EPS by more
    # than `estimate_cut_pct` over 90 days: nothing in the numbers explains the low price. Cyclicals are
    # never flagged (rising numbers at a peak are the reason for a low multiple).
    dislocation_ebit_change_min: float = -10.0
    dislocation_revenue_change_min: float = -5.0
    estimate_cut_pct: float = -5.0
    dislocation_min_quality: float = 50.0
    dislocation_min_value: float = 80.0
    # Compounder flag (`ranking.compounder`, the "Mastercard-like" list; user's choice, Oct 21, 2026), not ranked: every
    # trait at once, which Quality's average cannot require. High returns (ROIC / ROE percentile among peers) that are
    # either steady (`ROIC_Variability` percentile >= `compounder_min_steady_pct` with `ROIC_Trend` >=
    # `compounder_min_roic_trend`) or rising (`ROIC_Trend` >= `compounder_rising_roic_trend` with `ROIC_Variability`
    # percentile >= `compounder_rising_min_steady_pct`); revenue per share compounding smoothly
    # (`Per_Share_Compounding` percentile) and owner earnings per share too (`Owner_Earnings_Compounding`, % a year);
    # shares not meaningfully rising (`Share_Reduction_Rate` >= `compounder_min_share_reduction`); recent revenue, EBIT
    # and analyst EPS estimates held. No Value floor: listed cheapest first.
    compounder_min_roic_pct: float = 75.0
    compounder_min_steady_pct: float = 60.0
    compounder_min_roic_trend: float = -5.0
    compounder_rising_roic_trend: float = 3.0
    compounder_rising_min_steady_pct: float = 20.0
    compounder_min_compounding_pct: float = 75.0
    compounder_min_oe_growth: float = 5.0
    compounder_min_share_reduction: float = -0.5
    compounder_min_quality: float = 70.0
    compounder_min_fcf_pos_share: float = 80.0
    compounder_min_fcf_conversion: float = 0.7
    compounder_max_netdebt_ebit: float = 3.0
    # The fewest annual reports, consecutive from Y0, over which the "Operating companies" and "Fast growers" lists
    # (`screens.operating_list`, `fast_grower_list`, Nov 4, 2026) test growth year by year; display only. Named after
    # "Better every year" (Oct 28, 2026), which they replaced.
    improver_min_years: int = 3
    # The "net cash" pill (`report.pills`; its tab retired Nov 8, 2026, user's choice: Value already counts cash through
    # enterprise value), display only: net cash at least `net_cash_min_pct` (%) of the market cap (`Net_Cash_to_Cap`,
    # shareholder basis; not measured for banks, insurers and holdings). `net_cash_min_share_reduction`: the
    # Shortlist's "no dilution" step.
    net_cash_min_pct: float = 25.0
    net_cash_min_share_reduction: float = -0.5
    # Every screened highlight list keeps only the best `highlight_top_fraction` of the combined ranking
    # (`format.in_best_fraction`; user's choice, Oct 29, 2026: a list is a lens on cheap-and-good companies, not a way
    # around the ranking). "Cheapest" and "Best businesses" are not cut: the top of each score, undiluted.
    highlight_top_fraction: float = 0.2
    # "Below book" list (`ranking.asset_play`, `screens.asset_play_list`): tangible book (or clean book when intangibles
    # are not reported, de-levered when in net debt) >= `asset_play_min_nav_price` (%) of the price now (Nov 7, 2026:
    # the "growing into book" route retired, the list is named for a price below book), the book earning at least
    # `asset_play_min_roic` (normalised ROIC, %; financial review, Nov 7, 2026: book is worth about what it earns, so a
    # 3% earner at 0.7x book is no bargain; was > 0). Shown as "Balance sheet plays" (Nov 8, 2026). Out:
    # `asset_play_excluded_industries` (user's choice, Nov 8, 2026: an asset manager's book is the fund stakes and seed
    # capital it holds, its value the fee stream no book records).
    asset_play_min_nav_price: float = 100.0
    asset_play_excluded_industries: tuple[str, ...] = ("Asset Management",)
    asset_play_min_roic: float = 8.0
    asset_play_min_value: float = 50.0
    asset_play_min_quality: float = 35.0
    asset_play_min_fin_quality: float = 50.0
    asset_play_min_equity_assets: float = 6.0
    asset_play_max_netdebt_ebit: float = 3.0
    asset_play_min_interest_cover: float = 2.5
    asset_play_max_netdebt_nav_share: float = 0.5
    asset_play_min_fcf_pos_share: float = 75.0
    asset_play_min_share_reduction: float = -0.5
    asset_play_min_fin_share_reduction: float = 0.5
    # The "cannibal" pill (`ranking.cannibal`; its tab retired Nov 7, 2026): companies aggressively retiring shares:
    cannibal_min_share_reduction: float = 7.0  # more than 7%/yr net share count reduction (Nov 7, 2026; was 5)
    cannibal_max_share_reduction: float = 50.0  # sanity cap against split / unit shifts
    cannibal_min_value: float = 45.0  # repurchases make sense when shares are cheap or fair
    cannibal_min_roic: float = 0.0  # operating businesses generating positive capital returns
    cannibal_min_quality: float = 50.0  # a buyback in a poor business only shrinks it (Oct 29, 2026)
    # Shares retired must have been paid for: the buybacks of the span at least this share of the retired fraction x
    # today's market cap (tolerant, as shares were bought at older prices); else a consolidation (GGP.AX, Oct 29, 2026).
    cannibal_min_paid_share: float = 0.3
    # Buybacks of the stored years paid from their reported free cash flow plus the spare cash held today. No leverage
    # test: Value (`cannibal_min_value`) prices it (user's choice, Oct 28, 2026).
    cannibal_max_buyback_to_fcf: float = 1.0
    # Display only (`metrics.flows._financing`, `Leverage_Adjustment`): the unlevered return a share must earn; net
    # debt raises its equity's hurdle by (this - the after-tax rate on the debt) x net debt / equity value.
    asset_required_return: float = 9.0
    # ROIC spread (`ROIC_Variability`) is measured against the ROIC level, at least this many points, so a
    # 60% ROIC moving 8 points is as steady as a 15% one moving 2, and values near zero do not explode.
    roic_variability_floor: float = 10.0
    # Mood sentence: an industry note when at least this many reference companies of the industry are cheap.
    theme_min_cheap: int = 3
    # ROIC above this says nothing more about quality; companies with capital <= 0 get it.
    roic_cap: float = 100.0
    # Incremental ROIC is only measured when invested capital grew by at least this share.
    incremental_min_growth: float = 0.05
    # Owner-earnings trend parameters (`Owner_Trend`, shown; it never moves the yield):
    trend_cap_up: float = 15.0  # max credited uptrend, %/yr
    trend_floor_down: float = -10.0  # max penalized downtrend, %/yr
    trend_scale_floor: float = 0.01  # 1% of revenue/share (or book/share)
    # Fewest values a trend is fitted on (Value's owner trend, Quality's ROIC trend): with three, R² is 0.5 on pure
    # noise on average (1 / (n - 1)), so noise earned growth credit (Oct 28, 2026).
    trend_min_points: int = 4
    # Return quality parameters (Quality):
    roic_trend_years: float = 3.0  # 3-year trend adjustment on ROIC
    roic_adj_cap: float = 10.0  # cap on ROIC trend adjustment (+/- 10 pp)
    # ROIC_Variability assumed under 3 years of ROIC: unknown, not perfectly steady (the cache's median, Oct 28, 2026).
    missing_roic_variability: float = 0.10
    overearner_ratio: float = 1.5  # over-earner ratio vs 30th percentile
    # Leverage haircuts on Quality:
    leverage_haircut_operating_moderate: float = 4.0  # NetDebt/EBIT > 4x -> x0.75
    leverage_haircut_operating_severe: float = 6.0  # NetDebt/EBIT > 6x -> x0.50
    leverage_haircut_bank_moderate: float = 4.0  # insurers, holdings: Equity/Assets < 4% -> x0.75
    leverage_haircut_bank_severe: float = 2.5  # insurers, holdings: Equity/Assets < 2.5% -> x0.50
    bank_standard_equity: float = 8.0  # banks: ROE restated to this Equity/Assets (%), `quality._return_quality`
    value_criteria: tuple[Criterion, ...] = VALUE_CRITERIA
    quality_criteria: tuple[Criterion, ...] = QUALITY_CRITERIA
    financial_value_criteria: tuple[Criterion, ...] = FINANCIAL_VALUE_CRITERIA
    financial_quality_criteria: tuple[Criterion, ...] = FINANCIAL_QUALITY_CRITERIA
    quality_candidates: tuple[tuple[str, str], ...] = QUALITY_CANDIDATES
    quality_market_criteria: tuple[str, ...] = QUALITY_MARKET_CRITERIA
    # Quality families compared across one market (Oct 28, 2026): operating and asset-heavy companies are both
    # judged on ROIC, the same question, so a utility's return meets an industrial's; financials stay among their
    # peers (ROE on a levered balance sheet is another question).
    quality_market_families: tuple[str, ...] = ("operating", "asset_heavy")
    shown_percentiles: tuple[tuple[str, str], ...] = SHOWN_PERCENTILES
    shown_market_criteria: tuple[str, ...] = SHOWN_MARKET_CRITERIA

    def criteria(self, dimension: str, family: str) -> tuple[Criterion, ...]:
        """The Value or Quality criteria of a model family (`BUSINESS_MODELS`)."""
        financial = family == "financial"
        if dimension == "Value":
            return self.financial_value_criteria if financial else self.value_criteria
        return self.financial_quality_criteria if financial else self.quality_criteria

    @property
    def annual_years(self) -> list[str]:
        """Annual slots, newest first: Y0, Y1, ..."""
        return [f"Y{i}" for i in range(self.n_years)]

    @property
    def fiscal_years(self) -> list[str]:
        """All periods, TTM first then newest to oldest year."""
        return ["TTM", *self.annual_years]

    @property
    def avg_years(self) -> list[str]:
        return ["TTM", *self.annual_years[:3]]

    @property
    def latest(self) -> str:
        return self.annual_years[0]

    @property
    def prior(self) -> str:
        return self.annual_years[1]

    @property
    def oldest(self) -> str:
        return self.annual_years[-1]

    @property
    def chronological(self) -> list[str]:
        """Periods oldest to newest, TTM last (the order charts and trend maths use)."""
        return [*reversed(self.annual_years), "TTM"]

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "raw.parquet"

    @property
    def ranked_path(self) -> Path:
        return self.data_dir / f"ranked_{self.label}.parquet"

    @property
    def bench_path(self) -> Path:
        """The ranked benchmark of this run (for the charts' background cloud)."""
        return self.data_dir / f"bench_{self.label}.parquet"

    @property
    def snapshot_dir(self) -> Path:
        """Dated score snapshots that `fq evaluate` follows forward."""
        return self.data_dir / "snapshots"

    @property
    def evaluation_paths(self) -> tuple[Path, Path]:
        """(forward returns of the lists, information coefficients) from the last `fq evaluate`."""
        return (
            self.data_dir / f"evaluation_{self.label}_lists.parquet",
            self.data_dir / f"evaluation_{self.label}_ic.parquet",
        )

    @property
    def history_path(self) -> Path:
        """Whether Quality foresaw the fundamentals on the benchmark's own history (`history.quality_ic`), from the
        last `fq evaluate`; one file for every run label (it is a property of the method)."""
        return self.data_dir / "evaluation_quality_history.parquet"

    @property
    def site_dir(self) -> Path:
        """Where this run's HTML and CSV are written."""
        return self.output_dir / self.label
