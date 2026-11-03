"""Automatic archetype detection with provenance tracking."""

import pandas as pd

from fq.metrics.common import _col, _latest, _liquidity, _text
from fq.settings import Settings
from fq.universe import industry_key, industry_models, model_overrides

FEE_NETWORK_MAX_LEVERAGE = 4.0  # assets / equity a fee business with revenue < 10% of assets may carry


def detect_archetypes(df: pd.DataFrame, cfg: Settings) -> tuple[pd.Series, pd.Series]:
    """Detect company archetype and record the deciding rule.

    Precedence:
    1. Ticker in model_overrides.txt -> override
    2. Policyholder claims -> insurer (signature:claims)
    3. Bank / lender balance sheet + NII -> bank (signature:lender)
    4. Fee network (Credit Services/Capital Markets with low/no NII and operating balance sheet) -> operating (signature:fee_network).
       A balance sheet over `FEE_NETWORK_MAX_LEVERAGE` x equity with revenue under 10% of assets is no fee
       network: a securities broker's client funds, margin loans and trading book (CITIC, Guotai Haitong at
       6.5x) are the business, not spare cash. Such rows fall to the industry model (bank), as Nomura and GS do.
    5. Client cash -> client_cash (signature:client_cash)
    6. Asset-heavy plant -> asset_heavy (signature:plant)
    7. Yahoo industry in business_models.txt -> industry model (industry)
    8. Otherwise -> operating (default)
    """
    overrides = model_overrides()
    ind_models = industry_models()

    n = len(df)
    models: list[str] = ["operating"] * n
    sources: list[str] = ["default"] * n

    # Balance sheet & income items across newest periods
    assets = _latest(df, "Total Assets", cfg).where(lambda s: s > 0)
    equity = _latest(df, "Stockholders Equity", cfg)
    debt = _latest(df, "Total Debt", cfg).fillna(0)
    ppe = _latest(df, "Net PPE", cfg).fillna(0)

    rev_ttm = _col(df, "Total Revenue_TTM")
    rev_y0 = _col(df, f"Total Revenue_{cfg.latest}")
    revenue = rev_ttm.fillna(rev_y0).where(lambda s: s > 0)

    # Liquidity in newest period (TTM or Y0)
    liq_ttm = _liquidity(df, "TTM")
    liq_y0 = _liquidity(df, cfg.latest)
    liquidity = liq_ttm.where(liq_ttm > 0, liq_y0)

    # Claims / Revenue
    claims = _latest(df, "Net Policyholder Benefits And Claims", cfg).abs().fillna(0)
    claims_ratio = claims / revenue

    # NII / Revenue
    nii = _latest(df, "Net Interest Income", cfg).abs()
    nii_ratio = nii / revenue

    # Turnovers & ratios
    turnover = revenue / assets
    asset_leverage = assets / equity.where(equity > 0)
    ppe_intensity = ppe / assets

    tickers = [str(t).upper() for t in df["Ticker"]] if "Ticker" in df.columns else ["" for _ in range(n)]
    industries = [_text(df, "industry").iloc[i] if "industry" in df.columns else "" for i in range(n)]

    for i in range(n):
        ticker = tickers[i]
        industry = industries[i]
        ind_key = industry_key(industry) if isinstance(industry, str) else ""
        base_ind_model = ind_models.get(ind_key, "operating")

        # Rule 1: Ticker Override
        if ticker in overrides:
            models[i] = overrides[ticker]
            sources[i] = "override"
            continue

        # Rule 2: Claims (Insurer)
        if claims_ratio.iloc[i] >= 0.20 and base_ind_model in ("operating", "client_cash"):
            models[i] = "insurer"
            sources[i] = "signature:claims"
            continue

        # Rule 3: Lender (Bank)
        t_over = turnover.iloc[i]
        a_lev = asset_leverage.iloc[i]
        n_rat = nii_ratio.iloc[i]
        if pd.notna(t_over) and t_over < 0.10 and pd.notna(a_lev) and a_lev > 8.0:
            if pd.notna(n_rat) and n_rat >= 0.25:
                models[i] = "bank"
                sources[i] = "signature:lender"
                continue

        # Rule 4: Fee Network (Operating)
        if ind_key in ("creditservices", "capitalmarkets", "financialdatastockexchanges"):
            is_lender_bs = pd.notna(t_over) and t_over < 0.10 and pd.notna(a_lev) and a_lev > FEE_NETWORK_MAX_LEVERAGE
            low_nii = pd.isna(n_rat) or n_rat < 0.15
            if low_nii and not is_lender_bs:
                models[i] = "operating"
                sources[i] = "signature:fee_network"
                continue

        # Rule 5: Client Cash
        liq = liquidity.iloc[i]
        ass = assets.iloc[i]
        eq = equity.iloc[i]
        d = debt.iloc[i]
        if base_ind_model == "operating" and pd.notna(liq) and pd.notna(ass) and ass > 0:
            claims_holder = max(0.0, (eq if pd.notna(eq) else 0.0)) + d
            if liq > 0.45 * ass and liq > claims_holder:
                models[i] = "client_cash"
                sources[i] = "signature:client_cash"
                continue

        # Rule 6: Asset-Heavy Plant
        ppe_int = ppe_intensity.iloc[i]
        if base_ind_model == "operating" and pd.notna(ppe_int) and ppe_int > 0.60:
            if pd.notna(t_over) and t_over < 0.25:
                models[i] = "asset_heavy"
                sources[i] = "signature:plant"
                continue

        # Rule 7: Yahoo Industry
        if base_ind_model != "operating":
            models[i] = base_ind_model
            sources[i] = "industry"
            continue

        # Rule 8: Default
        models[i] = "operating"
        sources[i] = "default" if ind_key == "" else "industry"

    return (
        pd.Series(models, index=df.index, name="Business_Model"),
        pd.Series(sources, index=df.index, name="Archetype_Source"),
    )
