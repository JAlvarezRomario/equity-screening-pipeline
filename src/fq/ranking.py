"""Dual Value / Quality ranking. Pure functions: DataFrames in, DataFrames out."""

import logging
from collections.abc import Callable, Collection, Sequence

import numpy as np
import pandas as pd

from fq.settings import BUSINESS_MODELS, Criterion, Settings, display_name
from fq.universe import is_real_estate_owner

log = logging.getLogger(__name__)

WINSOR_MIN_VALUES = 20  # below this, quantiles are too noisy to clip on
MISSING_PERCENTILE = 0.5  # a missing criterion is neither worst nor best (`weighted_percentile_score`)


def _flag(df: pd.DataFrame, column: str) -> pd.Series:
    """A boolean flag column, False where missing (handles NaN / numpy bool)."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column] == True  # noqa: E712


# Flags that keep a company out of every highlight list (still scored and in the full ranking).
UNTRUSTED_FLAGS = ("Data_Suspect", "Insufficient_Data", "Industry_Excluded_Kept", "Peak_Margin", "Distress",
                   "Earnings_Collapse", "Revenue_Decline", "Accounting_Flag")  # fmt: skip


def trusted(df: pd.DataFrame) -> pd.Series:
    """Fit for a highlight list: none of `UNTRUSTED_FLAGS`, and no open `Model_Review` (a business model that the
    company's own statements question is not confirmed until `model_overrides.txt` says so: until then the
    company may be measured with the wrong yardstick)."""
    ok = pd.Series(True, index=df.index)
    for flag in UNTRUSTED_FLAGS:
        ok &= ~_flag(df, flag)
    if "Model_Review" in df.columns:
        ok &= df["Model_Review"].isna()
    return ok


def comparable(df: pd.DataFrame) -> pd.Series:
    """Rows of a ranked business model (`Settings.ranked_models`): they take rank positions, top-list
    places and the Dislocation flag, and form the cross-model (market) distribution of Value.

    Every model since Oct 2026 (user's choice); a model taken out of the setting is scored against its own peer
    group and shown, but cannot crowd the top of a ranking. `apply_dual_ranking`
    stores the answer as `Ranked_Model`. With no ranked row at all (a run of only banks), every row counts.
    """
    if "Ranked_Model" in df.columns:
        ok = _flag(df, "Ranked_Model")
    elif "Business_Model" in df.columns:
        ok = df["Business_Model"].isin(Settings().ranked_models)
    else:
        ok = ~_flag(df, "Financial_Mode")
    return ok if ok.any() else pd.Series(True, index=df.index)


# `Rank_Group` values (`rank_group`): only "ranked" takes a position; the others are scored and listed apart.
RANKED = "ranked"
RANK_GROUP_LABELS = {  # why a company has no rank, as the full table's pill and the company page's note say it
    "financial": "bank or insurer",
    "holding": "holding",
    "real_estate": "REIT",
    "apart": "model not ranked",
    "insufficient": "too little data",
    "unprofitable": "loss in a normal year",
}


def rank_group(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Which companies take a position in the rankings (user's choice, Nov 5, 2026: profitable operating companies
    only). "financial" (bank, insurer) and "holding" (`Settings.rank_apart_models`), "real_estate" (an equity REIT,
    `lists/real_estate_owners.txt`), "apart" (a model outside `Settings.ranked_models`), "insufficient" (no Value or
    no Quality score), "unprofitable" (normal-year owner earnings or normalised ROIC at or below zero: `Owner_Yield`,
    `ROIC_Norm`, both normalised, so one bad year or a cyclical trough keeps the rank; a blank is not a loss), else
    `RANKED`. Scores are not
    touched: the percentiles keep `comparable`'s market."""
    models = df["Business_Model"] if "Business_Model" in df.columns else pd.Series(None, index=df.index, dtype=object)
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    bank_or_insurer = models.isin({"bank", "insurer"}) & models.isin(cfg.rank_apart_models)
    insufficient = _num(df, "Value_Score").isna() | _num(df, "Quality_Score").isna()
    # A figure that is missing is not a loss (missing is not bad; without earnings power there is no Value score).
    profitable = ~(_num(df, "Owner_Yield") <= 0) & ~(_num(df, "ROIC_Norm") <= 0)
    return pd.Series(
        np.select(
            [bank_or_insurer, models.isin(cfg.rank_apart_models), industry.map(is_real_estate_owner).astype(bool),
             ~comparable(df), insufficient, ~profitable],
            ["financial", "holding", "real_estate", "apart", "insufficient", "unprofitable"],
            RANKED,
        ),
        index=df.index,
    )  # fmt: skip


def rankable(df: pd.DataFrame) -> pd.Series:
    """Rows that take a position (`Rank_Group` == "ranked"). A frame scored before the column existed (a benchmark
    file ranked before Nov 5, 2026) gets its group computed (`rank_group`), so a run never takes old positions from
    it; an old snapshot (scores without their inputs) keeps its own `Rank`, as it was ranked then; a frame without
    either falls back to `comparable`."""
    if "Rank_Group" in df.columns:
        return df["Rank_Group"] == RANKED
    if {"Value_Score", "Quality_Score", "Owner_Yield"} <= set(df.columns):
        return rank_group(df, Settings()) == RANKED
    if "Rank" in df.columns:
        return pd.to_numeric(df["Rank"], errors="coerce").notna()
    return comparable(df)


def combined_order(value_pos: pd.Series, quality_pos: pd.Series, penalty: float) -> pd.Series:
    """The combined ranking's sort key: the worse of the Value and Quality positions (user's choice, Oct 28, 2026:
    cheap AND good, so Value 99 / Quality 1 no longer ties 50 / 50), ties to the lower sum. A missing position counts
    as `penalty` (larger than any position)."""
    value_pos, quality_pos = value_pos.fillna(penalty), quality_pos.fillna(penalty)
    return np.fmax(value_pos, quality_pos) + (value_pos + quality_pos) / (2 * penalty + 1)


def model_groups(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(peer group, family) per row (`settings.BUSINESS_MODELS`); without `Business_Model` (frames built
    by hand), financial-mode rows form the financial family and everyone else is operating."""
    if "Business_Model" in df.columns:
        models = df["Business_Model"].fillna("operating")
        return models.map(lambda m: BUSINESS_MODELS[m][0]), models.map(lambda m: BUSINESS_MODELS[m][1])
    family = pd.Series(np.where(_flag(df, "Financial_Mode"), "financial", "operating"), index=df.index)
    return family, family


def _clean(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _winsor_bounds(v: pd.Series, winsor_pct: float) -> tuple[float, float] | None:
    """Inner-side quantiles: in samples under 1/winsor_pct the lone extreme still ties with its neighbour."""
    if winsor_pct <= 0 or v.notna().sum() < WINSOR_MIN_VALUES:
        return None
    return v.quantile(winsor_pct, interpolation="higher"), v.quantile(1 - winsor_pct, interpolation="lower")


def criterion_percentile(
    values: pd.Series, direction: str, winsor_pct: float, reference: pd.Series | None = None
) -> pd.Series:
    """Percentile (0-1, 1 = best) among the companies that have a value; NaN where missing.

    Tails are winsorized first, so one implausible value ties with the top group instead of
    standing alone. With `reference` (a benchmark), each value is placed in the reference
    distribution instead (mid-rank, winsor bounds from the reference): a reference member gets
    the same percentile as when the reference is ranked on its own.
    """
    v = _clean(values)
    if reference is None:
        if bounds := _winsor_bounds(v, winsor_pct):
            v = v.clip(*bounds)
        return v.rank(method="average", ascending=(direction == "top"), pct=True)
    ref = _clean(reference).dropna()
    if ref.empty:
        return pd.Series(np.nan, index=v.index)
    if bounds := _winsor_bounds(ref, winsor_pct):
        v, ref = v.clip(*bounds), ref.clip(*bounds)
    sign = 1 if direction == "top" else -1
    ordered = np.sort(ref.to_numpy() * sign)
    x = v.to_numpy() * sign
    below = np.searchsorted(ordered, x, side="left")
    at_or_below = np.searchsorted(ordered, x, side="right")
    pct = (below + at_or_below + 1) / (2 * len(ordered))
    return pd.Series(np.clip(pct, None, 1.0), index=v.index).where(v.notna())


def sector_percentile(
    values: pd.Series,
    direction: str,
    winsor_pct: float,
    sectors: pd.Series,
    ref_values: pd.Series,
    ref_sectors: pd.Series,
    min_members: int,
) -> pd.Series:
    """Percentile within the company's own sector of the reference; NaN where the sector has fewer
    than `min_members` reference companies with a value (too few to say what is cheap there)."""
    out = pd.Series(np.nan, index=values.index)
    counts = ref_sectors[_clean(ref_values).notna()].value_counts()
    for sector in counts[counts >= min_members].index:
        mask = sectors == sector
        if mask.any():
            ref = ref_values[ref_sectors == sector]
            out[mask] = criterion_percentile(values[mask], direction, winsor_pct, reference=ref)
    return out


def _peer_percentile(
    df: pd.DataFrame,
    column: str,
    direction: str,
    winsor_pct: float,
    rows: pd.Series,
    base: pd.DataFrame,
    min_members: int,
) -> pd.Series:
    """Percentile of `rows` within their own peer group of `base`, or their family where the peer
    group has fewer than `min_members` values (a handful of reinsurers say little)."""
    out = pd.Series(np.nan, index=df.index)
    peer, family = model_groups(df)
    base_peer, base_family = model_groups(base)
    values = base.get(column, pd.Series(np.nan, index=base.index))
    for (p, f), idx in df[rows].groupby([peer[rows], family[rows]]).groups.items():
        ref = values[base_peer == p]
        if _clean(ref).notna().sum() < min_members and p != f:
            ref = values[base_family == f]
        out[idx] = criterion_percentile(df.loc[idx, column], direction, winsor_pct, reference=ref)
    return out


def weighted_percentile_score(
    df: pd.DataFrame,
    criteria: Sequence[Criterion] | Callable[[str], Sequence[Criterion]],
    min_coverage: float,
    winsor_pct: float,
    reference: pd.DataFrame | None = None,
    sector_blend: float = 0.0,
    sector_min_members: int = 15,
    market_families: Collection[str] = (),
    market_criteria: Collection[str] = (),
) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]] | None:
    """Weighted mean percentile (0-100, higher is better) over the criteria each company has.

    `criteria` is one list, or a function family -> list (`Settings.criteria`): banks and insurers are
    scored on their own criteria. A missing value is neither the worst nor the best: it counts at the middle
    percentile (50), so a company known only by its strongest criterion does not keep that percentile undiluted
    (Oct 28, 2026: PURR, PXED and GGP.AX topped Quality on ROIC alone). A criterion no company has (no column) is
    left out. Below `min_coverage` (share of the family's weight with data) the score is NaN.

    Each percentile is taken within the company's peer group of `reference` (default `df`), so ROE is
    compared with ROE. Families in `market_families` (Value) instead start from the market: the ranked
    models of the reference (`comparable`), where a criterion exists there, so an implied return of 6%
    ranks the same for a utility and an industrial. With `sector_blend` > 0 that share comes from within
    the company's Yahoo sector among its peers (an operating company in a sector too small uses the
    market only; the others fall back to their peer group). Columns in `market_criteria` (Quality criteria that mean
    the same for every model, `Settings.quality_market_criteria`) take the market percentile for every family.
    Returns (score, coverage, {criterion: percentile 0-1}), or None if no criterion column exists.
    """
    of = criteria if callable(criteria) else (lambda _family: criteria)
    base = df if reference is None else reference
    market = base[comparable(base)]
    market_wide = market[model_groups(market)[1].isin(market_families)] if market_families else market
    if market_wide.empty:
        market_wide = market
    _, family = model_groups(df)
    total = pd.Series(0.0, index=df.index)
    weight = pd.Series(0.0, index=df.index)
    weight_all = pd.Series(0.0, index=df.index)
    weight_known = pd.Series(0.0, index=df.index)  # criteria with a column: missing values there count at 50
    pcts: dict[str, pd.Series] = {}
    by_sector = sector_blend > 0 and "sector" in df.columns and "sector" in base.columns
    found = False
    for fam in family.unique():
        rows = family == fam
        operating = fam == "operating"
        for column, direction, w in of(fam):
            weight_all += rows * w
            if column not in df.columns:
                log.warning("Ranking column '%s' not found; skipping", column)
                continue
            found = True
            weight_known += rows * w
            if column in market_criteria:
                ref = market[column] if column in market.columns else pd.Series(np.nan, index=market.index)
                pct = criterion_percentile(df[column], direction, winsor_pct, ref)
            else:
                pct = _peer_percentile(df, column, direction, winsor_pct, rows, base, sector_min_members)
            if fam in market_families and column not in market_criteria:
                peer = pct
                wide = criterion_percentile(df[column], direction, winsor_pct, market_wide.get(column)).where(rows)
                within = peer
                if by_sector:
                    peers = base[model_groups(base)[0].isin(model_groups(df)[0][rows].unique())]
                    ref_values = peers.get(column, pd.Series(np.nan, index=peers.index))
                    within = sector_percentile(df[column], direction, winsor_pct, df["sector"], ref_values,
                                               peers["sector"], sector_min_members).where(rows)  # fmt: skip
                    within = within if operating else within.fillna(peer)
                elif operating:
                    within = pd.Series(np.nan, index=df.index)
                blended = sector_blend * within + (1 - sector_blend) * wide
                pct = blended.fillna(wide if operating else wide.fillna(within))
            pct = pct.where(rows)
            pcts[column] = pct if column not in pcts else pcts[column].fillna(pct)
            total += pct.fillna(0) * w
            weight += pct.notna() * w
    if not found:
        return None
    coverage = (weight / weight_all.replace(0, np.nan)).clip(upper=1)
    neutral = (weight_known - weight).clip(lower=0) * MISSING_PERCENTILE
    score = ((total + neutral) / weight_known.replace(0, np.nan) * 100).where(coverage >= min_coverage)
    return score, coverage, pcts


def apply_dual_ranking(
    df: pd.DataFrame, cfg: Settings, reference: pd.DataFrame | None = None, benchmark: str | None = None
) -> pd.DataFrame:
    """Add Value/Quality scores, coverage and ranks, Combined_Rank_Score and Rank; sorted best first.

    Scores are percentiles within `df`, or within `reference` (a ranked benchmark named `benchmark`)
    when given, so a company scores the same in any group. Ranks are always within `df`; with a
    reference, `<Dim>_Rank_Bench` is the position among the reference companies. `Pct_<criterion>`
    (0-100) keeps each criterion's percentile for the score breakdown.
    The combined order (`Combined_Rank_Score`, `Rank`) is the worse of the Value and Quality positions, ties by
    their sum, among the reference companies when benchmarked; rows that are not `trusted` follow every trusted one.
    Companies without enough data in either dimension get `Insufficient_Data`, take no `Rank` and
    sort after every scored company. Only `rank_group` "ranked" rows (profitable operating companies, Nov 5, 2026)
    take `Rank`, `Value_Rank` and `Quality_Rank`; the others (`Rank_Group` says why) are scored as before, against
    `comparable`'s market, and sort after everyone in their own combined order.
    """
    out = df.copy()
    benchmarked = reference is not None
    if "Business_Model" in out.columns:
        out["Ranked_Model"] = out["Business_Model"].isin(cfg.ranked_models)
    penalty = len(out) + 1

    for name, blend in (
        ("Value", cfg.sector_blend),  # cheapness: against the market and the sector, or the peer group
        ("Quality", 0.0),  # a better business is better in any sector: peer group, or market-wide where comparable
    ):
        result = weighted_percentile_score(
            out, lambda family, n=name: cfg.criteria(n, family), cfg.min_coverage, cfg.winsor_pct, reference,
            blend, cfg.sector_min_members,
            cfg.value_market_families if name == "Value" else cfg.quality_market_families,
            cfg.quality_market_criteria if name == "Quality" else (),
        )  # fmt: skip
        score, coverage, pcts = result if result is not None else (np.nan, np.nan, {})
        for column, pct in pcts.items():
            out[f"Pct_{column}"] = pct * 100
        out[f"{name}_Score"] = score
        out[f"{name}_Coverage"] = coverage
    out["Rank_Group"] = rank_group(out, cfg)
    apart = ~rankable(out)
    for name in ("Value", "Quality"):
        out[f"{name}_Rank"] = out[f"{name}_Score"].where(~apart).rank(method="min", ascending=False)
        if benchmarked and f"{name}_Score" in reference.columns:
            ref_scores = reference.loc[rankable(reference), f"{name}_Score"].dropna().to_numpy()
            out[f"{name}_Rank_Bench"] = out[f"{name}_Score"].map(
                lambda s, r=ref_scores: np.nan if pd.isna(s) else 1 + int((r > s + 1e-9).sum())
            )
    # Candidates under observation: same percentiles as a Quality criterion, but no score uses them.
    candidates = [(c, d, 1.0) for c, d in cfg.quality_candidates if c in out.columns]
    result = weighted_percentile_score(out, candidates, 0.0, cfg.winsor_pct, reference) if candidates else None
    for column, pct in (result[2] if result is not None else {}).items():
        out[f"Cand_Pct_{column}"] = pct * 100
    # Shown percentiles (for highlight lists like Compounders, Asset Plays, and company display):
    shown = [(c, d, 1.0) for c, d in getattr(cfg, "shown_percentiles", ()) if c in out.columns]
    result_shown = (
        weighted_percentile_score(
            out,
            shown,
            0.0,
            cfg.winsor_pct,
            reference,
            0.0,
            cfg.sector_min_members,
            (),
            getattr(cfg, "shown_market_criteria", ()),
        )
        if shown
        else None
    )
    for column, pct in (result_shown[2] if result_shown is not None else {}).items():
        if f"Pct_{column}" not in out.columns:
            out[f"Pct_{column}"] = pct * 100
    if benchmarked:
        out["Benchmark"] = benchmark or "reference"
        ranked_ref = reference[rankable(reference)]
        out["Benchmark_N"] = (
            int(ranked_ref["Value_Score"].notna().sum()) if "Value_Score" in ranked_ref else len(ranked_ref)
        )
    out["Insufficient_Data"] = out["Value_Score"].isna() | out["Quality_Score"].isna()
    if out["Insufficient_Data"].any():
        log.info("%d companies lack enough data to rank", int(out["Insufficient_Data"].sum()))
    if apart.any():
        log.info("Scored but ranked apart: %s", out.loc[apart, "Rank_Group"].value_counts().to_dict())

    log.info("Value-Quality correlation (Spearman): %.2f", value_quality_correlation(out))
    # With a benchmark the positions among its companies decide, so a group is ordered as it would be
    # in the benchmark's own report. The worse of the two positions orders (`combined_order`); ties go to the lower
    # sum, then the higher score sum.
    dim = "_Rank_Bench" if benchmarked and "Value_Rank_Bench" in out else "_Rank"
    if dim == "_Rank_Bench":
        penalty = max(penalty, int(out["Benchmark_N"].max()) + 1)
    combined = combined_order(out[f"Value{dim}"], out[f"Quality{dim}"], penalty)
    # Untrusted rows (`trusted`: suspect data, distress, peak margins...) after every trusted scored row (Oct 30, 2026:
    # ranking on measured numbers puts the measurement errors at the top; 31 of the top 100 were untrusted).
    combined = combined.where(trusted(out) | out["Insufficient_Data"], penalty + combined)
    combined = combined.where(~out["Insufficient_Data"], 2 * penalty + combined)  # after every scored row
    # Rows ranked apart after everyone, in the order of their own scores (they have no positions).
    own = combined_order(out["Value_Score"].rank(ascending=False), out["Quality_Score"].rank(ascending=False), penalty)
    out["Combined_Rank_Score"] = combined.where(~apart, 4 * penalty + own)
    out["_tie"] = -(out["Value_Score"].fillna(0) + out["Quality_Score"].fillna(0))
    out = out.sort_values(["Combined_Rank_Score", "_tie"], kind="stable").drop(columns="_tie").reset_index(drop=True)
    ranked = rankable(out) & ~out["Insufficient_Data"]
    out["Rank"] = pd.Series(np.arange(1, len(out) + 1), index=out.index).where(ranked)
    themes = reference[rankable(reference)] if benchmarked else out[rankable(out)]  # industry-wide discounts
    scored = reference[comparable(reference)] if benchmarked else out[comparable(out)]  # the asset plays' yardstick
    out["Dislocation"], out["Mood_Check"] = dislocation(out, cfg, themes, benchmark if benchmarked else None)
    if out["Dislocation"].any():
        log.info("Possible mood mispricings: %s", ", ".join(out.loc[out["Dislocation"], "Ticker"]))
    out["Compounder"] = compounder(out, cfg)
    if out["Compounder"].any():
        log.info("Compounders: %s", ", ".join(out.loc[out["Compounder"], "Ticker"]))
    out["Asset_Play"], out["Asset_Play_Score"] = asset_play(out, cfg, scored)
    if out["Asset_Play"].any():
        log.info("Asset plays: %s", ", ".join(out.loc[out["Asset_Play"], "Ticker"]))
    out["Cannibal"] = cannibal(out, cfg)
    if out["Cannibal"].any():
        log.info("Cannibals: %s", ", ".join(out.loc[out["Cannibal"], "Ticker"]))
    return out


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    return _clean(df[column]) if column in df.columns else pd.Series(np.nan, index=df.index)


def dislocation(
    df: pd.DataFrame, cfg: Settings, reference: pd.DataFrame | None = None, benchmark: str | None = None
) -> tuple[pd.Series, pd.Series]:
    """(`Dislocation` flag, `Mood_Check` sentence) - do the numbers explain why a company is cheap?

    Only for cheap companies (Value at least `dislocation_min_value`). The share price path plays no
    part: a company can be undervalued whether its price fell or rose. Flagged when revenue and EBIT
    held recently, analysts did not cut next year's EPS materially, Quality is at least
    `dislocation_min_quality` and the data is trusted: nothing in the numbers explains the low price,
    which is what Mr Market's mood looks like. Otherwise the sentence names the reason. Absolute
    thresholds, so the flag does not depend on the other companies. Not part of any score: `fq evaluate`
    follows it as its own list, next to the cheap companies it leaves out.

    Cyclicals are never flagged: at the top of a cycle rising numbers are the reason for a low multiple.
    When at least `theme_min_cheap` companies of the same industry in `reference` (the benchmark, else the run)
    are cheap, the sentence says the discount looks industry-wide: a theme the trailing numbers cannot show yet.
    """
    revenue, ebit = _num(df, "Revenue_Change_Recent"), _num(df, "EBIT_Change_Recent")
    estimates, quality, value = _num(df, "EPS_Rev_90d"), _num(df, "Quality_Score"), _num(df, "Value_Score")
    cheap = value >= cfg.dislocation_min_value
    held = (revenue >= cfg.dislocation_revenue_change_min) & (ebit >= cfg.dislocation_ebit_change_min)
    cut = estimates < cfg.estimate_cut_pct
    cyclical = _flag(df, "Cyclical")
    ranked = rankable(df)
    trusted = ranked & ~_flag(df, "Data_Suspect") & ~_flag(df, "Insufficient_Data") & ~_flag(df, "Distress")
    flag = cheap & held & ~cut & (quality >= cfg.dislocation_min_quality) & trusted & ~cyclical

    ref = df if reference is None else reference
    industry = df["industry"] if "industry" in df.columns else pd.Series(None, index=df.index, dtype=object)
    if "industry" in ref.columns:
        cheap_in = ref.loc[_num(ref, "Value_Score") >= cfg.dislocation_min_value, "industry"].value_counts()
    else:
        cheap_in = pd.Series(dtype=int)
    where = f"the {display_name(benchmark)} benchmark" if benchmark else "this run"

    checks = []
    for i in df.index:
        if not cheap[i]:
            checks.append(None)
            continue
        reasons = [f"revenue {revenue[i]:+.0f}%" if revenue[i] < cfg.dislocation_revenue_change_min else None,
                   f"EBIT {ebit[i]:+.0f}%" if ebit[i] < cfg.dislocation_ebit_change_min else None,
                   f"EPS estimates {estimates[i]:+.0f}% in 90 days" if cut[i] else None]  # fmt: skip
        reasons = [r for r in reasons if r]
        text = f"Cheap (Value {value[i]:.0f}). "
        if reasons:
            text += "The numbers may explain it (recent " + ", ".join(reasons) + ")."
        elif not held[i]:
            text += "Recent revenue / EBIT change unknown, so a reason cannot be ruled out."
        elif cyclical[i]:
            text += "Cyclical: recent revenue / EBIT may be the top of the cycle, so they cannot rule out a reason."
        else:
            est = (
                "no analyst estimates to confirm"
                if pd.isna(estimates[i])
                else f"EPS estimates {estimates[i]:+.0f}% in 90 days"
            )
            text += f"Business held (recent revenue {revenue[i]:+.0f}%, EBIT {ebit[i]:+.0f}%, {est})"
            if flag[i]:
                text += ": nothing in the numbers explains the low price."
            elif not ranked[i]:
                text += ", but it is not in the rankings (judged against its own kind)."
            else:
                text += f", but Quality is below {cfg.dislocation_min_quality:.0f} or the data is not trusted."
        n = int(cheap_in.get(industry[i], 0)) if pd.notna(industry[i]) else 0
        if n >= cfg.theme_min_cheap:
            text += (
                f" {n} {industry[i]} companies in {where} are cheap: an industry-wide discount usually"
                " reflects a theme (check the news) more than one company's mood."
            )
        checks.append(text)
    return flag, pd.Series(checks, index=df.index, dtype=object)


def compounder(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """`Compounder` flag - does the company look like Mastercard, whatever its price?

    Quality averages its criteria, so a steady company that does not grow can score as a compounder does; this
    asks for every trait at once (thresholds in `Settings.compounder_*`): high returns on capital among peers
    that are either steady and not falling or rising, revenue and owner earnings per share compounding smoothly,
    shares not meaningfully rising, and recent revenue, EBIT and analyst EPS estimates held. Ranked, trusted
    (as `top_n`) and not cyclical: a cycle's upswing is not compounding. Not part of any score and no Value
    floor: the report lists the flagged companies in combined rank order, and `fq evaluate` follows them as their own list.
    """
    _, family = model_groups(df)
    fit = rankable(df) & trusted(df) & ~_flag(df, "Cyclical") & ~_flag(df, "Financial_Mode") & (family == "operating")
    steady_or_rising = (
        (_num(df, "Pct_ROIC_Variability") >= cfg.compounder_min_steady_pct)
        & (_num(df, "ROIC_Trend") >= cfg.compounder_min_roic_trend)
    ) | (
        (_num(df, "ROIC_Trend") >= cfg.compounder_rising_roic_trend)
        & (_num(df, "Pct_ROIC_Variability") >= cfg.compounder_rising_min_steady_pct)
    )
    held = (
        ~(_num(df, "Revenue_Change_Recent") < cfg.dislocation_revenue_change_min)
        & ~(_num(df, "EBIT_Change_Recent") < cfg.dislocation_ebit_change_min)
        & ~(_num(df, "EPS_Rev_90d") < cfg.estimate_cut_pct)
    )
    profile = (
        (_num(df, "Quality_Score") >= cfg.compounder_min_quality)
        & (_num(df, "Pct_ROIC_Norm") >= cfg.compounder_min_roic_pct)
        & steady_or_rising
        & (_num(df, "Pct_Per_Share_Compounding") >= cfg.compounder_min_compounding_pct)
        & (_num(df, "Owner_Earnings_Compounding") >= cfg.compounder_min_oe_growth)
        & (_num(df, "Share_Reduction_Rate") >= cfg.compounder_min_share_reduction)
        & ~(_num(df, "FCF_Positive_Share") < cfg.compounder_min_fcf_pos_share)
        & ~(_num(df, "FCF_Conversion") < cfg.compounder_min_fcf_conversion)
        & ~(_num(df, "NetDebt_to_EBIT") > cfg.compounder_max_netdebt_ebit)
        & held
    )
    return (fit & profile).astype(bool)


def cannibal(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """`Cannibal` flag - companies aggressively buying back and retiring their own shares (>7%/yr) out of their own
    money, not cyclical (an upswing pays for buybacks at the top), Quality >= `cannibal_min_quality`, the retired shares
    actually bought (`_buybacks_paid`, not a consolidation), and the buybacks of the stored years at most `cannibal_max_buyback_to_fcf` x their reported free cash flow (all
    capex, interest and leases charged) plus the spare cash they hold (`Excess_Cash`), so a buyback paid with new debt
    does not qualify but one paid from a cash pile does. Leverage is not tested here: Value (`cannibal_min_value`)
    already prices it. Years without a buyback figure leave the funding test open (missing is not bad)."""
    years = cfg.annual_years
    bought = sum(_num(df, f"Repurchase Of Capital Stock_{y}").abs().fillna(0.0) for y in years)
    fcf = sum(_num(df, f"FCF_Reported_{y}").fillna(0.0) for y in years)
    spare = _num(df, "Excess_Cash").clip(lower=0.0).fillna(0.0)
    funded = ~(bought > cfg.cannibal_max_buyback_to_fcf * fcf.clip(lower=0.0) + spare)

    red = _num(df, "Share_Reduction_Rate")
    val = _num(df, "Value_Score")
    roic = _num(df, "ROIC_Norm")

    qualifies = (
        rankable(df)
        & trusted(df)
        & ~_flag(df, "Cyclical")
        & (_num(df, "Quality_Score") >= cfg.cannibal_min_quality)
        & funded
        & _buybacks_paid(df, cfg)
        & (red > cfg.cannibal_min_share_reduction)
        & (red <= cfg.cannibal_max_share_reduction)
        & (val >= cfg.cannibal_min_value)
        & (roic > cfg.cannibal_min_roic)
    )
    return qualifies.fillna(False).astype(bool)


def _buybacks_paid(df: pd.DataFrame, cfg: Settings) -> pd.Series:
    """Shares retired were bought: over the span `Share_Reduction_Rate` measures (oldest annual share count to TTM, else
    Y0), the buybacks of the years in it reach `cannibal_min_paid_share` x the retired fraction x today's market cap.
    A count that fell with no buyback paid is a share consolidation Yahoo's split history missed (GGP.AX). No
    repurchase figure in the span leaves the test open (missing is not bad)."""
    years = cfg.annual_years
    end = _num(df, "Ordinary Shares Number_TTM").fillna(_num(df, f"Ordinary Shares Number_{cfg.latest}"))
    shares = np.column_stack([_num(df, f"Ordinary Shares Number_{y}").to_numpy(dtype=float) for y in years])
    present = shares > 0
    oldest = np.where(present.any(axis=1), len(years) - 1 - np.argmax(present[:, ::-1], axis=1), 0)
    start = shares[np.arange(len(df)), oldest]
    # The buybacks of year i moved the count from year i + 1's to year i's: the years newer than the oldest count.
    rep = np.column_stack([_num(df, f"Repurchase Of Capital Stock_{y}").to_numpy(dtype=float) for y in years])
    in_span = np.arange(len(years))[None, :] < oldest[:, None]
    bought = pd.Series(np.where(in_span, np.abs(np.nan_to_num(rep)), 0.0).sum(axis=1), index=df.index)
    reported = pd.Series((in_span & ~np.isnan(rep)).any(axis=1), index=df.index)
    retired = (1 - end / pd.Series(start, index=df.index)).clip(lower=0.0)
    cap = _num(df, "Market Cap") * 1e9
    return ~reported | ~(bought < cfg.cannibal_min_paid_share * retired * cap)


def asset_play(df: pd.DataFrame, cfg: Settings, reference: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """`Asset_Play` flag and `Asset_Play_Score` (0-100) for tangible-NAV bargains with balance-sheet and capital
    discipline.

    Qualifies trusted companies ranked or ranked apart as holdings (Nov 5, 2026: banks and insurers have lists of their
    own, and an equity REIT's book is depreciated cost under US GAAP but fair value under IFRS, so it is never asked)
    where enterprise-de-levered tangible book per share (`Effective_NAV_to_Price`, minority-adjusted tangible book
    `NAV_to_Price` with fallback to common book only when intangibles are unreported; a holding's book with its
    goodwill, `BV_to_Price`: stripping it values the subsidiaries it bought at zero)
    either already exceeds the share price (`>= asset_play_min_nav_price` with non-negative `Book_Per_Share_Compounding`)
    or is at least `asset_play_min_nav_compounding` and compounding smoothly enough (`>= asset_play_min_book_growth`,
    capped at `asset_play_max_book_growth_cap`) to reach the share price within `asset_play_horizon_years`. Requires
    positive normalised ROIC and owner earnings, `Value_Score >= asset_play_min_value`, no meaningful dilution, bounded
    debt and recurring positive FCF for non-financials, and capital adequacy plus active buybacks for banks/insurers.
    """
    fin = _flag(df, "Financial_Mode")
    group = df["Rank_Group"] if "Rank_Group" in df.columns else rank_group(df, cfg)
    holding = group == "holding"
    # Its own earnings tests below decide profitability; the kinds judged elsewhere stay out.
    fit = comparable(df) & trusted(df) & ~group.isin({"financial", "real_estate", "apart"})

    raw_nav = _num(df, "NAV_to_Price").fillna(_num(df, "BV_to_Price")).where(~holding, _num(df, "BV_to_Price"))
    net_cash = _num(df, "Net_Cash_to_Cap")
    net_debt_pct = (-net_cash).clip(lower=0.0).fillna(0.0)
    delevered = (raw_nav + net_debt_pct) / (100.0 + net_debt_pct) * 100.0
    eff_nav = _num(df, "Effective_NAV_to_Price").fillna(np.minimum(raw_nav, delevered).where(net_debt_pct > 0, raw_nav))
    eff_nav = eff_nav.where(~holding, raw_nav)
    bv_comp = _num(df, "Book_Per_Share_Compounding").fillna(_num(df, "Book_Compounding"))
    share_red = _num(df, "Share_Reduction_Rate")

    raw_bv = _num(df, "BV_to_Price")
    exceeds_now = (eff_nav >= cfg.asset_play_min_nav_price) & (raw_bv.fillna(eff_nav) >= 100.0) & (bv_comp >= 0.0)
    # Or a real discount that book growth closes within the horizon (growth capped: a burst is not a pace).
    growth = bv_comp.clip(upper=cfg.asset_play_max_book_growth_cap)
    growing_into = (
        (eff_nav >= cfg.asset_play_min_nav_compounding)
        & (raw_bv.fillna(eff_nav) >= cfg.asset_play_min_nav_compounding)
        & (growth >= cfg.asset_play_min_book_growth)
        & (eff_nav * (1 + growth / 100.0) ** cfg.asset_play_horizon_years >= cfg.asset_play_min_nav_price)
    )

    interest_cover = _num(df, "Interest_Coverage").fillna(0.0)
    debt_ok = ~(_num(df, "NetDebt_to_EBIT") > cfg.asset_play_max_netdebt_ebit) | (
        interest_cover >= cfg.asset_play_min_interest_cover
    )
    non_fin_ok = (
        ~fin
        & debt_ok
        & (net_debt_pct <= cfg.asset_play_max_netdebt_nav_share * raw_nav.clip(lower=0.0))
        & ~(_num(df, "FCF_Positive_Share") < cfg.asset_play_min_fcf_pos_share)
        & (_num(df, "Quality_Score") >= cfg.asset_play_min_quality)
        & ((net_cash.fillna(0.0) >= 0.0) | (share_red > 0.0) | ~(_num(df, "Per_Share_Compounding") < 0.0))
    )
    bank_or_insurer = df.get("Business_Model", pd.Series(index=df.index, dtype=object)).isin({"bank", "insurer"})
    fin_ok = (
        fin
        & ~(_num(df, "Equity_to_Assets") < cfg.asset_play_min_equity_assets)
        & (_num(df, "Quality_Score") >= cfg.asset_play_min_fin_quality)
        & (~bank_or_insurer | (share_red >= cfg.asset_play_min_fin_share_reduction))
    )
    qualifies = (
        fit
        & (exceeds_now | growing_into)
        & (non_fin_ok | fin_ok)
        & (_num(df, "Value_Score") >= cfg.asset_play_min_value)
        & ~(_num(df, "EPV_Per_Share_Yield") <= 0.0)
        & ~(_num(df, "ROIC_Norm") <= 0.0)
        & (share_red >= cfg.asset_play_min_share_reduction)
    ).astype(bool)

    work = df.assign(
        Effective_NAV_to_Price=eff_nav,
        Net_Cash_to_Cap=net_cash.fillna(0.0).where(~fin, 0.0),
        Book_Per_Share_Compounding=bv_comp,
        Share_Reduction_Rate=share_red,
    )
    if reference is None:
        ref = work[comparable(work)]
    else:
        ref_fin = _flag(reference, "Financial_Mode")
        ref_raw_nav = _num(reference, "NAV_to_Price").fillna(_num(reference, "BV_to_Price"))
        ref = reference.assign(
            Effective_NAV_to_Price=_num(reference, "Effective_NAV_to_Price").fillna(ref_raw_nav),
            Net_Cash_to_Cap=_num(reference, "Net_Cash_to_Cap").fillna(0.0).where(~ref_fin, 0.0),
            Book_Per_Share_Compounding=_num(reference, "Book_Per_Share_Compounding").fillna(
                _num(reference, "Book_Compounding")
            ),
            Share_Reduction_Rate=_num(reference, "Share_Reduction_Rate"),
        )
    pct = lambda col: criterion_percentile(work[col], "top", cfg.winsor_pct, ref[col]).fillna(0.5)  # noqa: E731
    score = (
        0.35 * pct("Effective_NAV_to_Price")
        + 0.25 * pct("Net_Cash_to_Cap")
        + 0.20 * pct("Share_Reduction_Rate")
        + 0.20 * pct("Book_Per_Share_Compounding")
    ) * 100.0
    return qualifies, score.where(qualifies)


def value_quality_correlation(df: pd.DataFrame, profitable_only: bool = True, operating_only: bool = True) -> float:
    """Spearman correlation of Value and Quality scores in this run (NaN with fewer than 3 companies).

    Excludes non-profitable companies by default: unprofitable companies are typically low-value
    and low-quality, distorting the market correlation upward. Excludes banks, insurers and holdings
    by default too (user's choice, Oct 28, 2026): their Value (profit / market cap = ROE / price-to-book)
    and Quality (ROE) share ROE, so they correlate by arithmetic (+0.60 vs +0.17 for operating companies).
    """
    if not {"Value_Score", "Quality_Score"} <= set(df.columns):
        return float("nan")
    mask = comparable(df)
    if operating_only and (mask & ~_flag(df, "Financial_Mode")).sum() >= 3:
        mask &= ~_flag(df, "Financial_Mode")
    if profitable_only and "FCFy_Norm" in df.columns:
        prof_mask = mask & (pd.to_numeric(df["FCFy_Norm"], errors="coerce") >= 0)
        if prof_mask.sum() >= 3:
            mask = prof_mask
    pair = df.loc[mask, ["Value_Score", "Quality_Score"]].dropna()
    return float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 3 else float("nan")


def top_n(df: pd.DataFrame, rank_col: str, n: int = 5) -> pd.DataFrame:
    """The n best `trusted` companies by a rank column: suspect or thin data, cyclicals at peak margins, distress,
    collapsing earnings, kept excluded-industry holdings and open model reviews stay in the full table but not in the
    headline lists."""
    if rank_col not in df.columns:
        return df.head(0)
    return df[trusted(df)].dropna(subset=[rank_col]).sort_values(rank_col).head(n)
