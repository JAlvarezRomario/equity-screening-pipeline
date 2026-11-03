"""Tables of the evaluation and the quality history (`fq evaluate`; not on the site)."""

import numpy as np
import pandas as pd

from fq.report.format import NA, fmt


def _one_method(df: pd.DataFrame, method: str | None) -> tuple[pd.DataFrame, str | None]:
    """The rows of one scoring method: `method` (today's) when it has results, else the one with most snapshots."""
    if df is None or df.empty or "Method" not in df.columns:
        return df, None
    if method not in set(df["Method"]):
        method = df.groupby("Method")["Snapshots"].max().idxmax()
    return df[df["Method"] == method], method


def history_table(df: pd.DataFrame | None) -> list[tuple[str, list[str], list[list[str]]]]:
    """(title, header, rows) for the instant Quality test (`history.quality_ic`): one row per measure, one column
    per outcome and lag, "IC (t)"."""
    if df is None or df.empty or df["IC"].isna().all():
        return []
    columns = list(dict.fromkeys(zip(df["Outcome"], df["Lag"], strict=True)))
    header = ["Measure", *(f"{o}, {lag} y" for o, lag in columns)]
    rows = []
    for name, group in df.groupby("Measure", sort=False):
        cells = []
        for outcome, lag in columns:
            hit = group[(group["Outcome"] == outcome) & (group["Lag"] == lag)]
            ic, t = (hit["IC"].iloc[0], hit["t"].iloc[0]) if len(hit) else (np.nan, np.nan)
            cells.append(NA if pd.isna(ic) else f"{ic:+.2f}" + (f" ({t:+.1f})" if pd.notna(t) else ""))
        rows.append([str(name), *cells])
    n = int(df["N"].median())
    lags = "-".join(str(lag) for lag in sorted(df["Lag"].unique()))
    title = f"Quality scored {lags} year(s) back vs what the statements did since: rank correlation (t), N ~{n}"
    return [(title, header, rows)]


def tracking_tables(
    lists: pd.DataFrame | None, ic: pd.DataFrame | None, method: str | None = None
) -> list[tuple[str, list[str], list[list[str]]]]:
    """(title, header, rows) for the forward-tracking card: excess return per list over the median company and
    over the market (`evaluate.MARKET`), and the information coefficient per measure with its Newey-West t, one
    column per horizon, for one scoring method (`_one_method`)."""
    out = []
    for title, df, value, spec in (
        ("Excess return over the median company (%)", lists, "Excess", "+.1f"),
        ("Excess return over ACWI, the global market (%)", lists, "Excess_Market", "+.1f"),
        ("Information coefficient (t)", ic, "IC", "+.2f"),
    ):
        key = "List" if value.startswith("Excess") else "Measure"
        df, shown = _one_method(df, method)
        if df is None or df.empty or value not in df.columns or df[value].isna().all():
            continue
        if shown is not None:
            title += f" - method {shown}" + ("" if shown == method else " (not today's)")
        months = sorted(df["Months"].unique())
        header = [key, *(f"{m} months" for m in months)]
        rows = []
        for name, group in df.groupby(key, sort=False):
            cells = []
            for m in months:
                hit = group[group["Months"] == m]
                if not len(hit):
                    cells.append(NA)
                    continue
                t = hit["IC_t"].iloc[0] if value == "IC" and "IC_t" in hit.columns else np.nan
                t_text = f" t {t:+.1f}" if pd.notna(t) else ""
                cells.append(f"{fmt(hit[value].iloc[0], spec)}{t_text} ({int(hit['Snapshots'].iloc[0])})")
            rows.append([str(name), *cells])
        out.append((title, header, rows))
    return out
