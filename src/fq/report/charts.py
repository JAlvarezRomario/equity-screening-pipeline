"""Plotly figures of the company pages, and the helpers every chart shares (`index_charts` has the front page's
and the backward and forward test's). Figures are embedded as interactive divs (no image export). Every file here
is an input of each company page (`html.PAGE_INPUTS`): a change re-renders them all, so keep other charts out."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fq.metrics.long_history import parse_history
from fq.report.format import chart_years, period_label, show_nav_chart
from fq.settings import Settings

log = logging.getLogger(__name__)

PLOT_CONFIG = {"responsive": True, "displaylogo": False}


# Every chart on a transparent background in the page's font, so it sits in its card in either theme (main.js
# recolours text and grid lines when the theme changes).
PAGE_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    autosize=True,
    font=dict(family='system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif', color="#344054"),
)


def _div(fig: go.Figure, div_id: str) -> str:
    fig.update_layout(**PAGE_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config=PLOT_CONFIG)


SCORE_COLORS = {"Value": "#2e86de", "Quality": "#10ac84"}


def _per_share_yield(shares, price, value, absolute: bool = False) -> float:
    """Per-share value as % of the current price (dividends / capex are stored negative)."""
    if not (pd.notna(shares) and shares > 0 and pd.notna(price) and price > 0):
        return np.nan
    if absolute:
        value = abs(value) if pd.notna(value) else 0
    return value / shares / price * 100


def _relative_x_axes(fig: go.Figure, row: pd.Series, codes: list[str]) -> None:
    """Same category slots for every company; tick text shows this company's real period ends."""
    labels = [period_label(row, c) for c in codes]
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=codes, tickvals=codes, ticktext=labels)


# Company charts: mid-tone colours that read on the light and the dark theme, titles in the page (not the figure),
# one hover box per year, no toolbar.
COMPANY_COLORS = {"owner": "#3b82f6", "income": "#94a3b8", "capex": "#f87171", "dividends": "#a78bfa",
                  "power": "#f59e0b", "guide": "#98a2b3", "treasury": "#64748b", "roic": "#10b981", "margin": "#38bdf8",
                  "revenue": "#3b82f6", "shares": "#f97316", "book": "#fb923c", "tangible": "#3b82f6"}  # fmt: skip
COMPANY_CONFIG = {"responsive": True, "displaylogo": False, "displayModeBar": False}
_LEGEND = dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)")
_PCT = "%{y:.1f}%"


def _company_div(fig: go.Figure, div_id: str) -> str:
    fig.update_layout(**PAGE_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config=COMPANY_CONFIG)


def _line_label(fig: go.Figure, y: float, text: str, color: str) -> None:
    """A label just above a horizontal line at the plot's left edge, inside it (fits every width; no legend)."""
    fig.add_annotation(x=0, xref="paper", xanchor="left", xshift=4, y=y, yref="y", yanchor="bottom", yshift=1,
                       text=text, showarrow=False, font=dict(size=11, color=color))  # fmt: skip


def _guide(fig: go.Figure, y: float, text: str | None, color: str, dash: str = "dot", width: float = 1.0) -> None:
    """A horizontal guide line, labelled unless `text` is None."""
    fig.add_hline(y=y, line=dict(color=color, dash=dash, width=width), layer="below")
    if text:
        _line_label(fig, y, text, color)


def _row_years(df: pd.DataFrame, ticker: str, periods: list[str]) -> tuple[pd.Series, list[str]]:
    row = df[df["Ticker"] == ticker].iloc[0]
    return row, chart_years(row, periods)


def _earnings_based(row: pd.Series) -> bool:
    """Financial and float models: owner earnings are net income, already the parent's."""
    return row.get("Financial_Mode") == True or row.get("Business_Model") == "float"  # noqa: E712


def _shares(row: pd.Series, y: str, cfg: Settings) -> float:
    shares = row.get(f"Ordinary Shares Number_{y}", np.nan)
    if pd.isna(shares):
        shares = row.get(f"Share Issued_{y}", np.nan)
    if pd.isna(shares) and y in ("TTM", cfg.latest):
        shares = row.get("Manual_Shares", np.nan)
    return shares


def company_charts_html(
    df: pd.DataFrame, ticker: str, cfg: Settings, periods: list[str], treasury_yield: float = 4.25
) -> dict[str, str]:
    """The company page's charts as HTML divs: earnings, returns and business always, NAV and the long share-count and
    dividend history when shown."""
    out = {"earnings": _company_div(company_chart_figure(df, ticker, cfg, periods, treasury_yield=treasury_yield), "company_chart"),
           "returns": _company_div(returns_figure(df, ticker, cfg, periods), "returns_chart"),
           "business": _company_div(business_figure(df, ticker, cfg, periods), "business_chart")}  # fmt: skip
    nav = nav_chart_figure(df, ticker, cfg, periods)
    out["nav"] = _company_div(nav, "nav_chart") if nav is not None else ""
    history = long_history_figure(df, ticker)
    out["long_history"] = _company_div(history, "long_history_chart") if history is not None else ""
    return out


def company_chart_figure(
    df: pd.DataFrame, ticker: str, cfg: Settings, periods: list[str], treasury_yield: float = 4.25
) -> go.Figure:
    """What a share earns each year, % of today's price: owner earnings, clean net income, capex and dividends per
    share, with the normalised earnings power, 10Y Treasury yield and 15% guide labelled at their right end."""
    row, years = _row_years(df, ticker, periods)
    price = row.get("currentPrice", np.nan)
    # Owner earnings (`metrics.valuation._value`) are consolidated except in financial and float mode (net income,
    # already the parent's): a share owns the parent's part.
    earnings_based = _earnings_based(row)
    parent = 1.0 if earnings_based else pd.to_numeric(row.get("Parent_Share"), errors="coerce")
    parent = parent if pd.notna(parent) else 1.0
    owner_y, eps_y, div_y, capex_y = [], [], [], []
    for y in years:
        shares = _shares(row, y, cfg)
        owner = pd.to_numeric(row.get(f"Owner_Earnings_{y}", np.nan), errors="coerce")
        owner_y.append(_per_share_yield(shares, price, owner * parent if pd.notna(owner) else np.nan))
        clean = row.get(f"Net_Income_Clean_{y}", np.nan)
        eps_y.append(_per_share_yield(shares, price, clean if pd.notna(clean) else row.get(f"Net Income_{y}", np.nan)))
        div_y.append(_per_share_yield(shares, price, row.get(f"Cash Dividends Paid_{y}", np.nan), True))
        capex_y.append(_per_share_yield(shares, price, row.get(f"Capital Expenditure_{y}", np.nan), True))

    c = COMPANY_COLORS
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=owner_y, name="Owner earnings yield", marker_color=c["owner"],
                         hovertemplate=_PCT))  # fmt: skip
    fig.add_trace(go.Bar(x=years, y=eps_y, name="Clean net income yield", marker_color=c["income"],
                         hovertemplate=_PCT))  # fmt: skip
    if not earnings_based:  # a lender's or insurer's capex says little
        fig.add_trace(go.Bar(x=years, y=capex_y, name="Capex yield", marker_color=c["capex"], hovertemplate=_PCT))
    fig.add_trace(go.Bar(x=years, y=div_y, name="Dividend yield", marker_color=c["dividends"], hovertemplate=_PCT))
    norm = pd.to_numeric(row.get("EPV_Per_Share_Yield"), errors="coerce")
    # 10-year Treasury yield guide: distinct dash and visible color, tasteful and clear.
    ty_label = None if pd.notna(norm) and abs(norm - treasury_yield) < 1.2 else f"10Y {treasury_yield:.2f}%"
    _guide(fig, treasury_yield, ty_label, c["treasury"], dash="dash", width=1.5)
    # 15% hurdle guide
    h15_label = None if pd.notna(norm) and abs(norm - 15) < 1.5 else "15%"
    _guide(fig, 15, h15_label, c["guide"], dash="dot", width=1.0)
    if pd.notna(norm):
        fig.add_trace(go.Scatter(x=[years[0], years[-1]], y=[norm] * 2, mode="lines", showlegend=False,
                                 name=f"Owner yield / share, today ({norm:.1f}%)", hoverinfo="skip",
                                 line=dict(color=c["power"], width=2.5)))  # fmt: skip
        _line_label(fig, norm, f"<b>Owner yield / share {norm:.1f}%</b>", c["power"])
    fig.update_layout(barmode="group", bargap=0.25, height=420, hovermode="x unified", legend=_LEGEND,
                      margin=dict(t=40, b=40, l=50, r=20))  # fmt: skip
    _relative_x_axes(fig, row, years)
    fig.update_yaxes(ticksuffix="%", zeroline=True)
    return fig


def returns_figure(df: pd.DataFrame, ticker: str, cfg: Settings, periods: list[str]) -> go.Figure:
    """ROIC (ROE in financial mode) and operating margin per year, %."""
    row, years = _row_years(df, ticker, periods)
    series = lambda prefix: [row.get(f"{prefix}_{y}", np.nan) for y in years]  # noqa: E731
    c = COMPANY_COLORS
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=series("ROIC"), name="ROE" if _earnings_based(row) else "ROIC",
                             mode="lines+markers", line=dict(color=c["roic"], width=2.5),
                             connectgaps=True, hovertemplate=_PCT))  # fmt: skip
    if not _earnings_based(row):
        fig.add_trace(go.Scatter(x=years, y=series("operatingMargins"), name="Operating margin", mode="lines+markers",
                                 line=dict(color=c["margin"], width=2),
                                 connectgaps=True, hovertemplate=_PCT))  # fmt: skip
    fig.update_layout(height=320, hovermode="x unified", legend=_LEGEND, margin=dict(t=40, b=40, l=50, r=20))
    _relative_x_axes(fig, row, years)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(ticksuffix="%", rangemode="tozero", automargin=True)
    return fig


def business_figure(df: pd.DataFrame, ticker: str, cfg: Settings, periods: list[str]) -> go.Figure:
    """Revenue (bars) and the share count (line, own axis); both axes start at 0 (user's rule, Oct 21, 2026), so a
    change looks as large as it is."""
    row, years = _row_years(df, ticker, periods)
    revenue = pd.to_numeric(pd.Series([row.get(f"Total Revenue_{y}", np.nan) for y in years]), errors="coerce")
    unit, scale = ("B", 1e9) if revenue.abs().max() >= 1e9 else ("M", 1e6)
    shares = [s / 1e6 if pd.notna(s) else np.nan for s in (_shares(row, y, cfg) for y in years)]
    c = COMPANY_COLORS
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=years, y=revenue / scale, name=f"Revenue ({unit}$)", marker_color=c["revenue"],
                         opacity=0.55, hovertemplate="%{y:,.2f}"), secondary_y=False)  # fmt: skip
    fig.add_trace(go.Scatter(x=years, y=shares, name="Shares (M)", mode="lines+markers",
                             line=dict(color=c["shares"], width=2.5),
                             connectgaps=True, hovertemplate="%{y:,.1f}"), secondary_y=True)  # fmt: skip
    fig.update_layout(height=320, hovermode="x unified", legend=_LEGEND, bargap=0.35,
                      margin=dict(t=40, b=40, l=60, r=65))  # fmt: skip
    _relative_x_axes(fig, row, years)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(
        title_text=f"Revenue ({unit}$)",
        rangemode="tozero",
        showline=True,
        linecolor="rgba(16,24,40,0.18)",
        automargin=True,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Shares (M)",
        rangemode="tozero",
        showgrid=False,
        showline=True,
        linecolor="rgba(16,24,40,0.18)",
        tickmode="auto",
        nticks=5,
        tickformat=",.0f",
        automargin=True,
        secondary_y=True,
    )
    return fig


LONG_HISTORY_MIN_YEARS = 3  # either series needs this many years for the chart


def long_history_figure(df: pd.DataFrame, ticker: str) -> go.Figure | None:
    """Share count and dividend history, every calendar year Yahoo keeps (`Shares_Hist`, `DPS_Hist`): dividends per
    share as bars (left axis, quote currency), the share count as a line (right axis, from 0). None when neither series
    has `LONG_HISTORY_MIN_YEARS` years; a missing series is simply not drawn."""
    row = df[df["Ticker"] == ticker].iloc[0]
    shares, dps = parse_history(row.get("Shares_Hist")), parse_history(row.get("DPS_Hist"))
    has_shares, has_dps = len(shares) >= LONG_HISTORY_MIN_YEARS, len(dps) >= LONG_HISTORY_MIN_YEARS
    if not (has_shares or has_dps):
        return None
    currency = row.get("currency") if isinstance(row.get("currency"), str) else ""
    c = COMPANY_COLORS
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if has_dps:
        fig.add_trace(go.Bar(x=dps.index, y=dps.to_numpy(), name=f"Dividends per share ({currency})".replace(" ()", ""),
                             marker_color=c["dividends"], hovertemplate="%{y:,.4g}"), secondary_y=False)  # fmt: skip
    if has_shares:
        fig.add_trace(go.Scatter(x=shares.index, y=shares.to_numpy() / 1e6, name="Shares (M)", mode="lines+markers",
                                 line=dict(color=c["shares"], width=2.5), connectgaps=True,
                                 hovertemplate="%{y:,.1f}"), secondary_y=True)  # fmt: skip
    fig.update_layout(height=340, hovermode="x unified", legend=_LEGEND, bargap=0.3,
                      margin=dict(t=40, b=40, l=60, r=65))  # fmt: skip
    fig.update_xaxes(tickformat="d", automargin=True)
    fig.update_yaxes(title_text=f"Dividends / share ({currency})".replace(" ()", ""), rangemode="tozero",
                     automargin=True, visible=has_dps, secondary_y=False)  # fmt: skip
    fig.update_yaxes(title_text="Shares (M)", rangemode="tozero", showgrid=not has_dps, tickformat=",.0f",
                     automargin=True, visible=has_shares, secondary_y=True)  # fmt: skip
    return fig


def nav_chart_figure(df: pd.DataFrame, ticker: str, cfg: Settings, periods: list[str]) -> go.Figure | None:
    """Book value and tangible book per share as % of today's price. None below the threshold."""
    row, years = _row_years(df, ticker, periods)
    if not show_nav_chart(row):
        return None
    nav = [row.get(f"NAVPS_Pct_{y}", np.nan) for y in years]
    book = [row.get(f"BVPS_Pct_{y}", np.nan) for y in years]
    c = COMPANY_COLORS
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=book, name="Book value (common equity)", marker_color=c["book"],
                         hovertemplate="%{y:.0f}%"))  # fmt: skip
    fig.add_trace(go.Bar(x=years, y=nav, name="Tangible book (minority-adj.)", marker_color=c["tangible"],
                         hovertemplate="%{y:.0f}%"))  # fmt: skip
    fig.add_hline(y=100, line=dict(dash="dash", color=c["guide"]), annotation_text="100% = today's price",
                  annotation_position="top left", annotation_font=dict(size=11, color=c["guide"]))  # fmt: skip
    fig.update_layout(barmode="group", height=320, hovermode="x unified", legend=_LEGEND,
                      margin=dict(t=40, b=40, l=50, r=20))  # fmt: skip
    _relative_x_axes(fig, row, years)
    fig.update_yaxes(ticksuffix="%", rangemode="tozero", zeroline=True)
    return fig


# Forward test (`forward_html`): mid tones that read on both themes; the portfolio in the accent blue, the
# benchmarks in neutral greys.
