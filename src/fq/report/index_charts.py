"""Plotly figures of the front page (the Value vs Quality map, the distributions) and of the backward and forward
test page. Kept apart from `charts` because no company page draws them: editing them leaves the 8,000 company
pages as they are (`html.PAGE_INPUTS`)."""

import json
from functools import cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fq import forward
from fq.report.charts import _company_div, _div
from fq.report.format import benchmark, fmt
from fq.report.links import slug
from fq.report.pills import OWNER_LED_MIN
from fq.report.tiles import unreliable

QUADRANT_JS = """
<script>
(function () {
  const plotDiv = document.getElementById('quadrant_plot');
  const meta = %s;
  const usBox = document.getElementById('us-companies-checkbox');
  const lossBox = document.getElementById('profitable-checkbox');
  const held = new Set(window.FQ_HELD || []);
  function update() {
    const holdings = document.body.classList.contains('holdings-mode');  // the page's Holdings switch
    const opacities = [];
    meta.forEach(function (m) {
      const custom = plotDiv.data[m.trace].customdata;
      if (!custom) return;
      opacities.push(custom.map(function (c) {
        if ((c[2] === 1 && !usBox.checked) || (c[1] !== 1 && !lossBox.checked)) return 0;
        return holdings && !held.has(c[0]) ? 0.12 : (m.opacity || 0.75);
      }));
    });
    Plotly.restyle(plotDiv, {'marker.opacity': opacities}, meta.map(function (m) { return m.trace; }));
  }
  usBox.addEventListener('change', update);
  lossBox.addEventListener('change', update);
  document.addEventListener('fq-holdings', update);
  plotDiv.on('plotly_click', function (ev) {
    if (ev.points.length && ev.points[0].customdata) {
      window.open('companies/' + ev.points[0].customdata[0] + '.html', '_blank');
    }
  });
})();
</script>
"""

QUADRANT_CONTROLS = """
<div class="filter-controls">
  <label><input type="checkbox" id="us-companies-checkbox" checked> Include US companies</label>
  <label><input type="checkbox" id="profitable-checkbox" checked> Include non-profitable</label>
</div>
"""


def quadrant_html(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> str:
    """Value-vs-Quality scatter with US / profitability toggles; dots link to company pages."""
    built = quadrant_figure(df, bench=bench)
    if built is None:
        return "<p>No valid data for the quadrant plot.</p>"
    fig, meta = built
    return f"<div>{QUADRANT_CONTROLS}{_div(fig, 'quadrant_plot')}{QUADRANT_JS % json.dumps(meta)}</div>"


BENCH_COLOR = "rgba(190, 190, 190, 0.25)"  # pale grey: the reference behind the run's colours
SP500_COLOR = "#2563eb"  # royal blue: S&P 500 index members
REST_COLOR = "#ea580c"  # warm orange/amber: rest of universe
MAX_DOT_PX = 30.0  # diameter of the largest company on the chart
MIN_DOT_PX = 3.0  # floor: caps span five orders of magnitude, so the smallest would otherwise vanish


def dot_diameter(cap: float, cap_ref: float) -> float:
    """Dot diameter in px for a market cap in $B: area proportional to cap, `cap_ref` = MAX_DOT_PX."""
    if not cap_ref > 0 or not cap > 0:
        return MIN_DOT_PX
    return max(MIN_DOT_PX, MAX_DOT_PX * (cap / cap_ref) ** 0.5)


@cache
def _sp500_tickers() -> frozenset[str]:
    try:
        from fq.universe import benchmark_part

        raw = set(benchmark_part("sp500"))
        return frozenset(raw | {t.replace("-", ".") for t in raw} | {t.replace(".", "-") for t in raw})
    except Exception:
        return frozenset()


def _bench_cloud(df: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
    """Benchmark companies with both scores that are not in the run (drawn behind it)."""
    if bench is None or not {"Value_Score", "Quality_Score", "Ticker"} <= set(bench.columns):
        return pd.DataFrame()
    cloud = bench[~bench["Ticker"].isin(df["Ticker"]) & ~unreliable(bench)]
    return cloud.dropna(subset=["Value_Score", "Quality_Score"])


def _plain(values, digits: int = 2) -> list:
    """Values rounded and as a list, so Plotly writes them as short JSON numbers: a Series becomes 8-byte binary
    (base64) that does not compress, ~350 KB of the front page's ~430 KB compressed (Oct 31, 2026)."""
    return np.round(pd.to_numeric(pd.Series(values), errors="coerce").astype(float), digits).tolist()


def quadrant_figure(
    df: pd.DataFrame, labels: bool = False, bench: pd.DataFrame | None = None
) -> tuple[go.Figure, list[dict]] | None:
    """The quadrant figure and per-trace metadata; `labels` writes tickers next to dots (for print).

    Axes are the 0-100 scores, so 50 is the median of the benchmark (or of the run without one);
    benchmark companies outside the run form a grey cloud behind it. Companies whose figures may be wrong
    (`tiles.unreliable`) are left out of the dots, the cloud and the correlation (user's choice, Nov 4, 2026: a
    misfiled cash flow put them at Value 99, top of the map).
    """
    needed = ["Value_Score", "Quality_Score", "Market Cap", "FCFy_Norm", "shortName", "Ticker", "country"]
    if not all(c in df.columns for c in needed):
        return None
    d = df[~unreliable(df)].dropna(subset=needed).copy()
    if d.empty:
        return None

    n = len(d)
    sp500 = _sp500_tickers()
    d["x"] = d["Quality_Score"]
    d["y"] = d["Value_Score"]
    d["is_sp500"] = d["Ticker"].isin(sp500)
    d["is_us"] = (d["country"] == "United States") | d["is_sp500"]
    d["profitable"] = (d["FCFy_Norm"] >= 0).astype(int)
    d["slug"] = d["Ticker"].map(slug)

    cloud = _bench_cloud(df, bench)
    cloud_caps = (
        pd.to_numeric(cloud["Market Cap"], errors="coerce") if "Market Cap" in cloud else pd.Series(dtype=float)
    )
    cap_ref = float(
        max(d["Market Cap"].max(), cloud_caps.max() if len(cloud_caps) else 0)
    )  # one scale for run and cloud
    d["marker_size"] = [dot_diameter(float(c), cap_ref) for c in d["Market Cap"]]

    fig = go.Figure()
    meta = []
    if not cloud.empty:
        cloud_sizes = [dot_diameter(c, cap_ref) if pd.notna(c) else MIN_DOT_PX
                       for c in cloud_caps.reindex(cloud.index)]  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=_plain(cloud["Quality_Score"]), y=_plain(cloud["Value_Score"]), mode="markers", name="Benchmark",
                marker=dict(size=_plain(cloud_sizes, 1), color=BENCH_COLOR), hoverinfo="none", showlegend=False,
            )
        )  # fmt: skip

    # Two distinct categorical traces: Rest of Universe first, S&P 500 on top
    traces_config = [
        (False, "Rest of universe", REST_COLOR, 0.55, 2),
        (True, "S&P 500", SP500_COLOR, 0.60, 1),
    ]

    for is_sp, label, color, opacity, rank in traces_config:
        seg = d[d["is_sp500"] == is_sp]
        if seg.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=_plain(seg["x"]), y=_plain(seg["y"]), mode="markers+text" if labels else "markers", name=label,
                legendrank=rank,
                textposition="top center", textfont=dict(size=9),
                customdata=[[a, int(b), int(c)] for a, b, c in zip(seg["slug"], seg["profitable"], seg["is_us"], strict=True)],
                text=seg["Ticker"] if labels else None, hoverinfo="none",
                marker=dict(
                    size=_plain(seg["marker_size"], 1),
                    color=color,
                    opacity=opacity,
                    line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)"),
                ),
            )
        )  # fmt: skip
        meta.append({"is_sp500": is_sp, "opacity": opacity, "trace": len(fig.data) - 1})

    d_corr = d[d["profitable"] == 1]
    if len(d_corr) < 3:
        d_corr = d
    n_corr = len(d_corr)
    corr = np.corrcoef(d_corr["x"], d_corr["y"])[0, 1] if n_corr >= 3 else float("nan")
    if n_corr >= 3:
        slope, intercept = np.polyfit(d_corr["x"], d_corr["y"], 1)
        xs = np.array([d_corr["x"].min(), d_corr["x"].max()])
        fig.add_trace(
            go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Best fit",
                       line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dash"), hoverinfo="none",
                       showlegend=False)
        )  # fmt: skip

    # The quadrants: solid median lines, no shaded corner (user's choice, Oct 31, 2026: the green confused).
    mid, top = 50.0, 102.0
    fig.add_vline(x=mid, line_width=2, line_color="rgba(128,128,128,0.85)", layer="below")
    fig.add_hline(y=mid, line_width=2, line_color="rgba(128,128,128,0.85)", layer="below")
    ref = benchmark(df)
    versus = f", vs {ref[0]} n={ref[1]}" if ref else ""
    fig.update_layout(
        title=dict(
            text=f"<b>Value vs. Quality quadrant (n={n}{versus})</b>",
            subtitle=dict(
                text="Dot area proportional to market cap • blue = S&P 500 • orange = rest of the universe",
                font=dict(size=11, color="#667085"),
            ),
        ),
        xaxis_title="<b>Quality score →</b>", yaxis_title="<b>Value score →</b>",
        xaxis_range=[0, top], yaxis_range=[0, top],
        showlegend=True,
        legend=dict(
            orientation="h",
            x=1.0,
            xanchor="right",
            y=1.04,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",  # needed for plotly_click; hoverinfo="none" on the traces shows no tooltip
        dragmode=False,
        margin=dict(l=40, r=40, b=40, t=85),
        annotations=[dict(x=top * 0.98, y=top * 0.05, text=f"ρ = {fmt(corr, '.2f')}", showarrow=False,
                          font=dict(size=14, color="grey"), xanchor="right", yanchor="bottom")],
    )  # fmt: skip
    return fig, meta


# (column, title, axis range, bin size[, scale, (guide line, its label)])
HISTOGRAMS = [
    ("FCFy_Norm", "Normalised FCF yield (%)", (-10, 50), 2),  # user's choice: deeper negatives cut off
    ("ROIC_Norm", "Normalised ROIC (%)", (-50, 100), 10),
    ("Revenue_Growth_CAGR", "Revenue CAGR (%)", (-20, 40), 2),
    # Founder- and family-run against hired managers (user's choice, Nov 1, 2026: Yahoo has it for nearly every
    # company, unlike the holder tables); shown, never scored. The guide is the "owner-led" pill's bar.
    ("heldPercentInsiders", "Insider ownership (%)", (0, 100), 5, 100, (OWNER_LED_MIN, "Owner-led")),
]
HOLDINGS_COLOR = "#f97316"  # vivid orange for holdings overlays


def _counted(title: str, data: pd.Series, ref: pd.Series | None = None, name: str = "", axis=None) -> str:
    """A distribution's title with how many companies it rests on (user's request, Oct 31, 2026: always say n), the
    benchmark's too when drawn, and how many fall outside the axis' range when any do (they count, but are not seen)."""
    beyond = int(((data < axis[0]) | (data > axis[1])).sum()) if axis is not None else 0
    counts = f"n = {len(data):,}" + (f" · {name} n = {len(ref):,}" if ref is not None and len(ref) else "")
    counts += f" · {beyond:,} beyond the axis" if beyond else ""
    return f"<b>{title}</b><br><span style='font-size:12px'>{counts}</span>"


def histogram_divs(
    df: pd.DataFrame,
    bench: pd.DataFrame | None = None,
    held: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """[(title, html div)] for the distribution cards; the benchmark (grey) is overlaid when given;
    holdings (orange) can be toggled via the Holdings switch.
    """
    ref = benchmark(df)
    held_df = df[df["Ticker"].isin(held)] if held and "Ticker" in df.columns else pd.DataFrame()
    has_held = not held_df.empty and len(held_df) < len(df)
    out = []
    for i, (col, title, xrange, size, *extra) in enumerate(HISTOGRAMS):
        if col not in df.columns or df[col].dropna().empty:
            continue
        scale, guide = (*extra, 1, None)[:2]

        def values(frame: pd.DataFrame | None, use: bool = True, col=col, scale=scale) -> pd.Series:
            if not use or frame is None or col not in frame.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(frame[col], errors="coerce").dropna() * scale

        data, ref_data, held_data = values(df), values(bench), values(held_df, has_held)
        # Bins only over the axis (Oct 31, 2026: Plotly bins the whole range, and one FCF yield of -262,119% made
        # 131,449 bars, which hung phones and crashed headless Chrome); the title still counts every company and
        # says how many lie beyond the axis.
        lo, hi = xrange
        bins = dict(start=lo, end=hi, size=size)
        fig = go.Figure()
        if not ref_data.empty:
            fig.add_trace(go.Histogram(x=_plain(ref_data[ref_data.between(lo, hi)]), xbins=bins, histnorm="percent",
                                       name=ref[0] if ref else "Benchmark", marker_color=BENCH_COLOR))  # fmt: skip
            fig.add_vline(x=ref_data.median(), line_width=1.5, line_dash="dot", line_color="grey",
                          annotation_text=f"{ref[0] if ref else 'Benchmark'}: {ref_data.median():.1f}",
                          annotation_position="top left")  # fmt: skip
        fig.add_trace(go.Histogram(x=_plain(data[data.between(lo, hi)]), xbins=bins, name="This run",
                                   histnorm="percent" if not ref_data.empty else None, opacity=0.75))  # fmt: skip
        fig.add_vline(x=data.median(), line_width=2, line_dash="dash", line_color="green",
                      annotation_text=f"Median: {data.median():.2f}", annotation_position="top right")  # fmt: skip
        if guide is not None:
            fig.add_vline(x=guide[0], line_width=1.5, line_dash="dot", line_color="#d97706",
                          annotation_text=f"{guide[1]} ≥ {guide[0]:g}", annotation_position="bottom right")  # fmt: skip
        if not held_data.empty:
            fig.add_trace(go.Histogram(x=_plain(held_data[held_data.between(lo, hi)]), xbins=bins, name="Holdings",
                                       marker_color=HOLDINGS_COLOR, histnorm="percent", opacity=0.75,
                                       visible=False))  # fmt: skip
            held_med = held_data.median()
            fig.add_vline(x=held_med, line_width=2, line_dash="dashdot", line_color=HOLDINGS_COLOR,
                          annotation_text=f"Holdings: {held_med:.2f}", annotation_position="bottom right",
                          visible=False, annotation=dict(visible=False))  # fmt: skip
        fig.update_layout(title=_counted(title, data, ref_data, ref[0] if ref else "Benchmark", xrange), xaxis_title=title,
                          yaxis_title="% of companies" if not ref_data.empty else "Count", barmode="overlay",
                          height=350, showlegend=not ref_data.empty, margin=dict(l=20, r=20, t=70, b=20),
                          legend=dict(orientation="h", y=-0.25))  # fmt: skip
        fig.update_xaxes(range=list(xrange))
        out.append((title, _div(fig, f"hist_{i}")))
    cap = market_cap_figure(df, bench, held=held)
    if cap is not None:
        out.append((CAP_TITLE, _div(cap, "hist_cap")))
    return out


CAP_TITLE = "Market cap (B$, log scale)"
RUN_COLOR = "#636efa"  # the colour the other histograms get from the chart template
CAP_STEP = 0.25  # log10 width of a bin: each bin spans a factor of 1.78


def _money(value: float) -> str:
    """A market cap in B$ with sensible digits: 0.35, 4.2, 38, 1,250."""
    return f"{value:,.2f}" if value < 1 else f"{value:,.1f}" if value < 10 else f"{value:,.0f}"


def market_cap_bins(caps: pd.Series, reference: pd.Series | None = None) -> np.ndarray:
    """Bin edges (log10 of B$, `CAP_STEP` apart) covering `caps` and `reference`, so both are counted in the same bins."""
    values = pd.concat([caps, reference]) if reference is not None else caps
    values = pd.to_numeric(values, errors="coerce")
    logs = np.log10(values[values > 0])
    low, high = np.floor(logs.min() / CAP_STEP) * CAP_STEP, np.ceil(logs.max() / CAP_STEP) * CAP_STEP
    return np.arange(low, high + CAP_STEP / 2, CAP_STEP)


def market_cap_figure(
    df: pd.DataFrame,
    bench: pd.DataFrame | None = None,
    held: frozenset[str] = frozenset(),
) -> go.Figure | None:
    """How the companies spread on market cap (B$): bins of equal ratio on a log scale, since caps run from under
    1 B$ to thousands. The run in colour; the benchmark in grey (as % of its companies) when given. Hover gives each
    bin's range and count. Portfolio holdings (orange) can be toggled via the Holdings switch."""
    if "Market Cap" not in df.columns:
        return None
    caps = pd.to_numeric(df["Market Cap"], errors="coerce")
    caps = caps[caps > 0]
    if caps.empty:
        return None
    ref = pd.to_numeric(bench["Market Cap"], errors="coerce") if bench is not None and "Market Cap" in bench else None
    ref = ref[ref > 0] if ref is not None else None
    ref = ref if ref is not None and not ref.empty else None

    held_df = df[df["Ticker"].isin(held)] if held and "Ticker" in df.columns else pd.DataFrame()
    held_caps = (
        pd.to_numeric(held_df["Market Cap"], errors="coerce")
        if not held_df.empty and "Market Cap" in held_df
        else pd.Series(dtype=float)
    )
    held_caps = held_caps[held_caps > 0]
    has_held = not held_caps.empty and len(held_caps) < len(caps)

    edges = market_cap_bins(caps, ref)
    centers = (edges[:-1] + edges[1:]) / 2

    def bars(
        values: pd.Series, name: str, color: str, percent: bool, opacity: float, visible: bool | None = None
    ) -> go.Bar:
        counts, _ = np.histogram(np.log10(values), bins=edges)
        heights = counts / len(values) * 100 if percent else counts
        label = "holdings" if name == "Holdings" else "companies"
        text = [f"{_money(10 ** lo)} - {_money(10 ** hi)} B$<br>{n:,} {label} ({n / len(values):.0%})"
                for lo, hi, n in zip(edges[:-1], edges[1:], counts, strict=True)]  # fmt: skip
        pct_heights = counts / len(values) * 100
        customdata = np.column_stack([counts, pct_heights])
        kwargs = dict(
            x=centers,
            y=heights,
            width=CAP_STEP,
            name=name,
            marker_color=color,
            opacity=opacity,
            text=text,
            textposition="none",
            hoverinfo="text",
            customdata=customdata,
        )
        if visible is not None:
            kwargs["visible"] = visible
        return go.Bar(**kwargs)

    named = benchmark(df)
    fig = go.Figure()
    if ref is not None:
        fig.add_trace(bars(ref, named[0] if named else "Benchmark", BENCH_COLOR, True, 1.0))
        fig.add_vline(x=np.log10(ref.median()), line_width=1.5, line_dash="dot", line_color="grey",
                      annotation_text=f"{named[0] if named else 'Benchmark'}: {_money(ref.median())}",
                      annotation_position="top left")  # fmt: skip
    fig.add_trace(bars(caps, "This run", RUN_COLOR, True, 0.75))
    fig.add_vline(x=np.log10(caps.median()), line_width=2, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {_money(caps.median())}", annotation_position="top right")  # fmt: skip
    if has_held:
        fig.add_trace(bars(held_caps, "Holdings", HOLDINGS_COLOR, True, 0.75, visible=False))
        fig.add_vline(x=np.log10(held_caps.median()), line_width=2, line_dash="dashdot", line_color=HOLDINGS_COLOR,
                      annotation_text=f"Holdings: {_money(held_caps.median())}", annotation_position="bottom right",
                      visible=False, annotation=dict(visible=False))  # fmt: skip
    ticks = np.arange(np.floor(edges[0]), np.ceil(edges[-1]) + 1)
    fig.update_layout(title=_counted(CAP_TITLE, caps, ref, named[0] if named else "Benchmark"),
                      xaxis_title="Market cap (B$, USD billions)",
                      yaxis_title="% of companies", barmode="overlay", bargap=0.05,
                      height=350, showlegend=ref is not None, margin=dict(l=20, r=20, t=70, b=20),
                      legend=dict(orientation="h", y=-0.25))  # fmt: skip
    fig.update_xaxes(tickvals=list(ticks), ticktext=[f"{10**t:,g}" for t in ticks], range=[edges[0], edges[-1]])
    return fig


FORWARD_COLORS = ("#f97316", "#10b981", "#a855f7", "#ef4444", "#eab308", "#06b6d4", "#ec4899", "#84cc16", "#8b5cf6",
                  "#14b8a6", "#f43f5e", "#d97706", "#0ea5e9", "#65a30d")  # fmt: skip
FORWARD_PORTFOLIO = "#2563eb"
FORWARD_BENCHMARKS = {"SPY": "#64748b", "VT": "#475569"}
FORWARD_ID = "forward_chart"


def _forward_labels(dates: pd.DatetimeIndex) -> list[str]:
    """The start's month ("Start Oct-26"), then a month for each month-end ("Nov-26"), any other date by its day."""
    month_end = dates + pd.offsets.MonthEnd(0) == dates
    later = [f"{d:%b-%y}" if end else f"{d:%d %b}" for d, end in zip(dates[1:], month_end[1:], strict=True)]
    return [f"Start {dates[0]:%b-%y}", *later]


def forward_figure(values: pd.DataFrame) -> go.Figure:
    """Cumulative USD total return (%) by month: one line per list and the portfolio; each benchmark as a dashed line
    and a horizontal line at its latest return, labelled at the right edge."""
    x = _forward_labels(values.index)
    fig = go.Figure()
    bench_labels = {label: ticker for ticker, label in forward.BENCHMARKS.items()}
    colors = iter(FORWARD_COLORS * 3)
    for name in values.columns:
        y = values[name]
        if name in bench_labels:
            continue
        color = FORWARD_PORTFOLIO if name == forward.PORTFOLIO_LIST else next(colors)
        width = 3 if name == forward.PORTFOLIO_LIST else 2
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers", line=dict(color=color, width=width),
                                 marker=dict(size=5), hovertemplate="%{y:+.1f}%"))  # fmt: skip
    for label, ticker in bench_labels.items():
        if label not in values.columns or values[label].isna().all():
            continue
        y, color = values[label], FORWARD_BENCHMARKS[ticker]
        fig.add_trace(go.Scatter(x=x, y=y, name=label, mode="lines+markers", hovertemplate="%{y:+.1f}%",
                                 line=dict(color=color, width=3.5, dash="dash"), marker=dict(size=6)))  # fmt: skip
        last = y.dropna().iloc[-1]
        fig.add_hline(y=last, line=dict(color=color, width=1.2, dash="dot"))
        fig.add_annotation(x=1, xref="paper", xanchor="right", y=last, yref="y", yanchor="bottom", yshift=1,
                           text=f"{ticker} {last:+.1f}%", showarrow=False, font=dict(size=11, color=color))  # fmt: skip
    fig.add_hline(y=0, line=dict(color="#98a2b3", width=1))
    fig.update_layout(height=520, margin=dict(l=50, r=20, t=10, b=40), hovermode="x unified",
                      legend=dict(orientation="h", x=0, xanchor="left", y=-0.12, yanchor="top", bgcolor="rgba(0,0,0,0)"),
                      xaxis=dict(type="category", title=None),
                      yaxis=dict(title="Cumulative total return, USD", ticksuffix="%", zeroline=False))  # fmt: skip
    return fig


def forward_html(values: pd.DataFrame, div_id: str = FORWARD_ID) -> str:
    """The backward or forward test's chart as an embeddable div (`div_id` apart when both are on the page)."""
    return _company_div(forward_figure(values), div_id)
