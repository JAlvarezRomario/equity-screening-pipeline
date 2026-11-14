"""Plotly figures of the front page (the Value vs Quality map, the distributions) and of the backward and forward
test page. Kept apart from `charts` because no company page draws them: editing them leaves the 8,000 company
pages as they are (`html.PAGE_INPUTS`)."""

import json
from functools import cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fq import forward
from fq.ranking import rankable
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
  const held = new Set(window.FQ_HELD || []);
  const watched = new Set(window.FQ_WATCHED || []);
  function update() {
    const holdings = document.body.classList.contains('holdings-mode');  // the page's Holdings switch
    const watchlist = document.body.classList.contains('watchlist-mode');  // the page's Watchlist switch
    const opacities = [];
    meta.forEach(function (m) {
      const custom = plotDiv.data[m.trace].customdata;
      if (!custom) return;
      opacities.push(custom.map(function (c) {
        if (c[1] === 1 && !usBox.checked) return 0;
        if (holdings) return held.has(c[0]) ? (m.opacity || 0.75) : 0.12;
        if (watchlist) return watched.has(c[0]) ? (m.opacity || 0.75) : 0.12;
        return m.opacity || 0.75;
      }));
    });
    Plotly.restyle(plotDiv, {'marker.opacity': opacities}, meta.map(function (m) { return m.trace; }));
  }
  usBox.addEventListener('change', update);
  document.addEventListener('fq-holdings', update);
  document.addEventListener('fq-watchlist', update);
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
</div>
"""


def quadrant_html(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> str:
    """Value-vs-Quality scatter with a US toggle; dots link to company pages."""
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
    cloud = bench[~bench["Ticker"].isin(df["Ticker"]) & ~unreliable(bench) & rankable(bench)]
    return cloud.dropna(subset=["Value_Score", "Quality_Score"])


def _plain(values, digits: int = 2) -> list:
    """Values rounded and as a list, so Plotly writes them as short JSON numbers: a Series becomes 8-byte binary
    (base64) that does not compress, ~350 KB of the front page's ~430 KB compressed (Oct 31, 2026)."""
    return np.round(pd.to_numeric(pd.Series(values), errors="coerce").astype(float), digits).tolist()


MAP_COLUMNS = ["Value_Score", "Quality_Score", "Market Cap", "shortName", "Ticker", "country"]


def map_points(df: pd.DataFrame) -> pd.DataFrame:
    """The companies the map draws: ranked ones only (`ranking.rankable`: profitable operating companies, user's
    choice, Nov 5, 2026), without those whose figures may be wrong (`tiles.unreliable`, Nov 4, 2026: a misfiled cash
    flow put them at Value 99), with every figure a dot needs. Its n, correlation and fit read these rows alone."""
    if not all(c in df.columns for c in MAP_COLUMNS):
        return df.head(0)
    return df[~unreliable(df) & rankable(df)].dropna(subset=MAP_COLUMNS)


def map_correlation(df: pd.DataFrame) -> float:
    """Spearman correlation of Value and Quality over the dots the map draws (`map_points`, user's choice, Nov 7,
    2026: the figure must describe what is in the chart); NaN with fewer than 3."""
    d = map_points(df)
    if len(d) < 3:
        return float("nan")
    return float(d[["Value_Score", "Quality_Score"]].corr(method="spearman").iloc[0, 1])


def quadrant_figure(
    df: pd.DataFrame, labels: bool = False, bench: pd.DataFrame | None = None
) -> tuple[go.Figure, list[dict]] | None:
    """The quadrant figure and per-trace metadata; `labels` writes tickers next to dots (for print).

    Axes are the 0-100 scores, so 50 is the median of the benchmark (or of the run without one);
    benchmark companies outside the run form a grey cloud behind it. Companies whose figures may be wrong
    (`tiles.unreliable`) are left out of the dots, the cloud and the correlation (user's choice, Nov 4, 2026: a
    misfiled cash flow put them at Value 99, top of the map). Only ranked companies (`ranking.rankable`: profitable
    operating companies, user's choice, Nov 5, 2026) are drawn, in the run and in the cloud.
    """
    d = map_points(df).copy()
    if d.empty:
        return None

    n = len(d)
    sp500 = _sp500_tickers()
    d["x"] = d["Quality_Score"]
    d["y"] = d["Value_Score"]
    d["is_sp500"] = d["Ticker"].isin(sp500)
    d["is_us"] = (d["country"] == "United States") | d["is_sp500"]
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
                customdata=[[a, int(b)] for a, b in zip(seg["slug"], seg["is_us"], strict=True)],
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

    corr = map_correlation(df)  # the dots drawn, no others (Nov 7, 2026)
    if n >= 3:
        slope, intercept = np.polyfit(d["x"], d["y"], 1)
        xs = np.array([d["x"].min(), d["x"].max()])
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
        annotations=[dict(x=top * 0.98, y=top * 0.05, text=f"Spearman ρ = {fmt(corr, '+.2f')} (n={n:,})", showarrow=False,
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
WATCHLIST_COLOR = "#10b981"  # emerald for watchlist overlays


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
    watched: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """[(title, html div)] for the distribution cards; the benchmark (grey) is overlaid when given;
    holdings (orange) and watchlist (emerald) can be toggled via their switches.
    """
    ref = benchmark(df)
    held_df = df[df["Ticker"].isin(held)] if held and "Ticker" in df.columns else pd.DataFrame()
    has_held = not held_df.empty and len(held_df) < len(df)
    watched_df = df[df["Ticker"].isin(watched)] if watched and "Ticker" in df.columns else pd.DataFrame()
    has_watched = not watched_df.empty and len(watched_df) < len(df)
    out = []
    for i, (col, title, xrange, size, *extra) in enumerate(HISTOGRAMS):
        if col not in df.columns or df[col].dropna().empty:
            continue
        scale, guide = (*extra, 1, None)[:2]

        def values(frame: pd.DataFrame | None, use: bool = True, col=col, scale=scale) -> pd.Series:
            if not use or frame is None or col not in frame.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(frame[col], errors="coerce").dropna() * scale

        data, ref_data = values(df), values(bench)
        held_data, watched_data = values(held_df, has_held), values(watched_df, has_watched)
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
        for overlay_name, overlay_data, color in (
            ("Holdings", held_data, HOLDINGS_COLOR),
            ("Watchlist", watched_data, WATCHLIST_COLOR),
        ):
            if not overlay_data.empty:
                fig.add_trace(go.Histogram(x=_plain(overlay_data[overlay_data.between(lo, hi)]), xbins=bins,
                                           name=overlay_name, marker_color=color, histnorm="percent", opacity=0.75,
                                           visible=False))  # fmt: skip
                med = overlay_data.median()
                fig.add_vline(x=med, line_width=2, line_dash="dashdot", line_color=color,
                              annotation_text=f"{overlay_name}: {med:.2f}", annotation_position="bottom right",
                              visible=False, annotation=dict(visible=False))  # fmt: skip
        fig.update_layout(title=_counted(title, data, ref_data, ref[0] if ref else "Benchmark", xrange), xaxis_title=title,
                          yaxis_title="% of companies" if not ref_data.empty else "Count", barmode="overlay",
                          height=350, showlegend=not ref_data.empty, margin=dict(l=20, r=20, t=70, b=20),
                          legend=dict(orientation="h", y=-0.25))  # fmt: skip
        fig.update_xaxes(range=list(xrange))
        out.append((title, _div(fig, f"hist_{i}")))
    cap = market_cap_figure(df, bench, held=held, watched=watched)
    if cap is not None:
        out.append((CAP_TITLE, _div(cap, "hist_cap")))
    return out


CAP_TITLE = "Market cap (USD, by size class)"
RUN_COLOR = "#636efa"  # the colour the other histograms get from the chart template
# Size classes by market cap in B$ (user's choice, Nov 8, 2026; replaced a log scale): the usual US bands
# (micro under 300 M$, small to 2 B$, mid to 10 B$, large to 200 B$), with nano below and the trillion club above.
CAP_CLASSES: tuple[tuple[str, float, float], ...] = (
    ("Nano", 0.0, 0.05),
    ("Micro", 0.05, 0.3),
    ("Small", 0.3, 2.0),
    ("Mid", 2.0, 10.0),
    ("Large", 10.0, 200.0),
    ("Mega", 200.0, 1000.0),
    ("Trillion+", 1000.0, np.inf),
)


def _money(value: float) -> str:
    """A market cap in B$ with sensible digits: 0.35, 4.2, 38, 1,250."""
    return f"{value:,.2f}" if value < 1 else f"{value:,.1f}" if value < 10 else f"{value:,.0f}"


def _band(lo: float, hi: float) -> str:
    """A size class's range in M$ / B$ / T$: "50-300 M$", "300 M$-2 B$", "over 1 T$"."""

    def amount(b: float) -> tuple[str, str]:
        return (f"{b * 1000:,.0f}", "M") if b < 1 else (f"{b:,.0f}", "B") if b < 1000 else (f"{b / 1000:,.0f}", "T")

    if lo <= 0:
        return "under {} {}$".format(*amount(hi))
    if np.isinf(hi):
        return "over {} {}$".format(*amount(lo))
    (a, ua), (b, ub) = amount(lo), amount(hi)
    return f"{a}-{b} {ub}$" if ua == ub else f"{a} {ua}$-{b} {ub}$"


def market_cap_class(caps: pd.Series) -> pd.Series:
    """The index of each market cap's size class (`CAP_CLASSES`; a cap equal to a class's lower bound is in it)."""
    edges = [lo for _, lo, _ in CAP_CLASSES[1:]]
    values = pd.to_numeric(caps, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.searchsorted(edges, values, side="right"), index=caps.index)


def _class_position(value: float) -> float:
    """Where a market cap sits on the class axis: its class's index, plus its log position within the class (open
    ends read as 10 M$ and 10 T$), so a median line falls inside its bar."""
    i = int(market_cap_class(pd.Series([value])).iloc[0])
    _, lo, hi = CAP_CLASSES[i]
    lo, hi = max(lo, 0.01), min(hi, 10_000.0)
    frac = (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return i - 0.5 + float(np.clip(frac, 0.0, 1.0))


def market_cap_figure(
    df: pd.DataFrame,
    bench: pd.DataFrame | None = None,
    held: frozenset[str] = frozenset(),
    watched: frozenset[str] = frozenset(),
) -> go.Figure | None:
    """How the companies spread on market cap: one bar per size class (`CAP_CLASSES`, nano to trillion+), each
    labelled with its USD range. The run in colour; the benchmark in grey (as % of its companies) when given; the
    medians as lines placed within their class. Hover gives each class's range and count. Portfolio holdings
    (orange) and watchlist (emerald) can be toggled via their switches."""
    if "Market Cap" not in df.columns:
        return None
    caps = pd.to_numeric(df["Market Cap"], errors="coerce")
    caps = caps[caps > 0]
    if caps.empty:
        return None
    ref = pd.to_numeric(bench["Market Cap"], errors="coerce") if bench is not None and "Market Cap" in bench else None
    ref = ref[ref > 0] if ref is not None else None
    ref = ref if ref is not None and not ref.empty else None

    def _subset_caps(tickers: frozenset[str]) -> tuple[pd.Series, bool]:
        sub = df[df["Ticker"].isin(tickers)] if tickers and "Ticker" in df.columns else pd.DataFrame()
        sub_caps = (
            pd.to_numeric(sub["Market Cap"], errors="coerce")
            if not sub.empty and "Market Cap" in sub
            else pd.Series(dtype=float)
        )
        sub_caps = sub_caps[sub_caps > 0]
        return sub_caps, not sub_caps.empty and len(sub_caps) < len(caps)

    held_caps, has_held = _subset_caps(held)
    watched_caps, has_watched = _subset_caps(watched)

    positions = list(range(len(CAP_CLASSES)))
    bands = [_band(lo, hi) for _, lo, hi in CAP_CLASSES]

    def bars(
        values: pd.Series, name: str, color: str, percent: bool, opacity: float, visible: bool | None = None
    ) -> go.Bar:
        counts = market_cap_class(values).value_counts().reindex(positions, fill_value=0).to_numpy()
        heights = counts / len(values) * 100 if percent else counts
        label = "holdings" if name == "Holdings" else "watchlist" if name == "Watchlist" else "companies"
        text = [f"{cls} cap, {band}<br>{n:,} {label} ({n / len(values):.0%})"
                for (cls, _, _), band, n in zip(CAP_CLASSES, bands, counts, strict=True)]  # fmt: skip
        pct_heights = counts / len(values) * 100
        customdata = np.column_stack([counts, pct_heights])
        kwargs = dict(
            x=positions,
            y=heights,
            width=0.8,
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
        fig.add_vline(x=_class_position(ref.median()), line_width=1.5, line_dash="dot", line_color="grey",
                      annotation_text=f"{named[0] if named else 'Benchmark'}: {_money(ref.median())} B$",
                      annotation_position="top left")  # fmt: skip
    fig.add_trace(bars(caps, "This run", RUN_COLOR, True, 0.75))
    fig.add_vline(x=_class_position(caps.median()), line_width=2, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {_money(caps.median())} B$", annotation_position="top right")  # fmt: skip
    for show, overlay_name, overlay_caps, color in (
        (has_held, "Holdings", held_caps, HOLDINGS_COLOR),
        (has_watched, "Watchlist", watched_caps, WATCHLIST_COLOR),
    ):
        if show:
            fig.add_trace(bars(overlay_caps, overlay_name, color, True, 0.75, visible=False))
            fig.add_vline(x=_class_position(overlay_caps.median()), line_width=2, line_dash="dashdot",
                          line_color=color, annotation_text=f"{overlay_name}: {_money(overlay_caps.median())} B$",
                          annotation_position="bottom right", visible=False, annotation=dict(visible=False))  # fmt: skip
    ticktext = [f"{cls}<br>{band}" for (cls, _, _), band in zip(CAP_CLASSES, bands, strict=True)]
    fig.update_layout(title=_counted(CAP_TITLE, caps, ref, named[0] if named else "Benchmark"),
                      xaxis_title="Size class (market cap in USD)",
                      yaxis_title="% of companies", barmode="overlay", bargap=0.05,
                      height=370, showlegend=ref is not None, margin=dict(l=20, r=20, t=70, b=20),
                      legend=dict(orientation="h", y=-0.35))  # fmt: skip
    fig.update_xaxes(tickvals=positions, ticktext=ticktext, range=[-0.5, len(CAP_CLASSES) - 0.5])
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
