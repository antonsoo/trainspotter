"""Self-contained HTML report: annotated SVG charts + a findings log.

Design: read as a lab instrument, not a dashboard. Every number is set in
a monospace numeral face like an oscilloscope readout; a "health strip" at
the top gives a single-glance timeline of every finding across the whole
run before you read a word; each chart below it shades the exact step
range a finding covers, in the same severity colors as the strip, so you
can trace a shaded band on the strip straight down to the curve that
caused it. No JavaScript, no external requests, no charting library --
everything is hand-built SVG so the file works by itself, forever, opened
from disk with no network.

Chart ordering follows what you'd actually look at first: the loss chart
(train and eval overlaid on the same axes, since their divergence *is* the
overfitting signal) leads, then LR, then grad_norm, then any other eval
metric, then step_time last. `epoch` is skipped -- it's a linear
reparameterization of step, never diagnostic on its own.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from trainspotter.detectors import Finding
from trainspotter.model import MetricSeries, Run

_SEVERITY_COLOR = {"error": "#ff5d5d", "warning": "#f2b84b", "info": "#5fb3ff"}
_SEVERITY_LABEL = {"error": "ERR", "warning": "WARN", "info": "INFO"}

_CHART_COLORS = {
    "train/loss": "#4fd1e8",
    "eval/loss": "#ffb454",
    "lr": "#b39dff",
    "grad_norm": "#7be08a",
}
_FALLBACK_COLORS = ["#9aa7b0", "#6fd6c4", "#e08fd6"]

# Charts skipped entirely (not diagnostic on their own) and metrics forced
# to the end of the chart list (useful, but secondary to loss/lr/grad_norm).
_SKIP_METRICS = {"epoch"}
_TRAILING_METRICS = ["step_time", "throughput"]
_LOSS_METRICS = ["train/loss", "eval/loss"]
_LEAD_METRICS = ["lr", "grad_norm"]

_W = 920
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 54, 18, 14, 26
_PLOT_H = 140
_CHART_H = _PLOT_H + _PAD_T + _PAD_B
_LEGEND_H = 18  # extra header row reserved when a chart overlays >1 series


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _esc_path(s: str) -> str:
    """Escape, then mark each path separator as a wrap point. A long
    source path has to wrap somewhere in a narrow readout column; plain
    `overflow-wrap: anywhere` will happily split *inside* a file extension
    (".jso" / "n"), which is what it did before this existed. `<wbr>` gives
    the browser a preferred break point at each "/" instead."""
    return _esc(s).replace("/", "/<wbr>")


def _fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1000 or av < 1e-3:
        return f"{v:.3g}"
    return f"{v:.4g}"


def _nice_num(x: float, round_to_nice: bool) -> float:
    """Paul Heckbert's "nice numbers" step: the classic axis-tick algorithm
    (see `Graphics Gems`, 1990) that picks 1/2/5x10^n rather than whatever
    an even division of the range happens to produce -- the difference
    between a "0.02, 0.04, 0.06" axis and a "0.0213, 0.0426..." one."""
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    frac = x / 10**exp
    if round_to_nice:
        nice_frac = 1.0 if frac < 1.5 else 2.0 if frac < 3.0 else 5.0 if frac < 7.0 else 10.0
    else:
        nice_frac = 1.0 if frac <= 1.0 else 2.0 if frac <= 2.0 else 5.0 if frac <= 5.0 else 10.0
    return float(nice_frac * 10**exp)


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Tick positions covering [lo, hi] at a Heckbert "nice" step, snapped
    to multiples of that step -- so 0 is exactly a tick whenever lo <= 0
    <= hi, instead of landing at an arbitrary fraction like -8.953."""
    if hi <= lo:
        return [lo]
    raw_step = _nice_num((hi - lo) / max(n, 1), round_to_nice=True)
    nice_lo = math.floor(lo / raw_step) * raw_step
    nice_hi = math.ceil(hi / raw_step) * raw_step
    ticks = []
    v = nice_lo
    # cap iterations defensively -- a pathological step can't spin forever
    for _ in range(n + 4):
        if v > nice_hi + raw_step * 0.5:
            break
        ticks.append(round(v, 12))
        v += raw_step
    return ticks or [lo, hi]


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * pct
    lo_i, hi_i = math.floor(idx), math.ceil(idx)
    if lo_i == hi_i:
        return sorted_values[lo_i]
    frac = idx - lo_i
    return sorted_values[lo_i] * (1 - frac) + sorted_values[hi_i] * frac


def _y_domain(finite_values: list[float]) -> tuple[float, float, float | None]:
    """The chart's y-axis bounds, and an optional cap.

    Two decisions that keep a report readable instead of misleading:
    - **Non-negative floor.** If every finite value is >= 0 (true of every
      metric trainspotter charts -- loss, lr, grad_norm, accuracy,
      step_time), the axis starts at exactly 0 rather than at
      `min - padding`, which used to produce a nonsensical negative tick
      like "-8.953" on a loss chart.
    - **Robust cap.** If the maximum is more than 1.5x the 99th
      percentile, one spike is stretching the whole axis and flattening
      everything else into a nearly-flat line at the bottom. The domain is
      capped at `p99 * 1.5`; points above the cap are still drawn (clamped
      to the top edge, with an explicit off-scale marker showing the real
      value -- see `_render_chart`) instead of silently hidden.
    """
    if not finite_values:
        return 0.0, 1.0, None
    values = sorted(finite_values)
    lo, hi = values[0], values[-1]
    non_negative = lo >= -1e-9
    p99 = _percentile(values, 0.99)
    cap: float | None = None
    display_hi = hi
    if p99 > 0 and hi > p99 * 1.5:
        cap = p99 * 1.5
        display_hi = cap
    display_lo = 0.0 if non_negative else lo
    span = display_hi - display_lo
    pad = span * 0.08 or abs(display_hi) * 0.08 or 1.0
    return display_lo, display_hi + pad, cap


def _plot_x(step: float, step_min: float, step_max: float) -> float:
    span = max(step_max - step_min, 1e-9)
    return _PAD_L + (step - step_min) / span * (_W - _PAD_L - _PAD_R)


def _plot_y(value: float, v_min: float, v_max: float, plot_top: float) -> float:
    span = max(v_max - v_min, 1e-9)
    return plot_top + (1.0 - (value - v_min) / span) * _PLOT_H


def _render_off_scale_labels(
    candidates: list[tuple[float, str, str]], plot_top: float
) -> str:
    """Off-scale value labels, one line of text per x-cluster.

    Two series can each have a capped point a step or two apart -- a real
    case on the divergence example, where eval/loss and train/loss both
    blow up within one step of each other. That's only a couple of pixels
    apart, so even stacking each label on its own row (an earlier version
    of this function) still read as visual noise crammed into one corner.
    Values within `min_gap` px of each other are merged into a single
    "54.88 / 55.49"-style label instead, each value kept in its own
    series' color via a `<tspan>`, so there's exactly one thing to read at
    each cluster, not a pile of near-overlapping ones."""
    min_gap = 26.0
    clusters: list[list[tuple[float, str, str]]] = []
    for cand in sorted(candidates, key=lambda c: c[0]):
        if clusters and cand[0] - clusters[-1][-1][0] < min_gap:
            clusters[-1].append(cand)
        else:
            clusters.append([cand])

    char_w = 5.5  # rough advance width of the 9px mono label font
    parts: list[str] = []
    for raw_cluster in clusters:
        # A metric pinned at one value for several consecutive off-scale
        # points (e.g. lr held at a constant post-incident value) doesn't
        # need that value repeated once per point.
        cluster = [
            c
            for i, c in enumerate(raw_cluster)
            if i == 0 or (c[1], c[2]) != (raw_cluster[i - 1][1], raw_cluster[i - 1][2])
        ]
        cx = sum(x for x, _t, _c in raw_cluster) / len(raw_cluster)
        y = plot_top + 20
        label_len = sum(len(t) for _x, t, _c in cluster) + 3 * (len(cluster) - 1)
        half_w = label_len * char_w / 2
        # Keep the label inside the plot area instead of letting a
        # centered label clip past the right edge for a cluster near the
        # end of the run -- exactly where off-scale points tend to land.
        if cx + half_w > _W - _PAD_R:
            anchor, tx = "end", float(_W - _PAD_R)
        elif cx - half_w < _PAD_L:
            anchor, tx = "start", float(_PAD_L)
        else:
            anchor, tx = "middle", cx
        tspans = []
        for i, (_x, text, color) in enumerate(cluster):
            if i > 0:
                tspans.append('<tspan fill="var(--text-dim)"> / </tspan>')
            tspans.append(f'<tspan fill="{color}">{_esc(text)}</tspan>')
        parts.append(f'<text class="off-scale-label" x="{tx:.1f}" y="{y:.1f}" text-anchor="{anchor}">{"".join(tspans)}</text>')
    return "".join(parts)


def _render_chart(
    entries: list[tuple[str, MetricSeries, str]], findings: list[Finding]
) -> str:
    """Render one chart overlaying 1+ metric series (`entries`) that share
    an x (step) and y (value) axis. A 2nd+ series is what makes the
    combined train/eval loss chart possible: same function, just called
    with two entries instead of one."""
    metric_names = [name for name, _series, _color in entries]
    all_steps = [s for _n, series, _c in entries for s in series.steps()]
    if not all_steps:
        return ""
    step_min, step_max = min(all_steps), max(all_steps)

    finite_values = [
        v for _n, series, _c in entries for v in series.values() if math.isfinite(v)
    ]
    v_min, v_max, cap = _y_domain(finite_values)

    has_legend = len(entries) > 1
    plot_top = _PAD_T + (_LEGEND_H if has_legend else 0)
    chart_h = _CHART_H + (_LEGEND_H if has_legend else 0)

    parts: list[str] = []
    label = " vs ".join(metric_names)
    parts.append(
        f'<svg class="chart" viewBox="0 0 {_W} {chart_h}" role="img" '
        f'aria-label="{_esc(label)} over steps {int(step_min)} to {int(step_max)}">'
    )

    if has_legend:
        lx = float(_PAD_L)
        for name, _series, color in entries:
            parts.append(f'<rect x="{lx:.1f}" y="0" width="10" height="10" fill="{color}"/>')
            parts.append(f'<text class="legend" x="{lx + 14:.1f}" y="9" text-anchor="start">{_esc(name)}</text>')
            lx += 16 + 7.2 * len(name) + 18

    # gridlines + axis labels
    for gy in _nice_ticks(v_min, v_max, 3):
        y = _plot_y(gy, v_min, v_max, plot_top)
        parts.append(f'<line class="grid" x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis-y" x="{_PAD_L - 8}" y="{y + 3:.1f}" text-anchor="end">{_esc(_fmt_num(gy))}</text>')
    for gx in _nice_ticks(step_min, step_max, 4):
        x = _plot_x(gx, step_min, step_max)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_top + _PLOT_H}"/>')
        parts.append(f'<text class="axis-x" x="{x:.1f}" y="{plot_top + _PLOT_H + 16}" text-anchor="middle">{int(gx)}</text>')

    # finding shading, drawn under the lines
    for f in findings:
        if f.metric not in metric_names:
            continue
        x0 = _plot_x(f.step_start, step_min, step_max)
        x1 = max(_plot_x(f.step_end, step_min, step_max), x0 + 3)
        color_f = _SEVERITY_COLOR[f.severity]
        parts.append(
            f'<rect class="finding-band" x="{x0:.1f}" y="{plot_top}" width="{x1 - x0:.1f}" '
            f'height="{_PLOT_H}" fill="{color_f}" fill-opacity="0.16"/>'
        )
        parts.append(f'<line x1="{x0:.1f}" y1="{plot_top}" x2="{x0:.1f}" y2="{plot_top + _PLOT_H}" stroke="{color_f}" stroke-opacity="0.55" stroke-width="1"/>')

    # Off-scale labels are collected across *all* series and placed together
    # at the end: two series can each have a capped point a step or two
    # apart (a real case on the divergence example -- eval/loss and
    # train/loss both blow up within one step of each other), which is
    # close enough in pixels that drawing each label independently produced
    # two strings of digits mashed on top of each other.
    label_candidates: list[tuple[float, str, str]] = []  # (x, text, color)

    for _name, series, color in entries:
        steps, values = series.steps(), series.values()
        # the line, as moveto/lineto segments broken at non-finite points;
        # a value above `cap` is clamped to the top edge, not hidden.
        segment: list[str] = []
        path_cmds: list[str] = []
        off_scale: list[tuple[int, float]] = []
        for s, v in zip(steps, values, strict=True):
            if not math.isfinite(v):
                if segment:
                    path_cmds.append("M" + " L".join(segment))
                    segment = []
                continue
            plotted = min(v, cap) if cap is not None else v
            if cap is not None and v > cap:
                off_scale.append((s, v))
            x, y = _plot_x(s, step_min, step_max), _plot_y(plotted, v_min, v_max, plot_top)
            segment.append(f"{x:.1f} {y:.1f}")
        if segment:
            path_cmds.append("M" + " L".join(segment))
        for cmd in path_cmds:
            parts.append(f'<path d="{cmd}" fill="none" stroke="{color}" stroke-width="1.75"/>')

        # off-scale markers: a small triangle at the clamped top, labeled
        # with the real value, so a capped axis never hides the number.
        for s, v in off_scale:
            x = _plot_x(s, step_min, step_max)
            y = plot_top + 3
            parts.append(f'<path class="marker-offscale" d="M{x - 4:.1f} {y + 7:.1f} L{x:.1f} {y:.1f} L{x + 4:.1f} {y + 7:.1f} Z" fill="{color}"/>')
            label_candidates.append((x, _fmt_num(v), color))

        # non-finite (NaN/Inf) markers
        for s, v in zip(steps, values, strict=True):
            if math.isfinite(v):
                continue
            x = _plot_x(s, step_min, step_max)
            y = plot_top + 10
            parts.append(f'<text class="marker-bad" x="{x:.1f}" y="{y:.1f}" text-anchor="middle">&#10005;</text>')

    parts.append(_render_off_scale_labels(label_candidates, plot_top))

    parts.append(f'<rect class="frame" x="{_PAD_L}" y="{plot_top}" width="{_W - _PAD_L - _PAD_R}" height="{_PLOT_H}" fill="none"/>')
    parts.append("</svg>")
    return "".join(parts)


def _render_health_strip(run: Run, findings: list[Finding]) -> str:
    all_steps = [s for series in run.metrics.values() for s in series.steps()]
    if not all_steps:
        return ""
    step_min, step_max = min(all_steps), max(all_steps)
    h = 34
    parts = [f'<svg class="strip" viewBox="0 0 {_W} {h}" role="img" aria-label="Finding timeline for the whole run">']
    parts.append(f'<rect x="0" y="0" width="{_W}" height="{h}" class="strip-bg"/>')
    parts.append(f'<line x1="0" y1="{h - 1}" x2="{_W}" y2="{h - 1}" class="strip-base"/>')
    # info and warning drawn first (underneath), error drawn last (on top),
    # so a severe finding is never hidden behind a milder overlapping one.
    for sev in ("info", "warning", "error"):
        for f in findings:
            if f.severity != sev:
                continue
            x0 = _plot_x(f.step_start, step_min, step_max)
            # A single-step finding needs a floor wider than 1-2px or it
            # vanishes into anti-aliasing; 4px keeps it visible at a glance.
            x1 = max(_plot_x(f.step_end, step_min, step_max), x0 + 4)
            color = _SEVERITY_COLOR[sev]
            parts.append(f'<rect x="{x0:.1f}" y="2" width="{x1 - x0:.1f}" height="{h - 6}" fill="{color}" fill-opacity="0.85"/>')
    parts.append("</svg>")
    return "".join(parts)


def _findings_rows(findings: list[Finding]) -> str:
    if not findings:
        return '<p class="clean">No pathologies detected in the metrics this detector set covers.</p>'
    rows = []
    for f in findings:
        step_range = str(f.step_start) if f.step_start == f.step_end else f"{f.step_start}&ndash;{f.step_end}"
        evidence = " &nbsp; ".join(f"<span class='k'>{_esc(str(k))}</span>={_esc(_fmt_num(v) if isinstance(v, float) else str(v))}" for k, v in f.evidence.items())
        fixes = "".join(f"<li>{_esc(fix)}</li>" for fix in f.fixes)
        rows.append(
            f"""<article class="finding sev-{f.severity}">
  <div class="finding-head">
    <span class="chip sev-{f.severity}">{_SEVERITY_LABEL[f.severity]}</span>
    <span class="steps">steps {step_range}</span>
    <span class="detector">{_esc(f.detector)}</span>
    <h3>{_esc(f.title)}</h3>
  </div>
  <p class="message">{_esc(f.message)}</p>
  <p class="evidence">{evidence}</p>
  <ul class="fixes">{fixes}</ul>
</article>"""
        )
    return "\n".join(rows)


def _chart_plan(run: Run) -> list[list[str]]:
    """Which metrics to chart, grouped (a group of 2 = one overlaid chart),
    and in what order: loss (combined) -> lr -> grad_norm -> other eval
    metrics -> step_time/throughput last. See the module docstring."""
    available = set(run.metric_names()) - _SKIP_METRICS
    plan: list[list[str]] = []

    loss_group = [m for m in _LOSS_METRICS if m in available]
    if loss_group:
        plan.append(loss_group)
        available -= set(loss_group)

    for m in _LEAD_METRICS:
        if m in available:
            plan.append([m])
            available.discard(m)

    trailing = [m for m in _TRAILING_METRICS if m in available]
    available -= set(trailing)

    plan.extend([m] for m in sorted(available))
    plan.extend([m] for m in trailing)
    return plan


def render_html(run: Run, findings: list[Finding], title: str = "trainspotter report") -> str:
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] += 1
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    charts_html = []
    fallback_i = 0
    for group in _chart_plan(run):
        entries: list[tuple[str, MetricSeries, str]] = []
        for name in group:
            series = run.metrics.get(name)
            if series is None or series.is_empty():
                continue
            color = _CHART_COLORS.get(name)
            if color is None:
                color = _FALLBACK_COLORS[fallback_i % len(_FALLBACK_COLORS)]
                fallback_i += 1
            entries.append((name, series, color))
        if not entries:
            continue
        heading = " / ".join(name for name, _s, _c in entries)
        charts_html.append(
            f'<section class="chart-block"><h2>{_esc(heading)}</h2>'
            f"{_render_chart(entries, findings)}</section>"
        )

    total_steps = max((s for series in run.metrics.values() for s in series.steps()), default=0)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="bezel">
  <header class="masthead">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true">&#9679;</span>
      <span class="brand-name">TRAINSPOTTER</span>
      <span class="brand-sub">run diagnostic</span>
    </div>
    <dl class="readouts">
      <div><dt>source</dt><dd>{_esc_path(run.source_path) if run.source_path else "(stdin)"}</dd></div>
      <div><dt>format</dt><dd>{_esc(run.source_format)}</dd></div>
      <div><dt>steps</dt><dd>0&ndash;{total_steps}</dd></div>
      <div><dt>generated</dt><dd>{generated}</dd></div>
    </dl>
  </header>

  <section class="summary">
    <div class="count sev-error"><span class="n">{counts['error']}</span><span class="l">error</span></div>
    <div class="count sev-warning"><span class="n">{counts['warning']}</span><span class="l">warning</span></div>
    <div class="count sev-info"><span class="n">{counts['info']}</span><span class="l">info</span></div>
    <div class="strip-wrap">{_render_health_strip(run, findings)}</div>
  </section>

  <main class="charts">
    {"".join(charts_html)}
  </main>

  <section class="findings-log">
    <h2 class="log-title">Findings log</h2>
    {_findings_rows(findings)}
  </section>

  <footer class="masthead-foot">
    <span>trainspotter &mdash; heuristic diagnostics, not ground truth. Read each finding's false-positive modes before acting on it.</span>
  </footer>
</div>
</body>
</html>
"""


_CSS = """
:root {
  --bg: #0a0d10;
  --panel: #10151a;
  --panel-2: #0d1216;
  --line: #1f2830;
  --line-strong: #2c3944;
  --text: #d9e2e8;
  --text-muted: #8b97a1;
  --text-dim: #57626b;
  --err: #ff5d5d;
  --warn: #f2b84b;
  --info: #5fb3ff;
  --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); }
body {
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.5;
  padding: 0 0 48px;
}
.bezel { max-width: 980px; margin: 0 auto; padding: 28px 20px 0; }

.masthead {
  display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
  gap: 12px 24px; padding-bottom: 16px; border-bottom: 1px solid var(--line-strong);
}
.brand { display: flex; align-items: baseline; gap: 10px; }
.brand-mark { color: var(--info); font-size: 11px; }
.brand-name { font-family: var(--mono); font-weight: 700; font-size: 17px; letter-spacing: 0.12em; }
.brand-sub { font-family: var(--mono); color: var(--text-dim); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }
.readouts { display: flex; flex-wrap: wrap; gap: 18px 28px; margin: 0; font-family: var(--mono); font-size: 12px; }
.readouts > div { display: flex; flex-direction: column; gap: 2px; max-width: 260px; }
.readouts dt { color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; font-size: 10px; }
.readouts dd { margin: 0; color: var(--text-muted); overflow-wrap: anywhere; }

.summary { padding: 18px 0 8px; }
.summary { display: grid; grid-template-columns: repeat(3, auto) 1fr; align-items: center; gap: 20px; }
.count { display: flex; align-items: baseline; gap: 8px; font-family: var(--mono); }
.count .n { font-size: 22px; font-weight: 700; }
.count .l { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); }
.count.sev-error .n { color: var(--err); }
.count.sev-warning .n { color: var(--warn); }
.count.sev-info .n { color: var(--info); }
.strip-wrap { min-width: 220px; }
.strip { width: 100%; height: 34px; display: block; }
.strip-bg { fill: var(--panel-2); }
.strip-base { stroke: var(--line-strong); stroke-width: 1; }

.charts { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
.chart-block { background: var(--panel); border: 1px solid var(--line); border-radius: 2px; padding: 10px 14px 4px; }
.chart-block h2 {
  font-family: var(--mono); font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-muted); margin: 0 0 4px;
}
.chart { width: 100%; height: auto; display: block; }
.grid { stroke: var(--line); stroke-width: 1; }
.frame { stroke: var(--line-strong); stroke-width: 1; }
.axis-y, .axis-x { font-family: var(--mono); font-size: 9px; fill: var(--text-dim); }
.legend { font-family: var(--mono); font-size: 10px; fill: var(--text-muted); }
.marker-bad { font-family: var(--mono); font-size: 13px; fill: var(--err); font-weight: 700; }
.off-scale-label { font-family: var(--mono); font-size: 9px; font-weight: 600; }

.findings-log { margin-top: 28px; }
.log-title {
  font-family: var(--mono); font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--text-muted); border-bottom: 1px solid var(--line-strong); padding-bottom: 8px;
}
.clean { color: var(--text-muted); font-family: var(--mono); font-size: 13px; }
.finding { border-bottom: 1px solid var(--line); padding: 14px 0; }
.finding:last-child { border-bottom: none; }
.finding-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; }
.finding-head h3 { font-size: 14.5px; margin: 0; color: var(--text); font-weight: 600; }
.chip {
  font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.06em;
  padding: 2px 6px; border-radius: 2px; border: 1px solid currentColor;
}
.chip.sev-error { color: var(--err); }
.chip.sev-warning { color: var(--warn); }
.chip.sev-info { color: var(--info); }
.steps, .detector { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }
.detector::before { content: "\\00b7 "; }
.message { margin: 6px 0 4px; color: var(--text); }
.evidence { font-family: var(--mono); font-size: 11.5px; color: var(--text-muted); margin: 0 0 6px; }
.evidence .k { color: var(--text-dim); }
.fixes { margin: 0; padding-left: 18px; color: var(--text-muted); font-size: 13px; }
.fixes li { margin: 2px 0; }

.masthead-foot {
  margin-top: 32px; padding-top: 14px; border-top: 1px solid var(--line);
  color: var(--text-dim); font-family: var(--mono); font-size: 11px;
}

@media (max-width: 600px) {
  .summary { grid-template-columns: repeat(3, auto); }
  .strip-wrap { grid-column: 1 / -1; order: 10; }
}
"""
