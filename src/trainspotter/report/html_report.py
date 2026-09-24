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
_DEFAULT_CHART_ORDER = ["train/loss", "eval/loss", "lr", "grad_norm"]

_W = 920
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 54, 18, 14, 26
_PLOT_H = 148
_CHART_H = _PLOT_H + _PAD_T + _PAD_B


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1000 or av < 1e-3:
        return f"{v:.3g}"
    return f"{v:.4g}"


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def _plot_x(step: float, step_min: float, step_max: float) -> float:
    span = max(step_max - step_min, 1e-9)
    return _PAD_L + (step - step_min) / span * (_W - _PAD_L - _PAD_R)


def _plot_y(value: float, v_min: float, v_max: float) -> float:
    span = max(v_max - v_min, 1e-9)
    return _PAD_T + (1.0 - (value - v_min) / span) * _PLOT_H


def _render_chart(name: str, series: MetricSeries, findings: list[Finding], color: str) -> str:
    steps = series.steps()
    values = series.values()
    step_min, step_max = min(steps), max(steps)
    finite = [(s, v) for s, v in zip(steps, values, strict=True) if math.isfinite(v)]
    non_finite = [(s, v) for s, v in zip(steps, values, strict=True) if not math.isfinite(v)]

    parts: list[str] = []
    parts.append(
        f'<svg class="chart" viewBox="0 0 {_W} {_CHART_H}" role="img" '
        f'aria-label="{_esc(name)} over steps {int(step_min)} to {int(step_max)}">'
    )

    if finite:
        v_vals = [v for _, v in finite]
        v_lo, v_hi = min(v_vals), max(v_vals)
        pad = (v_hi - v_lo) * 0.08 or (abs(v_hi) * 0.08 or 1.0)
        v_min, v_max = v_lo - pad, v_hi + pad
    else:
        v_min, v_max = 0.0, 1.0

    # gridlines + axis labels
    for gy in _nice_ticks(v_min, v_max, 3):
        y = _plot_y(gy, v_min, v_max)
        parts.append(f'<line class="grid" x1="{_PAD_L}" y1="{y:.1f}" x2="{_W - _PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis-y" x="{_PAD_L - 8}" y="{y + 3:.1f}" text-anchor="end">{_esc(_fmt_num(gy))}</text>')
    for gx in _nice_ticks(step_min, step_max, 4):
        x = _plot_x(gx, step_min, step_max)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{_PAD_T}" x2="{x:.1f}" y2="{_PAD_T + _PLOT_H}"/>')
        parts.append(f'<text class="axis-x" x="{x:.1f}" y="{_PAD_T + _PLOT_H + 16}" text-anchor="middle">{int(gx)}</text>')

    # finding shading, drawn under the line
    for f in findings:
        if f.metric != name:
            continue
        x0 = _plot_x(f.step_start, step_min, step_max)
        x1 = max(_plot_x(f.step_end, step_min, step_max), x0 + 3)
        color_f = _SEVERITY_COLOR[f.severity]
        parts.append(
            f'<rect class="finding-band" x="{x0:.1f}" y="{_PAD_T}" width="{x1 - x0:.1f}" '
            f'height="{_PLOT_H}" fill="{color_f}" fill-opacity="0.16"/>'
        )
        parts.append(f'<line x1="{x0:.1f}" y1="{_PAD_T}" x2="{x0:.1f}" y2="{_PAD_T + _PLOT_H}" stroke="{color_f}" stroke-opacity="0.55" stroke-width="1"/>')

    # the line itself, as a sequence of moveto/lineto segments broken at non-finite points
    segment: list[str] = []
    path_cmds: list[str] = []
    for s, v in zip(steps, values, strict=True):
        if not math.isfinite(v):
            if segment:
                path_cmds.append("M" + " L".join(segment))
                segment = []
            continue
        x, y = _plot_x(s, step_min, step_max), _plot_y(v, v_min, v_max)
        segment.append(f"{x:.1f} {y:.1f}")
    if segment:
        path_cmds.append("M" + " L".join(segment))
    for cmd in path_cmds:
        parts.append(f'<path d="{cmd}" fill="none" stroke="{color}" stroke-width="1.75"/>')

    # non-finite markers
    for s, _v in non_finite:
        x = _plot_x(s, step_min, step_max)
        y = _PAD_T + 10
        parts.append(f'<text class="marker-bad" x="{x:.1f}" y="{y:.1f}" text-anchor="middle">&#10005;</text>')

    parts.append(f'<rect class="frame" x="{_PAD_L}" y="{_PAD_T}" width="{_W - _PAD_L - _PAD_R}" height="{_PLOT_H}" fill="none"/>')
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
    # error and warning first (drawn under), info on top, so nothing severe gets hidden
    for sev in ("info", "warning", "error"):
        for f in findings:
            if f.severity != sev:
                continue
            x0 = _plot_x(f.step_start, step_min, step_max)
            x1 = max(_plot_x(f.step_end, step_min, step_max), x0 + 2)
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


def render_html(run: Run, findings: list[Finding], title: str = "trainspotter report") -> str:
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] += 1
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    chart_names = [n for n in _DEFAULT_CHART_ORDER if n in run.metrics]
    chart_names += [n for n in run.metric_names() if n not in chart_names]
    charts_html = []
    for name in chart_names:
        series = run.metrics[name]
        if series.is_empty():
            continue
        color = _CHART_COLORS.get(name, "#9aa7b0")
        charts_html.append(
            f'<section class="chart-block"><h2>{_esc(name)}</h2>'
            f'{_render_chart(name, series, findings, color)}</section>'
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
      <div><dt>source</dt><dd>{_esc(run.source_path or "(stdin)")}</dd></div>
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
.marker-bad { font-family: var(--mono); font-size: 13px; fill: var(--err); font-weight: 700; }

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
