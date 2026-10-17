"""Shared palette and SVG rules for HTML charts and standalone vector exports."""

THEME_STYLE = """
:root {
  --bg: #0a0d10;
  --panel: #10151a;
  --panel-2: #0d1216;
  --line: #1f2830;
  --line-strong: #2c3944;
  --text: #d9e2e8;
  --text-muted: #8b97a1;
  --text-dim: #6e7c87;
  --err: #ff5d5d;
  --warn: #f2b84b;
  --info: #5fb3ff;
  --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
"""

SVG_STYLE = """
.grid { stroke: var(--line); stroke-width: 1; }
.frame { stroke: var(--line-strong); stroke-width: 1; }
.axis-y, .axis-x { font-family: var(--mono); font-size: 9px; fill: var(--text-dim); }
.legend { font-family: var(--mono); font-size: 10px; fill: var(--text-muted); }
.marker-bad { font-family: var(--mono); font-size: 13px; fill: var(--err); font-weight: 700; }
.off-scale-label { font-family: var(--mono); font-size: 9px; font-weight: 600; }

"""
