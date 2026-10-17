"""A chart download with its own styling and the report's interpretation context."""

from __future__ import annotations

import base64
import html
import json
import re
import textwrap
import xml.etree.ElementTree as ET

from trainspotter.model import MetricSeries, Run
from trainspotter.report.chart_style import SVG_STYLE, THEME_STYLE

# Resolve only our own fixed styles. Data labels containing var(...) remain data.
_THEME_VALUES = dict(re.findall(r"--([\w-]+):\s*([^;]+);", THEME_STYLE))


def _literal_style(style: str) -> str:
    values = {**_THEME_VALUES, "mono": "monospace", "sans": "sans-serif"}
    return re.sub(r"var\(--([\w-]+)\)", lambda match: values[match[1]].strip(), style)


_EXPORT_STYLE = _literal_style(
    SVG_STYLE
    + 'svg { font-family: var(--sans); font-size: 14px; color: var(--text); }'
    # Static SVG 1.1 converters retain the explicit root width/height. Browsers
    # additionally fit the intrinsic ratio to a narrow screen viewport.
    + '@media screen { svg:root { max-width: 100%; height: auto; } }'
    + '.export-caption { font-family: var(--mono); font-size: 11px; fill: var(--text); }'
    + '.export-footer { font-family: var(--sans); font-size: 10px; fill: var(--text-muted); }'
)


def _xml_text(text: str) -> str:
    """XML 1.0 cannot carry every character that an HTML text node accepts."""
    return "".join(
        c if c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF
        or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF else "\ufffd"
        for c in text
    )


def _portable_chart(chart: str, width: int, height: int, top: int) -> str:
    """Copy geometry, resolving owned paint and explicitly positioning value labels.

    Some SVG 1.1 converters anchor each nested tspan separately. Explicit
    monospaced positions keep every colored value inside its original cluster.
    Text and identity attributes are never interpreted as theme expressions.
    """
    node = ET.fromstring(_xml_text(chart))
    node.attrib.update(x="0", y=str(top), width=str(width), height=str(height))
    for element in node.iter():
        for prop in ("fill", "stroke", "color"):
            if prop in element.attrib:
                element.set(prop, _literal_style(element.attrib[prop]))
        if element.get("class") != "off-scale-label":
            continue
        spans = list(element)
        length = sum(len(span.text or "") for span in spans)
        # The report's label placement uses the same 9px mono advance budget.
        advance = 5.5
        x = float(element.attrib["x"])
        anchor = element.get("text-anchor")
        if anchor == "end":
            x -= length * advance
        elif anchor == "middle":
            x -= length * advance / 2
        element.set("text-anchor", "start")
        for span in spans:
            span.set("x", f"{x:.2f}")
            span.set("text-anchor", "start")
            x += len(span.text or "") * advance
    return ET.tostring(node, encoding="unicode")


def chart_download(
    chart: str, run: Run, metrics: list[str],
    entries: list[tuple[str, MetricSeries, str]], index: int, *,
    input_complete: bool, input_notes: list[str],
) -> str:
    """Wrap the already drawn chart: trajectory geometry is shared, not recalculated."""
    bounds = re.search(r'viewBox="0 0 (\d+) (\d+)"', chart)
    if not bounds:
        raise ValueError("Cannot export a chart without its dimensions")
    width, chart_height = map(int, bounds.groups())
    heading = " / ".join(metrics)
    plotted = [name for name, _series, _color in entries]
    excluded = [name for name in metrics if name not in plotted]
    sources = run.meta.get("metric_sources", {})
    identity = run.meta.get("chart_identity", {})
    columns = identity.get("columns", {}) if isinstance(identity, dict) else {}
    labels = {}
    for name in metrics:
        source = str(sources.get(name, name)) if isinstance(sources, dict) else name
        labels[name] = str(columns.get(source, source)) if isinstance(columns, dict) else source
    selected = run.meta.get("selected_run")
    axis_source = run.meta.get("step_column")
    metadata = {
        "schemaVersion": 1, "format": "trainspotter-chart", "sourcePath": run.source_path,
        "sourceFormat": run.source_format,
        "selectedRun": selected if isinstance(selected, str) else None,
        "axisKind": run.axis_kind(), "axisLabel": run.axis_label(),
        "axisSource": axis_source if isinstance(axis_source, str) else None,
        "requestedMetrics": metrics, "plottedMetrics": plotted, "excludedMetrics": excluded,
        "metricSources": labels, "inputComplete": input_complete,
        "inputNotes": input_notes,
        "scope": "The report's displayed trajectory and heuristic finding bands. "
                 "Display thinning and off-scale caps remain; this is not the original log "
                 "or an independent diagnosis. Inspect the report and source log for evidence.",
    }
    # Visible previews stay within the fixed chart width. Full names and notes
    # remain in metadata, without trusting file-supplied names as markup.
    lines = ["TRAINSPOTTER / " + heading, "Source: " + (run.source_path or "(stdin)")]
    visible = []
    for line in lines:
        wrapped = textwrap.wrap(" ".join(line.split()), width=108) or [""]
        visible.extend(wrapped[:2])
        if len(wrapped) > 2:
            visible[-1] = visible[-1][:105] + "..."
    visible.append("Horizontal axis: " + run.axis_label())
    if not input_complete:
        visible.append("Input incomplete. Interpret the readable data with the original report.")
    elif input_notes:
        visible.append("Input notes apply. Read the original report before interpreting the curve.")
    if excluded:
        visible.append(textwrap.shorten("No trajectory: " + ", ".join(excluded), 108))
    header_height = 20 + 17 * len(visible)
    height = header_height + chart_height + 30
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="export-title">',
        '<title id="export-title">'
        + html.escape(heading + " over " + run.axis_label()) + '</title>',
        '<desc>Annotated chart from a Trainspotter report. Source and interpretation '
        'context are retained in the visible caption and metadata.</desc>',
        '<metadata>' + html.escape(json.dumps(metadata, ensure_ascii=True)) + '</metadata>',
        '<style>' + _EXPORT_STYLE + '</style>',
        f'<rect width="{width}" height="{height}" fill="{_THEME_VALUES["panel"].strip()}"/>',
    ]
    parts.extend(
        f'<text class="export-caption" x="18" y="{20 + 17 * i}">{html.escape(line)}</text>'
        for i, line in enumerate(visible)
    )
    # A nested viewport retains the original chart coordinate system exactly.
    parts.append(_portable_chart(chart, width, chart_height, header_height))
    parts.append(
        f'<text class="export-footer" x="18" y="{height - 10}">'
        'Read the original report and log for full diagnostic evidence. '
        'Shaded findings are heuristics.</text></svg>'
    )
    encoded = base64.b64encode(_xml_text("".join(parts)).encode("utf-8")).decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:70] or "chart"
    return (
        f'<a class="chart-download" download="trainspotter-{index}-{slug}.svg" '
        f'aria-label="Download {html.escape(heading, quote=True)} chart as SVG" '
        f'href="data:image/svg+xml;base64,{encoded}">Download SVG</a>'
    )
