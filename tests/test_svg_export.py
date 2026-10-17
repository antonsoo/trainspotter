"""Portable charts must retain uncertainty and remain inert, valid XML."""
from __future__ import annotations

import base64
import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from trainspotter.model import Run
from trainspotter.report import render_html

NS = {"svg": "http://www.w3.org/2000/svg"}


class Downloads(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.charts: list[tuple[str, str]] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and "chart-download" in (values.get("class") or ""):
            self.charts.append((values.get("download") or "", values.get("href") or ""))


def exported(run: Run) -> list[ET.Element]:
    links = Downloads(render_html(run, [])).charts
    assert len({name for name, _ in links}) == len(links)
    result = []
    for name, href in links:
        assert name.endswith(".svg") and "/" not in name and "\\" not in name
        assert href.startswith("data:image/svg+xml;base64,")
        root = ET.fromstring(base64.b64decode(href.split(",", 1)[1], validate=True))
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.find("svg:style", NS) is not None
        result.append(root)
    return result


def metadata(root: ET.Element) -> dict[str, object]:
    node = root.find("svg:metadata", NS)
    assert node is not None and node.text
    value: dict[str, object] = json.loads(node.text)
    return value


def test_svg_retains_usable_metric_when_a_peer_trajectory_is_conflicted() -> None:
    run = Run(source_path="run.jsonl", source_format="jsonl")
    run.add_point("train/loss", 0, 2)
    run.add_point("train/loss", 0, 3)
    run.add_point("eval/loss", 0, 2)
    run.add_point("eval/loss", 1, 1)
    root, = exported(run)
    evidence = metadata(root)
    assert evidence["requestedMetrics"] == ["train/loss", "eval/loss"]
    assert evidence["plottedMetrics"] == ["eval/loss"]
    assert evidence["excludedMetrics"] == ["train/loss"]
    assert evidence["inputComplete"] is False
    assert evidence["sourcePath"] == "run.jsonl"
    assert "No trajectory: train/loss" in "".join(root.itertext())
    assert "Input incomplete" in "".join(root.itertext())
    # With no readable peer, no trajectory or download is invented.
    del run.metrics["eval/loss"]
    assert exported(run) == []


def test_svg_keeps_gaps_nonfinite_markers_and_the_actual_axis_meaning() -> None:
    run = Run(meta={"step_axis": "logging_step", "step_column": "_step"})
    for step, value in enumerate([.4, None, .3, float("inf"), .2]):
        run.add_point("train/loss", step, value)
    root, = exported(run)
    evidence = metadata(root)
    assert evidence["axisKind"] == "logging_step"
    assert evidence["axisLabel"] == "logging steps"
    assert evidence["axisSource"] == "_step"
    assert evidence["inputComplete"] is False
    assert len(root.findall('.//svg:circle[@class="marker-single"]', NS)) == 3
    assert len(root.findall('.//svg:text[@class="marker-unreadable"]', NS)) == 1
    assert len(root.findall('.//svg:text[@class="marker-bad"]', NS)) == 1
    assert "Horizontal axis: logging steps" in "".join(root.itertext())


def test_svg_names_remain_text_and_illegal_xml_characters_cannot_break_the_file() -> None:
    name = '<script onload="bad()"> & ]]> 雨\x00\x01'
    run = Run(source_path=name, meta={"metric_sources": {name: "original ' column"}})
    run.add_point(name, 0, 1)
    run.add_point(name, 1, 2)
    root, = exported(run)
    evidence = metadata(root)
    assert evidence["sourcePath"] == name
    assert evidence["metricSources"] == {name: "original ' column"}
    assert "\ufffd" in "".join(root.itertext())
    assert not root.findall(".//svg:script", NS)
    assert not root.findall(".//svg:image", NS)
    assert not root.findall(".//svg:foreignObject", NS)
    for element in root.iter():
        assert not any(key.lower().startswith("on") for key in element.attrib)
