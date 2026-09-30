from __future__ import annotations

import re
from pathlib import Path

from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.report.html_report import _nice_ticks, render_html


def test_ticks_stay_inside_the_plotted_range() -> None:
    assert _nice_ticks(0.0, 4.4, 3) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _nice_ticks(0.0, 322.0, 4) == [0.0, 100.0, 200.0, 300.0]
    for lo, hi in [(0.0, 0.00017), (0.0, 0.6480000001), (-3.2, 7.9), (0.0, 1.08)]:
        ticks = _nice_ticks(lo, hi, 3)
        assert ticks, (lo, hi)
        assert all(lo <= t <= hi for t in ticks), (lo, hi, ticks)


def test_no_gridline_is_drawn_outside_a_chart_frame() -> None:
    # The divergence example's loss chart: its padded, capped axis ends between two nice ticks.
    run = load_run(Path(__file__).parent.parent / "examples" / "divergence.trainer_state.json")
    html = render_html(run, run_all(run))
    svgs = re.findall(r'<svg class="chart".*?</svg>', html, flags=re.S)
    assert svgs
    for svg in svgs:
        grid = re.findall(
            r'<line class="grid" x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"', svg
        )
        vertical = [(float(y1), float(y2)) for x1, y1, x2, y2 in grid if x1 == x2]
        horizontal = [float(y1) for x1, y1, x2, y2 in grid if y1 == y2]
        top, bottom = vertical[0]
        assert horizontal
        assert all(top - 0.5 <= y <= bottom + 0.5 for y in horizontal), (top, bottom, horizontal)
