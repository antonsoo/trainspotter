"""Shared test helpers: build a Run directly from arrays, no file I/O needed."""

from __future__ import annotations

from trainspotter.model import Run


def make_run(**metrics: list[float]) -> Run:
    """`make_run(**{"train/loss": [...]})` -> a Run with one step per index (0..n-1)."""
    run = Run(source_format="synthetic", source_path="<test>")
    for name, values in metrics.items():
        for step, value in enumerate(values):
            run.add_point(name, step, value)
    run.finalize()
    return run


def make_run_steps(name: str, steps: list[int], values: list[float]) -> Run:
    """Build a single-metric Run with explicit (possibly non-contiguous) steps."""
    run = Run(source_format="synthetic", source_path="<test>")
    for step, value in zip(steps, values, strict=True):
        run.add_point(name, step, value)
    run.finalize()
    return run
