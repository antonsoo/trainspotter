# Contributing

trainspotter is a small, focused tool; contributions that fit its scope are
welcome.

## Setup

```bash
git clone https://github.com/antonsoo/trainspotter.git
cd trainspotter
uv sync --all-extras --dev
```

## Before opening a PR

```bash
uv run ruff check src tests examples
uv run mypy src
uv run pytest
```

All three must pass. If you add a detector, follow the existing pattern:
a class-level docstring documenting the algorithm and its false-positive
modes, plus tests with a planted-signal positive and a clean-curve
negative (see `tests/test_detector_*.py`).

## Regenerating the example logs

```bash
uv pip install scikit-learn  # or: uv sync --extra examples
uv run python examples/generate_examples.py
```

This overwrites the committed files in `examples/`; it's deterministic
(seeded), so a clean run should reproduce byte-identical output.

## Reporting a false positive or false negative

Open an issue with the log file (or enough of it to reproduce) and which
detector misfired. Every detector's docstring lists its known
false-positive modes -- if yours isn't one of them, that's a bug.
