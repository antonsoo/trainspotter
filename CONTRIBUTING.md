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

This overwrites the committed files in `examples/`. Every seeded quantity
(loss, LR, grad_norm, which steps get evaluated) reproduces bit-for-bit;
`step_time` is genuinely measured wall-clock time, so it -- and only it --
will differ from the committed files on every regeneration, including the
throughput-regression run's slowdown ratio (it'll still be a real
regression, just not exactly 1.90x).

## Reporting a false positive or false negative

Open an issue with the log file (or enough of it to reproduce) and which
detector misfired. Every detector's docstring lists its known
false-positive modes -- if yours isn't one of them, that's a bug.

## Community and private reports

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Anton Soloviev
maintains this project and handles conduct reports at
[anton@praviel.com](mailto:anton@praviel.com).

Use the bug or improvement forms for public issues. For a suspected security
vulnerability or a conduct concern, email the maintainer privately with the
repository name and relevant details. Do not post credentials, personal data,
private logs, or confidential documents in a public issue.
