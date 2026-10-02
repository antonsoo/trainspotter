"""Metric-name normalization shared by every reader.

Different frameworks spell the same thing differently: `loss`, `train_loss`,
`train/loss`, `Loss/train`, `train_loss_step`; `learning_rate`,
`train/learning_rate` (what `transformers.Trainer` sends to W&B and
TensorBoard), `lr-AdamW` (Lightning's `LearningRateMonitor`). The detectors
look for a small fixed vocabulary -- `train/loss`, `eval/loss`, `lr`,
`grad_norm`, `step_time`, `throughput`, `epoch` -- so readers map the names
they recognize onto it and pass everything else through unchanged.

A canonical name belongs to one logged metric. When a log has several that
could be it (`train_loss_step` and `train_loss_epoch`; `lr-AdamW/pg1` and
`lr-AdamW/pg2`), the most direct one takes the name and the others keep
their own: merged, they would be one series with two values at a step.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainspotter.model import Run

_EVAL_SPLITS = ("eval", "val", "valid", "validation", "dev")

# canonical -> token sequences, best first: a name is split on "/", "_", "-", "." and spaces,
# lower-cased, and compared whole. The position in the tuple is the rank a name claims with.
_TOKEN_NAMES: dict[str, tuple[tuple[str, ...], ...]] = {
    "train/loss": (
        ("loss",),
        ("train", "loss"),
        ("training", "loss"),
        ("loss", "train"),
        ("loss", "training"),
        ("train", "loss", "step"),
        ("batch", "loss"),
        ("train", "batch", "loss"),
        ("train", "loss", "epoch"),
        ("epoch", "loss"),
        ("train", "epoch", "loss"),
    ),
    "eval/loss": (
        *((split, "loss") for split in _EVAL_SPLITS),
        *(("loss", split) for split in _EVAL_SPLITS),
        *((split, "loss", "epoch") for split in _EVAL_SPLITS),
        *((split, "epoch", "loss") for split in _EVAL_SPLITS),
        *((split, "loss", "step") for split in _EVAL_SPLITS),
    ),
    "lr": (
        ("lr",),
        ("learning", "rate"),
        ("train", "lr"),
        ("train", "learning", "rate"),
        ("optim", "lr"),
        ("optimizer", "lr"),
        ("trainer", "lr"),
        ("charts", "learning", "rate"),
        ("epoch", "learning", "rate"),
        ("epoch", "lr"),
        ("train", "epoch", "learning", "rate"),
        ("train", "epoch", "lr"),
    ),
    "grad_norm": (
        ("grad", "norm"),
        ("gradient", "norm"),
        ("train", "grad", "norm"),
        ("train", "gradient", "norm"),
        ("grad", "norm", "global"),
        ("global", "grad", "norm"),
        ("total", "grad", "norm"),
        ("grad", "norm", "before", "clip"),
    ),
    "step_time": (
        ("step", "time"),
        ("time", "per", "step"),
        ("iter", "time"),
        ("seconds", "per", "step"),
        ("train", "step", "time"),
    ),
    "throughput": (
        ("samples", "per", "second"),
        ("train", "samples", "per", "second"),
        ("throughput",),
        ("tokens", "per", "second"),
        ("train", "tokens", "per", "second"),
    ),
    "epoch": (("epoch",), ("train", "epoch")),
}

_BY_TOKENS: dict[tuple[str, ...], tuple[str, int]] = {
    tokens: (canon, rank)
    for canon, names in _TOKEN_NAMES.items()
    for rank, tokens in enumerate(names)
}
_CANONICAL = frozenset(_TOKEN_NAMES)

# Lightning's LearningRateMonitor: "lr-AdamW", and "lr-AdamW/pg1" per parameter group;
# Ultralytics: "lr/pg0". The first group is the schedule; the rest follow it.
_LR_PER_OPTIMIZER = re.compile(r"^lr[-/]([A-Za-z0-9]+?)(?:[-/]?pg(\d+))?$")
_FIRST_LR_RANK = 100


def _tokens(raw: str) -> tuple[str, ...]:
    return tuple(t for t in re.split(r"[/_.\-\s]+", raw.strip().lower()) if t)


def _claim(raw: str) -> tuple[str, int] | None:
    """The canonical name `raw` would take, and how good a claim it has (lower is better)."""
    key = raw.strip()
    if key in _CANONICAL:
        return key, -1
    found = _BY_TOKENS.get(_tokens(key))
    if found is not None:
        return found
    match = _LR_PER_OPTIMIZER.match(key)
    if match is not None:
        optimizer, group = match.group(1), match.group(2)
        if group is None and re.fullmatch(r"pg\d+", optimizer):  # "lr/pg0": no optimizer named
            group = optimizer[2:]
        return "lr", _FIRST_LR_RANK + (int(group) if group else 0)
    return None


def _passthrough(raw: str) -> str:
    key = raw.strip()
    # eval_* -> eval/* for anything not covered above (e.g. eval_accuracy)
    if key.lower().startswith("eval_"):
        return "eval/" + key[len("eval_") :]
    return key


def build_key_map(raw_keys: Iterable[str]) -> dict[str, str]:
    """Map every raw metric/column name of one log to the name it is stored under.

    Recognized names take a canonical name, one raw name per canonical name (the
    best claim, the earliest on a tie). Everything else passes through, so nothing
    is silently dropped.
    """
    keys = list(dict.fromkeys(raw_keys))
    winners: dict[str, tuple[int, int, str]] = {}
    for order, raw in enumerate(keys):
        claim = _claim(raw)
        if claim is None:
            continue
        canon, rank = claim
        if canon not in winners or (rank, order) < winners[canon][:2]:
            winners[canon] = (rank, order, raw)
    taken = {raw: canon for canon, (_, _, raw) in winners.items()}
    out: dict[str, str] = {}
    for raw in keys:
        if raw in taken:
            out[raw] = taken[raw]
            continue
        name = _passthrough(raw)
        # A name that lost its claim must not land on the canonical name by the back door.
        out[raw] = raw.strip() if name in winners else name
    return out


def normalize_key(raw: str) -> str:
    """Map one raw metric/column name to trainspotter's canonical vocabulary.
    Readers map a log's names together with `build_key_map`; this is that for a single name."""
    return build_key_map([raw])[raw]


def try_float(value: object) -> float | None:
    """Best-effort numeric coercion. Returns None only for genuinely missing
    values (blank cells, JSON null, non-numeric strings) -- NOT for NaN/Inf,
    which are real, meaningful readings (a diverged run legitimately logs
    them) and must reach the divergence detector, not be swallowed here."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"none", "null"}:
            return None
        try:
            return float(s)  # float() itself parses "nan"/"inf"/"-inf"
        except ValueError:
            return None
    return None


# Beyond 2**53 a float no longer holds every integer, so a larger "step" is not a step count.
_MAX_STEP = 2**53


def try_step(value: object) -> int | None:
    """A step number, or None when the value is missing or can't be one (NaN, an
    infinity, a number too large to be a count). Unlike a metric, a step that is
    not a finite number places nothing on the x-axis."""
    number = try_float(value)
    if number is None or not -_MAX_STEP <= number <= _MAX_STEP:  # NaN fails both comparisons
        return None
    return int(number)



def select_run(run: Run, counts: dict[str, int], wanted: str | None, what: str) -> str:
    """Pick the run to analyse out of a source that holds several, and record the choice.

    `counts` maps each run's name to how many points it has, in the order the source lists
    them. With `wanted`, that run (its name, or the start of it if only one run starts that
    way); otherwise the run with the most points, the first on a tie. `run.meta["runs"]`
    lists every run and `run.meta["selected_run"]` names the one chosen, so that a report
    built on one run out of several says so.
    """
    if not counts:
        raise ValueError(f"{what} has no runs in it")
    if wanted is None:
        chosen = max(counts, key=lambda name: counts[name])
    elif wanted in counts:
        chosen = wanted
    else:
        matches = [name for name in counts if name.startswith(wanted)]
        if len(matches) != 1:
            listing = ", ".join(repr(name) for name in counts)
            problem = "matches no run" if not matches else f"is the start of {len(matches)} runs"
            raise ValueError(f"{what}: --run {wanted!r} {problem}. It holds: {listing}")
        chosen = matches[0]
    if len(counts) > 1:
        run.meta["runs"] = [{"name": name, "points": n} for name, n in counts.items()]
        run.meta["selected_run"] = chosen
    return chosen
