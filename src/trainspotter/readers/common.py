"""Metric-name normalization shared by every reader.

Different frameworks spell the same thing differently ("loss" vs "train_loss"
vs "train/loss"). Detectors need a stable vocabulary to look for, so readers
map recognized names to a canonical form and pass everything else through
unchanged (namespaced readers like W&B already write "train/loss" and need
no mapping at all).
"""

from __future__ import annotations

# canonical -> raw spellings seen in the wild, checked case-insensitively
# after stripping a leading "train_"/"train/" where relevant.
_ALIASES: dict[str, tuple[str, ...]] = {
    "train/loss": ("loss", "train_loss", "training_loss"),
    "eval/loss": ("eval_loss", "eval/loss", "val_loss", "validation_loss"),
    "lr": ("learning_rate", "lr", "train_lr"),
    "grad_norm": ("grad_norm", "gradient_norm", "grad_norm_before_clip", "grad_norm/global"),
    "step_time": ("step_time", "time_per_step", "iter_time", "seconds_per_step"),
    "throughput": (
        "samples_per_second",
        "train_samples_per_second",
        "throughput",
        "tokens_per_second",
    ),
    "epoch": ("epoch",),
}

_RAW_TO_CANON: dict[str, str] = {
    raw.lower(): canon for canon, raws in _ALIASES.items() for raw in raws
}


def normalize_key(raw: str) -> str:
    """Map a raw metric/column name to trainspotter's canonical vocabulary.
    Unrecognized names pass through unchanged so nothing is silently dropped."""
    key = raw.strip()
    canon = _RAW_TO_CANON.get(key.lower())
    if canon is not None:
        return canon
    # eval_* -> eval/* for anything not already covered above (e.g. eval_accuracy)
    if key.lower().startswith("eval_"):
        return "eval/" + key[len("eval_") :]
    return key


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
