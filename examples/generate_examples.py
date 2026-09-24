"""Generate the example logs committed under examples/.

Trains a small (64-32-10) numpy MLP on scikit-learn's `load_digits` dataset
(1797 real 8x8 handwritten-digit images, ships with scikit-learn, no
download) with five different, deliberately chosen training configurations:
one clean baseline, and four that induce a specific, real pathology by
actually training a worse run -- not by editing numbers after the fact.

Every number in the committed logs came out of an actual forward/backward
pass on real data; `step_time` is genuinely measured wall-clock time,
including where a run's config makes it slower on purpose (see
`_throughput_drop_run`).

Run: `python examples/generate_examples.py` (needs numpy + scikit-learn;
`pip install trainspotter[examples]` or `uv pip install scikit-learn`).
Deterministic: every run is seeded.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# A minimal numpy MLP: 64 -> 32 (ReLU) -> 10 (softmax), trained by hand so
# every quantity logged (loss, grad_norm, step_time) is directly computed
# here, not borrowed from a framework's own logging.
# ---------------------------------------------------------------------------


class MLP:
    def __init__(
        self, rng: np.random.Generator, n_in: int = 64, n_hidden: int = 32, n_out: int = 10
    ):
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.w1 = rng.normal(0, scale1, size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, scale2, size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        logits = a1 @ self.w2 + self.b2
        return a1, logits

    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, weight_decay: float, *, stable: bool = True
    ) -> tuple[float, dict[str, np.ndarray]]:
        n = x.shape[0]
        a1, logits = self.forward(x)
        if stable:
            logits = logits - logits.max(axis=1, keepdims=True)
        # else: the classic real bug -- skip the numerically-stable softmax
        # normalization, used deliberately for the divergence example below.
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        log_probs = logits - np.log(exp.sum(axis=1, keepdims=True))
        loss = -log_probs[np.arange(n), y].mean()
        loss += 0.5 * weight_decay * (np.sum(self.w1**2) + np.sum(self.w2**2))

        d_logits = probs.copy()
        d_logits[np.arange(n), y] -= 1.0
        d_logits /= n

        grad_w2 = a1.T @ d_logits + weight_decay * self.w2
        grad_b2 = d_logits.sum(axis=0)
        d_a1 = d_logits @ self.w2.T
        d_z1 = d_a1 * (a1 > 0)
        grad_w1 = x.T @ d_z1 + weight_decay * self.w1
        grad_b1 = d_z1.sum(axis=0)

        return float(loss), {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.w1 -= lr * grads["w1"]
        self.b1 -= lr * grads["b1"]
        self.w2 -= lr * grads["w2"]
        self.b2 -= lr * grads["b2"]

    def eval_loss_acc(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        _a1, logits = self.forward(x)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = x.shape[0]
        log_probs = logits - np.log(exp.sum(axis=1, keepdims=True))
        loss = float(-log_probs[np.arange(n), y].mean())
        acc = float((probs.argmax(axis=1) == y).mean())
        return loss, acc


def grad_norm(grads: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


# ---------------------------------------------------------------------------
# The five runs.
# ---------------------------------------------------------------------------


def load_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0  # pixel intensities are 0..16
    y = digits.target.astype(np.int64)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    return x_train, y_train, x_val, y_val


def _train_loop(
    *,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    total_steps: int,
    batch_size: int,
    weight_decay: float,
    lr_fn: Any,
    eval_every: int = 5,
    artificial_slowdown_steps: range | None = None,
    slowdown_seconds: float = 0.02,
    unstable_from_step: int | None = None,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    model = MLP(rng)
    log_history: list[dict[str, float]] = []
    n = x_train.shape[0]
    idx = rng.permutation(n)
    cursor = 0

    for step in range(total_steps):
        t0 = time.perf_counter()
        if cursor + batch_size > n:
            idx = rng.permutation(n)
            cursor = 0
        batch_idx = idx[cursor : cursor + batch_size]
        cursor += batch_size

        lr = lr_fn(step)
        stable = unstable_from_step is None or step < unstable_from_step
        loss, grads = model.loss_and_grads(
            x_train[batch_idx], y_train[batch_idx], weight_decay, stable=stable
        )
        gnorm = grad_norm(grads)
        if np.isfinite(loss) and abs(loss) < 1e6:
            model.step(grads, lr)

        if artificial_slowdown_steps is not None and step in artificial_slowdown_steps:
            # A genuine, measured stall (e.g. simulating I/O contention), not a fabricated number.
            time.sleep(slowdown_seconds)
        step_time = time.perf_counter() - t0

        entry: dict[str, float] = {
            "step": step,
            "epoch": round(step * batch_size / n, 4),
            "loss": loss,
            "learning_rate": lr,
            "grad_norm": gnorm,
            "step_time": step_time,
        }
        log_history.append(entry)

        if step % eval_every == 0 or step == total_steps - 1:
            eval_loss, eval_acc = model.eval_loss_acc(x_val, y_val)
            log_history.append(
                {
                    "step": step,
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_acc,
                    "epoch": entry["epoch"],
                }
            )

        if not np.isfinite(loss):
            # Once it's diverged there's nothing more to learn from continuing;
            # a real training job would crash or be killed here too.
            break

    return log_history


def baseline_run(x_train, y_train, x_val, y_val) -> list[dict[str, float]]:  # type: ignore[no-untyped-def]
    total_steps = 400
    return _train_loop(
        seed=0,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        total_steps=total_steps,
        batch_size=32,
        weight_decay=1e-3,
        lr_fn=lambda s: cosine_with_warmup(s, total_steps, warmup_steps=20, peak_lr=0.5),
    )


def divergence_run(x_train, y_train, x_val, y_val) -> list[dict[str, float]]:  # type: ignore[no-untyped-def]
    # Trains normally for 25 steps, then simulates a real incident: an LR
    # schedule bug (a bad resume, or a misconfigured scheduler) drives the
    # LR to 30 and, at the same time, the loss switches to a version with
    # the classic missing-max-subtraction softmax bug. Softmax
    # cross-entropy's gradient is bounded by construction, so plain high LR
    # alone just made this small MLP oscillate near chance level rather
    # than actually diverge (see the README's honesty note) -- this
    # combination is what reliably reproduces a real NaN.
    warmup_phase = 25
    return _train_loop(
        seed=1,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        total_steps=60,
        batch_size=32,
        weight_decay=0.0,
        lr_fn=lambda s: 0.15 if s < warmup_phase else 30.0,
        eval_every=5,
        unstable_from_step=warmup_phase,
    )


def overfitting_run(x_train, y_train, x_val, y_val) -> list[dict[str, float]]:  # type: ignore[no-untyped-def]
    # Only 4 examples per class: the model can memorize this easily.
    rng = np.random.default_rng(2)
    small_idx = []
    for cls in range(10):
        cls_idx = np.where(y_train == cls)[0]
        small_idx.extend(rng.choice(cls_idx, size=4, replace=False))
    small_idx = np.array(small_idx)
    total_steps = 800  # long enough for eval loss to clearly turn upward, not just plateau
    return _train_loop(
        seed=2,
        x_train=x_train[small_idx],
        y_train=y_train[small_idx],
        x_val=x_val,
        y_val=y_val,
        total_steps=total_steps,
        batch_size=8,
        weight_decay=0.0,  # no regularization, by design
        lr_fn=lambda s: cosine_with_warmup(s, total_steps, warmup_steps=10, peak_lr=0.5),
        eval_every=5,
    )


def missing_warmup_run(x_train, y_train, x_val, y_val) -> list[dict[str, float]]:  # type: ignore[no-untyped-def]
    total_steps = 250
    return _train_loop(
        seed=3,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        total_steps=total_steps,
        batch_size=32,
        weight_decay=1e-3,
        lr_fn=lambda s: 0.5,  # jumps straight to peak LR, no ramp
        eval_every=5,
    )


def throughput_drop_run(x_train, y_train, x_val, y_val) -> list[dict[str, float]]:  # type: ignore[no-untyped-def]
    total_steps = 300
    return _train_loop(
        seed=4,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        total_steps=total_steps,
        batch_size=32,
        weight_decay=1e-3,
        lr_fn=lambda s: cosine_with_warmup(s, total_steps, warmup_steps=20, peak_lr=0.5),
        eval_every=10,
        artificial_slowdown_steps=range(150, 260),
        slowdown_seconds=0.02,
    )


# ---------------------------------------------------------------------------
# Writers: HF trainer_state.json and a generic CSV, from the same log_history.
# ---------------------------------------------------------------------------


def write_hf_json(name: str, log_history: list[dict[str, float]]) -> None:
    path = OUT_DIR / f"{name}.trainer_state.json"
    payload = {"log_history": log_history, "best_metric": None}
    path.write_text(json.dumps(payload, indent=2))


def write_csv(name: str, log_history: list[dict[str, float]]) -> None:
    path = OUT_DIR / f"{name}.csv"
    fieldnames: list[str] = []
    for entry in log_history:
        for key in entry:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in log_history:
            writer.writerow(entry)


def main() -> None:
    # The divergence run deliberately overflows float64 (that's the point --
    # see divergence_run's docstring); silence numpy's warning about it here
    # rather than have every regeneration print a scary-looking traceback.
    np.seterr(over="ignore", invalid="ignore")
    x_train, y_train, x_val, y_val = load_data(seed=0)
    runs = {
        "baseline": baseline_run(x_train, y_train, x_val, y_val),
        "divergence": divergence_run(x_train, y_train, x_val, y_val),
        "overfitting": overfitting_run(x_train, y_train, x_val, y_val),
        "missing_warmup": missing_warmup_run(x_train, y_train, x_val, y_val),
        "throughput_drop": throughput_drop_run(x_train, y_train, x_val, y_val),
    }
    for name, log_history in runs.items():
        write_hf_json(name, log_history)
        write_csv(name, log_history)
        n_train_steps = sum(1 for e in log_history if "loss" in e)
        print(f"{name}: {n_train_steps} train steps logged ({len(log_history)} total entries)")


if __name__ == "__main__":
    main()
