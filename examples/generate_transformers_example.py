"""Generate examples/transformers_tinygpt2.trainer_state.json: a REAL run of
`transformers.Trainer` (not trainspotter's own numpy MLP) on a tiny 2-layer
GPT-2, so the HF reader is also demonstrated against a `trainer_state.json`
that transformers itself wrote.

This needs `torch` and `transformers`, which are NOT project dependencies
(trainspotter's own detectors/readers need only numpy) -- install them in a
throwaway environment to regenerate this file:

    uv venv /tmp/ts-transformers-venv
    uv pip install --python /tmp/ts-transformers-venv/bin/python torch \\
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/ts-transformers-venv/bin/python transformers
    /tmp/ts-transformers-venv/bin/python examples/generate_transformers_example.py

The corpus is a small, hand-written toy corpus (repeated short sentences
about training, not real prose from any external source) -- there's no
claim this model learns anything useful; the point is a real
`trainer_state.json` with a real (if uninteresting) loss curve, not a
benchmark result.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments

OUT_DIR = Path(__file__).parent
SEED = 0

TOY_SENTENCES = [
    "the loss went down after the warmup finished",
    "gradient clipping kept the norm under control",
    "the learning rate followed a cosine schedule",
    "the model overfit once the dropout was removed",
    "eval accuracy improved for the first few epochs",
    "the optimizer used weight decay on every layer",
    "a spike in the loss meant the batch was corrupted",
    "the scheduler warmed up over the first hundred steps",
    "checkpoints were saved every five hundred steps",
    "the validation set was held out before training started",
    "batch size affected both throughput and convergence",
    "the tokenizer split rare words into subword pieces",
    "attention weights were visualized after training",
    "the run diverged when the learning rate was too high",
    "early stopping used the best validation loss",
]


class ToyCorpusDataset(Dataset):
    def __init__(self, tokenizer: GPT2TokenizerFast, block_size: int, n_examples: int, seed: int):
        rng = random.Random(seed)
        text = " . ".join(rng.choice(TOY_SENTENCES) for _ in range(n_examples * 8))
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        n_blocks = len(ids) // block_size
        self.examples = [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.examples[idx]
        return {"input_ids": chunk, "labels": chunk.clone()}


def main() -> None:
    torch.manual_seed(SEED)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=64,
        n_ctx=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    train_ds = ToyCorpusDataset(tokenizer, block_size=32, n_examples=400, seed=SEED)
    eval_ds = ToyCorpusDataset(tokenizer, block_size=32, n_examples=60, seed=SEED + 1)

    output_dir = OUT_DIR / "_tmp_transformers_run"
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-4,
        warmup_steps=20,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_state()  # save_strategy="no" skips checkpoints, but not this

    state_path = output_dir / "trainer_state.json"
    dest = OUT_DIR / "transformers_tinygpt2.trainer_state.json"
    dest.write_text(state_path.read_text())
    shutil.rmtree(output_dir)
    print(f"wrote {dest} ({len(trainer.state.log_history)} log entries)")


if __name__ == "__main__":
    main()
