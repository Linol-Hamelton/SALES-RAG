"""P15.D-train: fine-tune bge-reranker-v2-m3 on collected training pairs.

Input:  SOFT_TUNE_DATA/reranker_training_pairs.jsonl (8466 pairs)
Output: SOFT_TUNE_DATA/reranker_finetuned/ (HF model dir, ready for deploy)

Method: CrossEncoder binary classification.
  - 80/20 train/val split (random with seed)
  - 3 epochs, batch_size=8 (fits 8GB VRAM with seq_len=512)
  - lr=2e-5, warmup 10%
  - Eval: accuracy + AUC on held-out val set

Expected: model.save() outputs sentence-transformers format usable by
existing reranker.py without code changes.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
PAIRS = DATA / "reranker_training_pairs.jsonl"
OUT_MODEL = DATA / "reranker_finetuned"
# Local dir has only tokenizer/config — fall back to HF Hub name for auto-download
BASE_MODEL = "BAAI/bge-reranker-v2-m3"

SEED = 42
VAL_FRAC = 0.20
EPOCHS = 3
BATCH = 8
LR = 2e-5
MAX_SEQ_LEN = 512


def load_pairs():
    rows = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(SEED)
    random.shuffle(rows)
    return rows


def main():
    print(f"Loading pairs from {PAIRS}...")
    rows = load_pairs()
    print(f"  {len(rows)} pairs total")
    n_val = int(len(rows) * VAL_FRAC)
    val = rows[:n_val]
    train = rows[n_val:]
    print(f"  train: {len(train)}, val: {len(val)}")

    train_examples = [InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
                       for r in train]
    val_examples = [(r["query"], r["passage"], r["label"]) for r in val]

    print(f"\nLoading base model: {BASE_MODEL}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_SEQ_LEN, device="cuda")

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)

    val_pairs = [[q, p] for q, p, _ in val_examples]
    val_labels = [l for _, _, l in val_examples]
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=val_pairs,
        labels=val_labels,
        name="val",
        write_csv=False,
    )

    # Baseline evaluation (before training)
    print("\nBaseline (pre-train) eval...")
    baseline = evaluator(model, output_path=None)
    print(f"  baseline metrics: {baseline}")

    print(f"\nFine-tuning {EPOCHS} epochs, batch={BATCH}, lr={LR}...")
    warmup_steps = int(len(train_loader) * EPOCHS * 0.1)
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        epochs=EPOCHS,
        evaluator=evaluator,
        evaluation_steps=int(len(train_loader) * 0.5),  # eval twice per epoch
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        output_path=str(OUT_MODEL),
        show_progress_bar=True,
    )

    print(f"\nFinal eval after training...")
    final = evaluator(model, output_path=None)
    print(f"  final metrics: {final}")

    print(f"\n✓ Model saved → {OUT_MODEL}")
    print(f"  Improvement: {final}")


if __name__ == "__main__":
    main()
