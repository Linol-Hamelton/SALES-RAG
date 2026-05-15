"""P15.D-train v2: PROPER reranker fine-tune with soft labels + hard negatives.

Input: SOFT_TUNE_DATA/reranker_training_pairs_v2.jsonl
  - Positives: top-5 refs per query, label 0.7-1.0 (weighted by judge winner)
  - Negatives: cross-topic refs, label 0.0
  - Ratio: 2 neg per pos (8.5k pos / 16.9k neg)

Output: SOFT_TUNE_DATA/reranker_finetuned_v2/

Method: CrossEncoder with BCEWithLogitsLoss on soft labels.
  - 80/20 train/val split
  - 2 epochs (less than v1: smaller risk of overfitting to weak signals)
  - Pre-train validation with our smoke set + post-train comparison
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
PAIRS = DATA / "reranker_training_pairs_v2.jsonl"
OUT_MODEL = DATA / "reranker_finetuned_v2"
BASE_MODEL = "BAAI/bge-reranker-v2-m3"

SEED = 42
VAL_FRAC = 0.20
EPOCHS = 2
BATCH = 8
LR = 1e-5  # Lower LR than v1 (be more conservative)
MAX_SEQ_LEN = 512

# Smoke test cases for sanity check
SMOKE_TESTS = [
    ("Почему у вас так дорого", "Сделка #62122 на 40 500 ₽ — брендирование 8 макетов с гарантией 12 месяцев", "relevant"),
    ("Почему у вас так дорого", "История компании Лабус с 2009 года, основана в Махачкале", "irrelevant"),
    ("Хочу заказать буклеты для стоматологии", "Буклеты A5 фальцеванные с 4-цв печатью, бумага мелованная 130 г/м2", "relevant"),
    ("Хочу заказать буклеты для стоматологии", "Листовки A6 одностороння печать, бумага офсет 80 г/м2", "irrelevant"),
    ("Сколько стоит логотип", "Разработка логотипа пакет Стандарт 25-44 тыс ₽, медиана 47 сделок", "relevant"),
    ("Сколько стоит логотип", "Доставка готового заказа курьером по Махачкале", "irrelevant"),
]


def run_smoke(model, label):
    print(f"\n--- Smoke test ({label}) ---")
    for q, d, expected in SMOKE_TESTS:
        score = float(model.predict([(q, d)])[0])
        marker = "✓" if (expected == "relevant" and score > 0.3) or (expected == "irrelevant" and score < 0.3) else "✗"
        print(f"  {marker} {expected:11} | score={score:+.3f}  Q={q[:50]}")


def main():
    print(f"Loading pairs from {PAIRS}...")
    rows = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(SEED)
    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRAC)
    val = rows[:n_val]
    train = rows[n_val:]
    print(f"  total: {len(rows)}, train: {len(train)}, val: {len(val)}")

    train_examples = [InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
                       for r in train]

    print(f"\nLoading base model: {BASE_MODEL}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_SEQ_LEN, device="cuda")

    # BASELINE smoke test
    run_smoke(model, "BASELINE before training")

    # Evaluator: binary, threshold 0.5
    val_pairs = [[r["query"], r["passage"]] for r in val]
    val_labels = [1 if r["label"] > 0.5 else 0 for r in val]
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=val_pairs,
        labels=val_labels,
        name="val",
        write_csv=False,
    )
    baseline_eval = evaluator(model, output_path=None)
    print(f"\nBaseline binary eval: {baseline_eval}")

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)

    print(f"\nFine-tuning {EPOCHS} epochs, batch={BATCH}, lr={LR}...")
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(len(train_loader) * EPOCHS * 0.1)
    model.fit(
        train_dataloader=train_loader,
        epochs=EPOCHS,
        evaluator=evaluator,
        evaluation_steps=int(len(train_loader) * 0.5),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        output_path=str(OUT_MODEL),
        show_progress_bar=True,
    )

    # POST-train evaluation
    print(f"\nLoading saved model for fresh eval...")
    fresh = CrossEncoder(str(OUT_MODEL), max_length=MAX_SEQ_LEN, device="cuda")
    final_eval = evaluator(fresh, output_path=None)
    print(f"\nFinal binary eval: {final_eval}")

    # POST-train smoke test
    run_smoke(fresh, "FINE-TUNED after training")
    print(f"\n✓ Model saved → {OUT_MODEL}")
    print(f"  Eval baseline → final: {baseline_eval} → {final_eval}")


if __name__ == "__main__":
    main()
