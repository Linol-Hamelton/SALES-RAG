"""P16.A.2-train v3: reranker fine-tune with collision-aware negatives.

Input: SOFT_TUNE_DATA/reranker_training_pairs_v3.jsonl
  - Positives: top-5 refs per query, label 0.7-1.0
  - Hard negs: cross-direction refs, label 0.0
  - Soft negs: same-direction-different-topic refs, label 0.3
  - Collision skips: buklet/listovka cross-pollination filtered out at prep time

Output: SOFT_TUNE_DATA/reranker_finetuned_v3/

Method: CrossEncoder with BCEWithLogitsLoss on soft labels.
  - 80/20 train/val split
  - 3 epochs (v2 was 2 — v3 has cleaner labels, can train longer)
  - lr=5e-6 (lower than v2 lr=1e-5 — protect against overfitting)
  - Pre-train + post-train smoke including COLLISION cases.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
PAIRS = DATA / "reranker_training_pairs_v3.jsonl"
OUT_MODEL = DATA / "reranker_finetuned_v3"
BASE_MODEL = "BAAI/bge-reranker-v2-m3"

SEED = 42
VAL_FRAC = 0.20
EPOCHS = 3
BATCH = 8
LR = 5e-6  # Even lower than v2 (1e-5) — let model converge slowly on soft labels
MAX_SEQ_LEN = 512

# Smoke tests — v2 cases (cross-topic) + NEW collision cases
SMOKE_TESTS_CROSS_TOPIC = [
    ("Почему у вас так дорого", "Сделка #62122 на 40 500 ₽ — брендирование 8 макетов с гарантией 12 месяцев", "relevant"),
    ("Почему у вас так дорого", "История компании Лабус с 2009 года, основана в Махачкале", "irrelevant"),
    ("Хочу заказать буклеты для стоматологии", "Буклеты A5 фальцеванные с 4-цв печатью, бумага мелованная 130 г/м2", "relevant"),
    ("Сколько стоит логотип", "Разработка логотипа пакет Стандарт 25-44 тыс ₽, медиана 47 сделок", "relevant"),
    ("Сколько стоит логотип", "Доставка готового заказа курьером по Махачкале", "irrelevant"),
]

SMOKE_TESTS_COLLISION = [
    # query asks for buklet, listovka shouldn't be far away (same direction)
    ("Хочу заказать буклеты для стоматологии", "Листовки A6 одностороння печать, бумага офсет 80 г/м2", "loosely-relevant"),
    # query asks for visitka, flyer is same direction
    ("Сколько стоит визитка стандарт", "Листовка A5 4+0 250 шт", "loosely-relevant"),
    # query asks for visitka AND buklet — both relevant
    ("Нужны и визитки и буклеты для клиники", "Визитка 90×50 односторонняя 1000 шт", "relevant"),
    ("Нужны и визитки и буклеты для клиники", "Буклет А4 фальц 2 шт 4+4 500 шт", "relevant"),
    # cross-direction = strong negative
    ("Хочу заказать буклеты для стоматологии", "Брендмауэр 6×12 м на здание ТЦ", "irrelevant"),
]


def run_smoke(model, label, tests, threshold=0.3):
    print(f"\n--- {label} ---")
    oks = 0
    for q, d, expected in tests:
        score = float(model.predict([(q, d)])[0])
        if expected == "relevant":
            ok = score > threshold
        elif expected == "irrelevant":
            ok = score < threshold
        elif expected == "loosely-relevant":
            # Want 0.1 < score < 0.6 — neither full-positive nor zero
            ok = 0.05 < score < 0.7
        else:
            ok = False
        if ok:
            oks += 1
        marker = "✓" if ok else "✗"
        print(f"  {marker} {expected:18} score={score:+.3f}  Q={q[:50]}")
    print(f"  Result: {oks}/{len(tests)}")
    return oks


def main():
    print(f"Loading pairs from {PAIRS}...")
    if not PAIRS.exists():
        print(f"ERROR: {PAIRS} not found. Run reranker_train_prep_v3.py first.")
        return
    rows = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(SEED)
    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRAC)
    val = rows[:n_val]
    train = rows[n_val:]
    type_counts = {"pos": 0, "hard_neg": 0, "soft_neg": 0}
    for r in rows:
        t = r.get("type", "?")
        if t in type_counts:
            type_counts[t] += 1
    print(f"  total: {len(rows)}, train: {len(train)}, val: {len(val)}")
    print(f"  types: {type_counts}")

    train_examples = [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
        for r in train
    ]

    print(f"\nLoading base model: {BASE_MODEL}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_SEQ_LEN, device="cuda")

    # BASELINE smoke
    run_smoke(model, "BASELINE cross-topic", SMOKE_TESTS_CROSS_TOPIC, threshold=0.3)
    run_smoke(model, "BASELINE collision", SMOKE_TESTS_COLLISION, threshold=0.3)

    # Evaluator: binary, label > 0.5 → positive class
    val_pairs = [[r["query"], r["passage"]] for r in val]
    val_labels = [1 if r["label"] > 0.5 else 0 for r in val]
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=val_pairs,
        labels=val_labels,
        name="val_v3",
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

    # POST-train smoke
    print("\n=== Post-training smoke tests ===")
    cross_topic_score = run_smoke(fresh, "FINE-TUNED v3 cross-topic", SMOKE_TESTS_CROSS_TOPIC, threshold=0.3)
    collision_score = run_smoke(fresh, "FINE-TUNED v3 collision (key for print_flyer fix)", SMOKE_TESTS_COLLISION, threshold=0.3)

    print(f"\n✓ Model saved → {OUT_MODEL}")
    print(f"  Eval baseline → final: {baseline_eval} → {final_eval}")
    print(f"  Cross-topic smoke: {cross_topic_score}/{len(SMOKE_TESTS_CROSS_TOPIC)}")
    print(f"  Collision smoke:   {collision_score}/{len(SMOKE_TESTS_COLLISION)}")


if __name__ == "__main__":
    main()
