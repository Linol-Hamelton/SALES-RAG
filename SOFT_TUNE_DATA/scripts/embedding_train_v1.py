"""P21.B.2: BGE-M3 embedding fine-tune via MultipleNegativesRankingLoss.

Direct training loop (bypass model.fit() trainer API to avoid Windows
multiprocessing issues with sentence-transformers 3.x+ + datasets).

Input: SOFT_TUNE_DATA/embedding_training_pairs_v1.jsonl (~7,778 pairs)
Output: SOFT_TUNE_DATA/embedding_finetuned_v1/ (~2.3GB safetensors)
"""
from __future__ import annotations

import json
import os
import random
import sys
import traceback
from pathlib import Path

# Suppress tokenizer parallelism warnings / multiprocessing issues on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
PAIRS = DATA / "embedding_training_pairs_v1.jsonl"
OUT_MODEL = DATA / "embedding_finetuned_v1"
BASELINE_NPY = DATA / "embedding_baseline_smoke.npy"
BASE_MODEL = "BAAI/bge-m3"

SEED = 42
VAL_FRAC = 0.10
EPOCHS = 3
BATCH = 16   # P21.B fix: reduced from 32 to 16 to avoid possible OOM on RTX 4060 8GB
LR = 5e-6
MAX_SEQ_LEN = 512
WARMUP_FRAC = 0.10
GRAD_ACCUM_STEPS = 2  # effective batch = BATCH * GRAD_ACCUM_STEPS = 32

SMOKE_TESTS = [
    ("Сколько стоят 1000 буклетов А5", "Буклеты А5 фальцованные печать 4+4 мелованная 130 г/м2", "high"),
    ("Сколько стоит логотип для кафе", "Разработка логотипа Стандарт пакет 25-44 тыс ₽", "high"),
    ("Печать визиток 500 штук", "Визитки 90×50 односторонняя печать 4+0 250 г/м2", "high"),
    ("Сколько стоят буклеты А5", "Листовки А6 одностороння печать 80 г/м2", "mid"),
    ("Печать визиток 500 штук", "Листовки А5 250 г/м2 ламинация", "mid"),
    ("Сколько стоят буклеты А5", "Брендмауэр 6×12 м на здание ТЦ", "low"),
    ("Сколько стоит логотип", "Доставка готового заказа курьером", "low"),
    ("Печать визиток 500 штук", "Объёмные буквы высотой 50 см с подсветкой LED", "low"),
]


def run_smoke(model, label):
    print(f"\n--- Smoke ({label}) ---", flush=True)
    passed = 0
    cosines = []
    for q, p, expected in SMOKE_TESTS:
        with torch.no_grad():
            emb_q = model.encode(q, normalize_embeddings=True, convert_to_numpy=True)
            emb_p = model.encode(p, normalize_embeddings=True, convert_to_numpy=True)
        cos = float(np.dot(emb_q, emb_p))
        cosines.append(cos)
        if expected == "high":
            ok = cos > 0.6
        elif expected == "mid":
            ok = 0.3 < cos < 0.7
        elif expected == "low":
            ok = cos < 0.4
        else:
            ok = False
        marker = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"  {marker} {expected:<5} cos={cos:+.3f}  Q={q[:50]}", flush=True)
    print(f"  Result: {passed}/{len(SMOKE_TESTS)}", flush=True)
    return passed, cosines


def main():
    print(f"=== P21.B Embedding fine-tune (direct training loop) ===", flush=True)
    print(f"Loading pairs from {PAIRS}...", flush=True)
    if not PAIRS.exists():
        print(f"ERROR: {PAIRS} not found.", flush=True)
        sys.exit(1)

    rows = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(SEED)
    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRAC)
    val = rows[:n_val]
    train = rows[n_val:]
    print(f"  total: {len(rows)}, train: {len(train)}, val: {len(val)}", flush=True)

    train_examples = [InputExample(texts=[r["query"], r["passage"]]) for r in train]

    print(f"\nGPU check:", flush=True)
    print(f"  CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  VRAM: {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model: {BASE_MODEL}", flush=True)
    model = SentenceTransformer(BASE_MODEL, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    print(f"  Loaded (dim={model.get_sentence_embedding_dimension()}, max_seq_len={model.max_seq_length})", flush=True)

    # BASELINE smoke
    baseline_score, baseline_cos = run_smoke(model, "BASELINE BGE-M3 base")
    np.save(BASELINE_NPY, np.array(baseline_cos, dtype=np.float32))

    # Setup direct training
    print(f"\nSetup training: {EPOCHS} epochs, batch={BATCH} (×{GRAD_ACCUM_STEPS} grad_accum = {BATCH*GRAD_ACCUM_STEPS} effective), lr={LR}", flush=True)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH,
        collate_fn=model.smart_batching_collate,
        num_workers=0,  # P21.B fix: disable multiprocessing (Windows fork issues)
    )

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = max(50, int(total_steps * WARMUP_FRAC))
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}", flush=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)),
    )

    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> STARTING training loop <<<", flush=True)
    global_step = 0
    accumulated_loss = 0.0
    model.train()
    train_loss.train()

    for epoch in range(EPOCHS):
        epoch_losses = []
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
        for step, batch in enumerate(progress):
            features, labels = batch
            # Move to device
            for f in features:
                for k in list(f.keys()):
                    f[k] = f[k].to(device)
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            try:
                loss = train_loss(features, labels)
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                accumulated_loss += float(loss.detach().item())

                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    epoch_losses.append(accumulated_loss)
                    progress.set_postfix(loss=f"{accumulated_loss:.4f}", step=global_step)
                    accumulated_loss = 0.0

            except torch.cuda.OutOfMemoryError as e:
                print(f"\nCUDA OOM at step {step}: {e}", flush=True)
                torch.cuda.empty_cache()
                continue

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"  Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, steps: {len(epoch_losses)}", flush=True)

    print(f">>> Training COMPLETED <<<", flush=True)

    # Save model
    print(f"\nSaving model to {OUT_MODEL}...", flush=True)
    model.save(str(OUT_MODEL))

    # POST-train smoke
    model.eval()
    post_score, post_cos = run_smoke(model, "FINE-TUNED v1")

    # Divergence
    print(f"\nBaseline → Fine-tuned cosines:", flush=True)
    for i, (q, p, expected) in enumerate(SMOKE_TESTS):
        delta = post_cos[i] - baseline_cos[i]
        marker = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "≈"
        print(f"  {marker} {expected:<5} {baseline_cos[i]:+.3f} → {post_cos[i]:+.3f} ({delta:+.3f})  {q[:40]}", flush=True)

    print(f"\n✓ Model saved → {OUT_MODEL}", flush=True)
    print(f"  Smoke: {baseline_score}/{len(SMOKE_TESTS)} → {post_score}/{len(SMOKE_TESTS)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
