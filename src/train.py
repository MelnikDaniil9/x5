import math
import os
import time
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import precision_score, recall_score, f1_score

from config import (
    DATA_PATH, OUTPUT_DIR, DEVICE, MIXED_PRECISION,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    ID2LABEL, SEED,
)
from dataset import load_splits, make_loaders
from model import load_model
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ids_to_tags(batch_ids, batch_label_ids):
    """
    Преобразуем батч предсказаний и истинных меток [-100, id...] -> списки BIO-строк,
    игнорируя позиции с -100.
    """
    all_preds, all_labels = [], []
    for preds, labels in zip(batch_ids, batch_label_ids):
        p_seq, l_seq = [], []
        for p, l in zip(preds, labels):
            if l == -100:  # спецтокены/паддинг
                continue
            p_seq.append(ID2LABEL[int(p)])
            l_seq.append(ID2LABEL[int(l)])
        all_preds.append(p_seq)
        all_labels.append(l_seq)
    return all_preds, all_labels

def evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            pred_ids = logits.argmax(-1).detach().cpu().tolist()
            label_ids = batch["labels"].detach().cpu().tolist()
            p, l = ids_to_tags(pred_ids, label_ids)
            all_preds.extend(p)
            all_labels.extend(l)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    return prec, rec, f1

def train():
    set_seed(SEED)

    print(f"Loading data from {DATA_PATH}")
    train_ds, val_ds, collator = load_splits(DATA_PATH, val_size=0.1)
    train_loader, val_loader = make_loaders(train_ds, val_ds, collator, BATCH_SIZE)

    model = load_model().to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=MIXED_PRECISION)

    best_f1 = -1.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}, mixed_precision={MIXED_PRECISION}")
    print(f"Steps per epoch: {len(train_loader)}, total: {total_steps}, warmup: {warmup_steps}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=MIXED_PRECISION):
                out = model(**batch)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, len(train_loader))
        prec, rec, f1 = evaluate(model, val_loader)

        print(f"[Epoch {epoch}] loss={avg_loss:.4f}  val: P={prec:.4f} R={rec:.4f} F1={f1:.4f}  time={dt:.1f}s")

        # save best
        if f1 > best_f1:
            best_f1 = f1
            print(f"↳ New best F1={best_f1:.4f}. Saving to {OUTPUT_DIR}")
            model.save_pretrained(OUTPUT_DIR)
            # сохраняем и токенизатор рядом с моделью
            from dataset import tokenizer
            tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training done. Best F1={best_f1:.4f}")

if __name__ == "__main__":
    train()
