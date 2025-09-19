#train_pipeline.py
"""
Основной скрипт обучения модели.
Шаги:
1. Загрузка и подготовка данных (с разбиением на train/val).
2. Сохранение/загрузка закодированных данных (ускорение повторных запусков).
3. Инициализация модели и оптимизатора.
4. Цикл обучения по эпохам: тренировка и оценка на валидации.
5. Сохранение обученной модели.
"""
import os
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils import compute_metrics

from data_processing import (
    load_train_data,
    preprocess_query,
    tokenize_and_align_labels,
    augment_sample,
    collate_fn,
)
from model import NERModel, id2label

# === Отключаем ворнинги HuggingFace ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Гиперпараметры ===
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.02
TRAIN_PATH = "data/train.csv"
CACHE_DIR = "cache"
CKPT_DIR = "model_checkpoint"
CKPT_PATH = os.path.join(CKPT_DIR, "ner_model.pth")

# Фикс сидов
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# === 1. Подготовка данных с кешированием ===
def prepare_encodings(dataset, with_augment=False):
    enc = {"input_ids": [], "attention_mask": [], "labels": []}
    for query, entities in dataset:
        text = preprocess_query(query)
        input_ids, labels, _ = tokenize_and_align_labels(text, entities)
        attn_mask = [1] * len(input_ids)

        enc["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
        enc["attention_mask"].append(torch.tensor(attn_mask, dtype=torch.long))
        enc["labels"].append(torch.tensor(labels, dtype=torch.long))

        if with_augment and random.random() < 0.3:
            aug_text, aug_entities = augment_sample(text, entities)
            aug_input_ids, aug_labels, _ = tokenize_and_align_labels(aug_text, aug_entities)
            attn_mask_aug = [1] * len(aug_input_ids)

            enc["input_ids"].append(torch.tensor(aug_input_ids, dtype=torch.long))
            enc["attention_mask"].append(torch.tensor(attn_mask_aug, dtype=torch.long))
            enc["labels"].append(torch.tensor(aug_labels, dtype=torch.long))
    return enc


if os.path.exists(os.path.join(CACHE_DIR, "train_encodings.pt")) and os.path.exists(
    os.path.join(CACHE_DIR, "val_encodings.pt")
):
    print("✅ Загружаю готовые закодированные данные из cache/")
    train_encodings = torch.load(os.path.join(CACHE_DIR, "train_encodings.pt"))
    val_encodings = torch.load(os.path.join(CACHE_DIR, "val_encodings.pt"))
else:
    print("⚡ Токенизирую и сохраняю данные (это займёт время только 1 раз)...")
    data = load_train_data(TRAIN_PATH)
    random.shuffle(data)
    split_index = math.floor(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    train_encodings = prepare_encodings(train_data, with_augment=True)
    val_encodings = prepare_encodings(val_data, with_augment=False)

    torch.save(train_encodings, os.path.join(CACHE_DIR, "train_encodings.pt"))
    torch.save(val_encodings, os.path.join(CACHE_DIR, "val_encodings.pt"))
    print("✅ Данные сохранены в cache/")


# === 2. Dataset-обёртка ===
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.enc = encodings

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.enc["labels"][idx],
        }


train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
)

# === 3. Модель + оптимизатор + AMP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NERModel().to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = GradScaler("cuda")

best_macro_f1 = -1.0

# === 4. Цикл обучения ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=True):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            loss = model(input_ids, attention_mask, labels=labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))

    # --- Валидация ---
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_true = batch["labels"].to(device)

            pred_tag_idxs_batch = model(input_ids, attention_mask)

            attn = attention_mask.cpu().numpy()
            true_cpu = labels_true.cpu().numpy()
            for i, pred_seq in enumerate(pred_tag_idxs_batch):
                real_len = int(attn[i].sum())
                true_seq_ids = true_cpu[i][:real_len].tolist()
                pred_tags = [id2label[idx] for idx in pred_seq]
                true_tags = [id2label[idx] for idx in true_seq_ids]
                all_preds.append(pred_tags)
                all_true.append(true_tags)

    metrics = compute_metrics(all_preds, all_true)
    macro_f1 = metrics["macro_f1"]
    print(f"Epoch {epoch}/{EPOCHS} | Train loss: {avg_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  -> Saved best model to {CKPT_PATH} (macro_f1={best_macro_f1:.4f})")
