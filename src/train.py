import os

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, get_scheduler

from dataset import load_splits_from_df
from utils import evaluate_model, save_model

from config import BATCH_SIZE, MODEL_NAME, DEVICE, LEARNING_RATE, EPOCHS, WARMUP_RATIO, LABELS


def train_one_fold(train_df, val_df, fold_id=None, output_dir="outputs"):
    print(f"Starting training on fold {fold_id}...")

    # Загружаем датасет из DataFrame
    train_ds, val_ds, collator = load_splits_from_df(train_df, val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            out = model(**batch)
            loss = out.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1 = evaluate_model(model, val_loader, DEVICE)

        print(f"[Fold {fold_id}] Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.4f} | Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(output_dir, f"ner_rubert_fold{fold_id}")
            save_model(model, save_path)

    return best_f1


if __name__ == "__main__":
    print("⚠️ Используй crossval.py для запуска кросс-валидации.")
