import os
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from dataset import ID2LABEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIRS = [
    "outputs/ner_rubert_fold1",
    "outputs/ner_rubert_fold2",
    "outputs/ner_rubert_fold3",
    "outputs/ner_rubert_fold4",
    "outputs/ner_rubert_fold5",
]

models = []
tokenizer = None
for path in MODEL_DIRS:
    if os.path.exists(path):
        print(f"Загружаю модель {path}")
        model = BertForTokenClassification.from_pretrained(path).to(DEVICE)
        model.eval()
        models.append(model)
        if tokenizer is None:
            tokenizer = BertTokenizerFast.from_pretrained(path)

if not models:
    raise ValueError("❌ Не найдено ни одной модели в MODEL_DIRS")
print(f"✅ Загружено {len(models)} моделей")

def predict(text):
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    all_logits = []
    with torch.no_grad():
        for model in models:
            out = model(**enc)
            all_logits.append(out.logits.cpu().numpy())

    avg_logits = np.mean(all_logits, axis=0)
    pred_ids = avg_logits.argmax(axis=-1)[0]

    offsets = enc["offset_mapping"][0].cpu().numpy()
    entities = merge_entities(text, offsets, pred_ids)
    return entities

def merge_entities(text, offsets, pred_ids):
    entities, current = [], None
    for (start, end), pred_id in zip(offsets, pred_ids):
        if start == end:
            continue
        label = ID2LABEL[pred_id]
        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue
        tag, ent_type = label.split("-", 1)
        if tag == "B":
            if current:
                entities.append(current)
            current = {"entity": ent_type, "start": start, "end": end, "text": text[start:end]}
        elif tag == "I":
            if current and current["entity"] == ent_type and start >= current["end"]:
                current["end"] = end
                current["text"] = text[current["start"]:end]
            else:
                if current:
                    entities.append(current)
                current = {"entity": ent_type, "start": start, "end": end, "text": text[start:end]}
    if current:
        entities.append(current)
    return entities
