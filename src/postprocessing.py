"""
Скрипт для применения обученной модели на тестовых данных и формирования submission_result.csv.
Теперь добавлен постпроцессинг:
- удаляются сущности на пробелах/пунктуации
- объединяются пересекающиеся/смежные сущности одного типа
- корректно проставляются B-/I- метки
"""

import os
import re
import torch
import pandas as pd
from data_processing import preprocess_query, tokenizer
from model import NERModel, id2label


# Пути
DATA_DIR = "data"
INPUT_SUBMISSION = os.path.join(DATA_DIR, "submission.csv")   # исходный файл
OUTPUT_SUBMISSION = os.path.join(DATA_DIR, "submission_result.csv")  # предсказанный файл
CKPT_PATH = os.path.join("model_checkpoint", "ner_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 🔹 Вспомогательные
# =====================
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
SPACE_RE = re.compile(r"^\s+$")


def split_entity_into_bio_spans(text: str, start: int, end: int, ent_type: str):
    """Разбивает сущность на BIO-спаны по словам (как в train)."""
    chunk = text[start:end]
    spans = [(m.start() + start, m.end() + start) for m in WORD_RE.finditer(chunk)]
    if not spans:
        return [(start, end, f"B-{ent_type}")]
    bio = []
    for i, (s, e) in enumerate(spans):
        tag = "B" if i == 0 else "I"
        bio.append((s, e, f"{tag}-{ent_type}"))
    return bio


def clean_and_merge_entities(text, entities):
    """
    Постпроцессинг:
    - удаляем сущности на пробелах/пунктуации
    - объединяем пересекающиеся/смежные в одну
    - выставляем корректные B/I метки
    """
    cleaned = []
    for start, end, tag in entities:
        chunk = text[start:end]
        if not chunk.strip():  # пустота или пробелы
            continue
        cleaned.append([start, end, tag])

    cleaned.sort(key=lambda x: x[0])

    merged = []
    for ent in cleaned:
        if not merged:
            merged.append(ent)
            continue

        prev = merged[-1]
        prev_type = prev[2].split("-", 1)[-1]
        curr_type = ent[2].split("-", 1)[-1]

        if ent[0] <= prev[1] and prev_type == curr_type:
            # пересечение или смежные + одинаковый тип
            prev[1] = max(prev[1], ent[1])
        else:
            merged.append(ent)

    # пересобираем BIO
    final = []
    for i, (s, e, tag) in enumerate(merged):
        ent_type = tag.split("-", 1)[-1]
        if i == 0 or merged[i-1][2].split("-", 1)[-1] != ent_type:
            final.append((s, e, f"B-{ent_type}"))
        else:
            final.append((s, e, f"I-{ent_type}"))
    return final


# =====================
# 🔹 Загружаем модель
# =====================
model = NERModel().to(device)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# =====================
# 🔹 Загружаем submission.csv
# =====================
df = pd.read_csv(INPUT_SUBMISSION, sep=";")

# =====================
# 🔹 Предсказания
# =====================
new_annotations = []

for _, row in df.iterrows():
    idx = row.get("id", row.name)  # если id нет, берём индекс строки
    original_query = row.get("search_query", row.get("sample"))
    text = preprocess_query(original_query)

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) == 0:
        new_annotations.append([])
        continue

    attn_mask = [1] * len(input_ids)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    mask_tensor = torch.tensor([attn_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        pred_tag_idxs = model(input_ids_tensor, mask_tensor)[0]  # list[int]

    pred_tags = [id2label[tag_idx] for tag_idx in pred_tag_idxs]

    # === Восстановим word-level сущности ===
    entities_pred = []
    current = None

    for tag, (start_char, end_char) in zip(pred_tags, offsets):
        if start_char == end_char:
            continue
        if tag.startswith("B-"):
            if current is not None:
                entities_pred.extend(
                    split_entity_into_bio_spans(text, current[0], current[1], current[2])
                )
            ent_type = tag.split("-", 1)[1]
            current = [start_char, end_char, ent_type]
        elif tag.startswith("I-"):
            ent_type = tag.split("-", 1)[1]
            if current is not None and current[2] == ent_type:
                current[1] = end_char
            else:
                current = [start_char, end_char, ent_type]
        else:
            if current is not None:
                entities_pred.extend(
                    split_entity_into_bio_spans(text, current[0], current[1], current[2])
                )
                current = None

    if current is not None:
        entities_pred.extend(
            split_entity_into_bio_spans(text, current[0], current[1], current[2])
        )

    # ✅ Постпроцессинг
    entities_pred = clean_and_merge_entities(text, entities_pred)

    new_annotations.append(entities_pred)

# Записываем обратно
df["annotation"] = new_annotations
df.to_csv(OUTPUT_SUBMISSION, sep=";", index=False)

print(f"✅ Saved submission with postprocessed predictions to {OUTPUT_SUBMISSION}")
