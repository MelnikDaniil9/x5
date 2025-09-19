"""
Скрипт для применения обученной модели на тестовых данных и формирования submission_result.csv.

Постпроцессинг делает:
- конвертацию subtoken → word-level спанов;
- удаление "сущностей" на пробелах/пунктуации;
- слияние вложенных/пересекающихся спанов;
- корректную расстановку BIO на последовательности слов одного типа.
"""

import os
import re
import torch
import pandas as pd
from data_processing import preprocess_query, tokenizer
from model import NERModel, id2label

# Пути
DATA_DIR = "data"
INPUT_SUBMISSION = os.path.join(DATA_DIR, "submission.csv")          # исходный файл (столбец sample)
OUTPUT_SUBMISSION = os.path.join(DATA_DIR, "submission_result.csv")  # результат (search_query + новая annotation)
CKPT_PATH = os.path.join("model_checkpoint", "ner_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 🔹 Регулярки и хелперы
# =====================
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")   # слово = буквы/цифры (RU/EN)
NON_WORD_RE = re.compile(r"^\W+$", re.UNICODE)  # только пробелы/пунктуация


def split_entity_into_bio_spans(text: str, start: int, end: int, ent_type: str):
    """
    Разбивает (start,end) на word-level спаны по WORD_RE.
    Первый → B-тип, остальные → I-тип.
    Если внутри нет слов, вернёт [].
    """
    chunk = text[start:end]
    word_spans = [(m.start() + start, m.end() + start) for m in WORD_RE.finditer(chunk)]
    if not word_spans:
        return []
    spans = []
    for i, (s, e) in enumerate(word_spans):
        tag = "B" if i == 0 else "I"
        spans.append((s, e, f"{tag}-{ent_type}"))
    return spans


def merge_overlapping_spans(spans):
    """
    Сливает пересекающиеся или вложенные спаны одного типа.
    Например: [(0,1,'B-TYPE'), (0,11,'I-TYPE')] → [(0,11,'B-TYPE')]
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = []
    for s, e, tag in spans:
        ent_type = tag.split("-", 1)[-1]
        if not merged:
            merged.append([s, e, ent_type])
            continue

        last_s, last_e, last_type = merged[-1]
        if ent_type == last_type and s >= last_s and e <= last_e:
            continue  # текущий внутри предыдущего
        if ent_type == last_type and s <= last_e:
            merged[-1][1] = max(last_e, e)  # расширяем
        else:
            merged.append([s, e, ent_type])

    return [(s, e, f"B-{t}") for s, e, t in merged]


def postprocess_to_word_level_bio(text: str, raw_entities):
    """
    Превращает subtoken-level сущности в word-level BIO с чисткой.
    """
    # 1) Разбиваем каждый спан на word-level
    word_level = []
    for s, e, tag in raw_entities:
        ent_type = tag.split("-", 1)[-1]
        word_level.extend(split_entity_into_bio_spans(text, s, e, ent_type))

    # 2) Фильтрация пустых и пунктуации
    filtered = []
    for s, e, tag in word_level:
        if s >= e:
            continue
        piece = text[s:e]
        if not piece or not piece.strip():
            continue
        if NON_WORD_RE.match(piece):
            continue
        filtered.append((s, e, tag))

    # 3) Слияние пересекающихся/вложенных
    merged = merge_overlapping_spans(filtered)

    # 4) Сортируем и корректируем BIO между словами
    merged.sort(key=lambda x: (x[0], x[1]))
    final = []
    prev_type = None
    for s, e, tag in merged:
        curr_type = tag.split("-", 1)[-1]
        if prev_type == curr_type:
            final.append((s, e, f"I-{curr_type}"))
        else:
            final.append((s, e, f"B-{curr_type}"))
        prev_type = curr_type

    return final


# =====================
# 🔹 Модель
# =====================
model = NERModel().to(device)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# =====================
# 🔹 Данные submission.csv
# =====================
df = pd.read_csv(INPUT_SUBMISSION, sep=";")

# Унифицируем колонки: sample → search_query
if "search_query" not in df.columns and "sample" in df.columns:
    df = df.rename(columns={"sample": "search_query"})

# =====================
# 🔹 Предсказания + постпроцессинг
# =====================
new_annotations = []

for _, row in df.iterrows():
    idx = row["id"] if "id" in df.columns else row.name
    original_query = row["search_query"]

    # Для токенизации используем предобработку; в файл пишем оригинальный текст
    text_for_model = preprocess_query(original_query)

    enc = tokenizer(text_for_model, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if not input_ids:
        new_annotations.append([])
        continue

    attn_mask = [1] * len(input_ids)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    mask_tensor = torch.tensor([attn_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        pred_tag_idxs = model(input_ids_tensor, mask_tensor)[0]

    pred_tags = [id2label[tag_idx] for tag_idx in pred_tag_idxs]

    # --- собираем "сырой" список сущностей ---
    raw_entities = []
    current = None
    for tag, (start_char, end_char) in zip(pred_tags, offsets):
        if start_char == end_char:
            continue
        if tag.startswith("B-"):
            if current is not None:
                raw_entities.append((current[0], current[1], f"B-{current[2]}"))
            ent_type = tag.split("-", 1)[1]
            current = [start_char, end_char, ent_type]
        elif tag.startswith("I-"):
            ent_type = tag.split("-", 1)[1]
            if current is not None and current[2] == ent_type:
                current[1] = end_char
            else:
                current = [start_char, end_char, ent_type]
        else:  # O
            if current is not None:
                raw_entities.append((current[0], current[1], f"B-{current[2]}"))
                current = None
    if current is not None:
        raw_entities.append((current[0], current[1], f"B-{current[2]}"))

    # --- постпроцессинг ---
    final_entities = postprocess_to_word_level_bio(text_for_model, raw_entities)
    new_annotations.append(final_entities)

# =====================
# 🔹 Сохранение
# =====================
df["annotation"] = new_annotations
cols = ["id", "search_query", "annotation"] if "id" in df.columns else ["search_query", "annotation"]
df.to_csv(OUTPUT_SUBMISSION, columns=cols, sep=";", index=False)

print(f"✅ Saved submission with postprocessed predictions to {OUTPUT_SUBMISSION}")
