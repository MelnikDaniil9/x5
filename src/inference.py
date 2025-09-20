"""
Inference script for NER model.

Режим по умолчанию: обрабатывает data/submission.csv и сохраняет
data/submission_result.csv с перезаписанным столбцом annotation.
Все функции постпроцессинга вынесены в postprocess.py.
"""

import os
import torch
import pandas as pd
from typing import List, Tuple

from .data_processing import preprocess_query, tokenizer
from .model import NERModel, id2label
from .postprocess import decode_to_entities  # ← весь постпроцессинг здесь

# Пути
DATA_DIR = "data"
INPUT_SUBMISSION = os.path.join(DATA_DIR, "submission.csv")          # исходный файл (столбец sample или search_query)
OUTPUT_SUBMISSION = os.path.join(DATA_DIR, "submission_result.csv")  # результат (search_query + новая annotation)
CKPT_PATH = os.path.join("model_checkpoint", "ner_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 🔹 Модель (один раз на модуль)
# =====================
def _load_model() -> NERModel:
    model = NERModel().to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


MODEL: NERModel = _load_model()


# =====================
# 🔹 Инференс одного текста
# =====================
def run_inference_on_text(query: str) -> List[Tuple[int, int, str]]:
    """
    Обработка одного текстового запроса → список (start, end, 'B-/I-<TYPE>') на word-level.
    """
    # Текст для модели нормализуем, но в offsets остаются индексы нормализованного текста.
    text_for_model = preprocess_query(query)

    enc = tokenizer(text_for_model, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if not input_ids:
        return []

    attn_mask = [1] * len(input_ids)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    mask_tensor = torch.tensor([attn_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        pred_tag_idxs = MODEL(input_ids_tensor, mask_tensor)[0]  # list[int]

    pred_tags = [id2label[tag_idx] for tag_idx in pred_tag_idxs]

    # Построение финальных сущностей (word-level BIO) по offsets
    final_entities = decode_to_entities(text_for_model, pred_tags, offsets)
    return final_entities


# =====================
# 🔹 Batch-режим: submission.csv → submission_result.csv
# =====================
def run_debug_mode(
    input_csv: str = INPUT_SUBMISSION,
    output_csv: str = OUTPUT_SUBMISSION
) -> None:
    """
    Обрабатывает CSV с колонками:
      - 'id' (опционально)
      - 'sample' или 'search_query' (обязательно одна из них)

    Перезаписывает колонку 'annotation' предсказанными сущностями (word-level BIO).
    """
    df = pd.read_csv(input_csv, sep=";")

    # Унифицируем колонки: sample → search_query
    if "search_query" not in df.columns and "sample" in df.columns:
        df = df.rename(columns={"sample": "search_query"})

    if "search_query" not in df.columns:
        raise ValueError("Входной CSV должен содержать столбец 'search_query' или 'sample'.")

    new_annotations = []
    for _, row in df.iterrows():
        query = row["search_query"]
        entities = run_inference_on_text(query)
        new_annotations.append(entities)

    df["annotation"] = new_annotations
    cols = ["id", "search_query", "annotation"] if "id" in df.columns else ["search_query", "annotation"]
    df.to_csv(output_csv, columns=cols, sep=";", index=False)
    print(f"✅ Saved submission to {output_csv}")

if __name__ == "__main__":
    run_debug_mode(INPUT_SUBMISSION, OUTPUT_SUBMISSION)
