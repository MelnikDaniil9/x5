import ast
import random
import pandas as pd
import torch

from typing import List, Dict, Tuple
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from config import MODEL_NAME, MAX_LEN, LABEL2ID, SEED

random.seed(SEED)

# Один общий токенизатор на проект
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def _encode_with_labels(sample: str, spans: List[Tuple[int, int, str]]) -> Dict[str, List[int]]:
    """
    Токенизируем строку и раскладываем BIO-метки по токенам по offset_mapping.
    """
    enc = tokenizer(
        sample,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=True,
    )
    offsets = enc["offset_mapping"]
    labels = ["O"] * len(offsets)

    for start, end, tag in spans:
        for i, (s, e) in enumerate(offsets):
            # спецтокены обычно (0,0) — пропускаем
            if s == e == 0:
                continue
            # нет пересечения
            if s >= end or e <= start:
                continue
            if tag == "O":
                labels[i] = "O"
            elif s == start:
                # принимаем исходную метку из спана (B-XXX или I-XXX)
                labels[i] = tag
            elif labels[i] == "O":
                ent_type = tag.split("-")[-1]  # TYPE/BRAND/VOLUME/PERCENT/...
                labels[i] = "I-" + ent_type

    # в числовые id; спецтокены, где offset (0,0), делаем -100
    label_ids = []
    for (s, e), lab in zip(offsets, labels):
        if s == e == 0:
            label_ids.append(-100)
        else:
            label_ids.append(LABEL2ID.get(lab, -100))

    enc.pop("offset_mapping")
    enc["labels"] = label_ids
    return enc

class NEREncodingsDataset(TorchDataset):
    def __init__(self, encodings: List[Dict[str, List[int]]]):
        self.data = encodings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.data[idx].items()}
        return item

def load_splits(csv_path: str, val_size: float = 0.1):
    """
    Читаем train.csv (sep=';'), парсим annotation, строим список encodings,
    руками делим на train/val.
    """
    df = pd.read_csv(csv_path, sep=";")
    assert {"sample", "annotation"}.issubset(df.columns), \
        f"Ожидались колонки 'sample' и 'annotation', получили: {df.columns}"

    df["annotation"] = df["annotation"].apply(ast.literal_eval)

    encs = [_encode_with_labels(s, a) for s, a in zip(df["sample"], df["annotation"])]

    # сплит
    n = len(encs)
    idx = list(range(n))
    random.shuffle(idx)
    cut = int(n * (1 - val_size))
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    train_encs = [encs[i] for i in train_idx]
    val_encs = [encs[i] for i in val_idx]

    train_ds = NEREncodingsDataset(train_encs)
    val_ds = NEREncodingsDataset(val_encs)

    collator = DataCollatorForTokenClassification(tokenizer)
    return train_ds, val_ds, collator

def make_loaders(train_ds, val_ds, collator, batch_size: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader

def load_splits_from_df(train_df, val_df):
    """
    Создаём PyTorch Dataset-ы из pandas.DataFrame (для кросс-валидации).
    НИКАКОГО HuggingFace datasets здесь не используем.
    """
    # если annotation хранится строкой -> превращаем в список
    if isinstance(train_df["annotation"].iloc[0], str):
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["annotation"] = train_df["annotation"].apply(ast.literal_eval)
        val_df["annotation"] = val_df["annotation"].apply(ast.literal_eval)

    train_encs = [_encode_with_labels(s, a) for s, a in zip(train_df["sample"], train_df["annotation"])]
    val_encs   = [_encode_with_labels(s, a) for s, a in zip(val_df["sample"],   val_df["annotation"])]

    train_ds = NEREncodingsDataset(train_encs)
    val_ds   = NEREncodingsDataset(val_encs)

    collator = DataCollatorForTokenClassification(tokenizer)
    return train_ds, val_ds, collator
