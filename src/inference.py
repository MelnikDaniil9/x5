import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from config import OUTPUT_DIR, ID2LABEL

device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем сохранённую модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)
model.to(device).eval()

def merge_entities(text, offsets, pred_ids):
    """
    Склеиваем последовательные B-/I- токены в цельные сущности.
    """
    entities = []
    current = None

    for (start, end), pred_id in zip(offsets, pred_ids):
        if start == end:  # спецтокены ([CLS], [SEP])
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
            current = {
                "entity": ent_type,
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        elif tag == "I" and current and current["entity"] == ent_type:
            # продолжаем ту же сущность
            current["end"] = end
            current["text"] = text[current["start"]:end]
        else:
            # странная ситуация → начинаем новую сущность
            if current:
                entities.append(current)
            current = {
                "entity": ent_type,
                "start": start,
                "end": end,
                "text": text[start:end],
            }

    if current:
        entities.append(current)

    return entities

def predict(text: str):
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=64)
    offsets = enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
    pred_ids = out.logits.argmax(-1).squeeze().cpu().tolist()

    entities = merge_entities(text, offsets.squeeze().tolist(), pred_ids)
    return entities

if __name__ == "__main__":
    examples = [
        "кола 2л без сахара",
        "lays чипсы 150г",
        "global village сок 1л",
    ]
    for ex in examples:
        print(f"\nText: {ex}")
        print(predict(ex))
