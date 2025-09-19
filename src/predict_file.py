import re
import pandas as pd
from inference import predict

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")

def split_entity_into_bio_spans(text: str, start: int, end: int, ent_type: str):
    chunk = text[start:end]
    spans = [(m.start() + start, m.end() + start) for m in WORD_RE.finditer(chunk)]
    if not spans:
        return [(start, end, f"B-{ent_type}")]
    bio = []
    for i, (s, e) in enumerate(spans):
        tag = "B" if i == 0 else "I"
        bio.append((s, e, f"{tag}-{ent_type}"))
    return bio

def predict_file(input_path: str, output_path: str):
    df = pd.read_csv(input_path, sep=";")
    new_annotations = []
    for text in df["sample"]:
        entities = predict(text)
        spans = []
        for ent in entities:
            spans.extend(split_entity_into_bio_spans(text, ent["start"], ent["end"], ent["entity"]))
        new_annotations.append(spans)
    df["annotation"] = new_annotations
    df.to_csv(output_path, sep=";", index=False)
    print(f"✅ Предсказания ансамбля сохранены в {output_path}")

if __name__ == "__main__":
    predict_file("data/submission.csv", "data/submission_filled.csv")
