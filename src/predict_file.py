import pandas as pd
from inference import predict

def predict_file(input_path: str, output_path: str):
    # читаем входной файл
    df = pd.read_csv(input_path, sep=";")

    new_annotations = []
    for text in df["sample"]:
        entities = predict(text)

        # формат, как в train.csv: список (start, end, tag)
        annots = []
        for ent in entities:
            tag = f"B-{ent['entity']}"
            annots.append((ent["start"], ent["end"], tag))

        new_annotations.append(annots)

    # перезаписываем колонку
    df["annotation"] = new_annotations

    # сохраняем результат
    df.to_csv(output_path, sep=";", index=False)
    print(f"✅ Предсказания сохранены в {output_path}")

if __name__ == "__main__":
    predict_file("data/submission.csv", "data/submission_filled.csv")
