import torch
from seqeval.metrics import f1_score

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
            label_ids = batch["labels"].cpu().numpy()

            for i, label in enumerate(label_ids):
                pred = logits[i].argmax(axis=-1)
                true_labels = [l for l in label if l != -100]
                true_preds = [p for (p, l) in zip(pred, label) if l != -100]

                labels.append([str(l) for l in true_labels])
                preds.append([str(p) for p in true_preds])

    return f1_score(labels, preds)


def save_model(model, path):
    model.save_pretrained(path)
    print(f"💾 Модель сохранена в {path}")
