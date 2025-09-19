#utils.py
"""
Вспомогательные функции, включая расчёт метрик.
Здесь — токен-уровневые метрики по каждой сущности и Macro-F1 (среднее по 4 типам).
"""
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(pred_tags_list, true_tags_list):
    """
    pred_tags_list и true_tags_list — списки последовательностей (список строк BIO) одинаковой длины для каждого примера.
    Валидация в train_pipeline уже подрезает длины по attention_mask.
    """
    entity_types = ["TYPE", "BRAND", "VOLUME", "PERCENT"]
    metrics = {}

    for ent in entity_types:
        true_binary = []
        pred_binary = []
        for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
            # сопоставляем токен-уровень
            for t, p in zip(true_tags, pred_tags):
                true_binary.append(1 if (t != "O" and t.endswith(ent)) else 0)
                pred_binary.append(1 if (p != "O" and p.endswith(ent)) else 0)
        p = precision_score(true_binary, pred_binary, zero_division=0)
        r = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        metrics[f"{ent}_precision"] = p
        metrics[f"{ent}_recall"] = r
        metrics[f"{ent}_f1"] = f1

    metrics["macro_f1"] = (
        metrics["TYPE_f1"] + metrics["BRAND_f1"] + metrics["VOLUME_f1"] + metrics["PERCENT_f1"]
    ) / 4.0
    return metrics
