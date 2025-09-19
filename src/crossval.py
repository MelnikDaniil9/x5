import pandas as pd
from sklearn.model_selection import KFold
import torch

from train import train_one_fold

N_FOLDS = 5
SEED = 42


def run_crossval(data_path="data/train.csv"):
    df = pd.read_csv(data_path, sep=";")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        f1 = train_one_fold(train_df, val_df, fold_id=fold+1)
        fold_scores.append(f1)

        torch.cuda.empty_cache()

    print("\n===== Cross-validation results =====")
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i}: F1={score:.4f}")
    print(f"Средний F1 по {N_FOLDS} фолдам: {sum(fold_scores)/len(fold_scores):.4f}")


if __name__ == "__main__":
    run_crossval()
