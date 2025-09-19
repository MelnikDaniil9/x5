import os
import torch

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "ner_rubert_model")

# Модель и токенизация
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 64

# Обучение
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = torch.cuda.is_available()  # RTX 4060 Ti → да

# BIO-метки
LABEL_LIST = [
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
