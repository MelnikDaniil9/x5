import os
import torch

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "ner_rubert_model")

# Модель и токенизация
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 128

# Обучение
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.03
WARMUP_RATIO = 0.2
SEED = 42

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = torch.cuda.is_available()  # RTX 4060 Ti → да

# BIO-метки
LABELS = [
    "O", 
    "B-BRAND", "I-BRAND",
    "B-TYPE", "I-TYPE",
    "B-VOLUME", "I-VOLUME",
    "B-PACKAGE", "I-PACKAGE"
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
