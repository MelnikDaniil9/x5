from transformers import AutoModelForTokenClassification
from config import MODEL_NAME, LABEL2ID, ID2LABEL

def load_model():
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model
