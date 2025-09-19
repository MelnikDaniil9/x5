#model.py
"""
Модуль с определением модели NER.
Содержит:
- Словарь меток label2id и id2label.
- Класс NERModel на основе трансформера и CRF.
"""

from transformers import XLMRobertaModel
import torch.nn as nn
from torchcrf import CRF

# Определяем все возможные метки BIO
labels = [
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT"
]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
num_labels = len(labels)


class NERModel(nn.Module):
    """
    Архитектура: XLM-RoBERTa (encoder) -> Dropout -> Linear(num_labels) -> CRF.
    В режиме обучения forward возвращает loss (тензор).
    В режиме инференса forward возвращает декодированные последовательности меток (списки индексов).
    """
    def __init__(self):
        super().__init__()
        self.transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base") #Более тяжелая модель xlm-roberta-large
        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # CRF слой для последовательного декодирования
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        :param input_ids: LongTensor [batch, seq_len]
        :param attention_mask: LongTensor [batch, seq_len] (1 для реальных токенов, 0 для паддинга)
        :param labels: LongTensor [batch, seq_len] (опционально, только при обучении)
        :return: loss (тензор) при обучении или списки списков меток (индексы) при инференсе
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, T, H]
        emissions = self.classifier(sequence_output)               # [B, T, C]

        mask_bool = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # CRF возвращает log_likelihood (мы берём отрицательный)
            log_likelihood = self.crf(emissions, labels, mask=mask_bool, reduction="mean")
            loss = -log_likelihood
            return loss
        else:
            # Декодируем последовательности меток (Viterbi)
            best_paths = self.crf.decode(emissions, mask=mask_bool)
            return best_paths
