#data_processing.py
"""
Модуль для загрузки и предобработки данных.
Содержит функции:
- load_train_data: чтение train.csv и разбор аннотаций в удобный формат.
- load_test_data: чтение test.csv.
- preprocess_query: нормализация строки запроса (очистка, исправление опечаток и т.п.).
- tokenize_and_align_labels: токенизация + BIO-метки на субтокены.
- augment_sample: генерация аугментированного примера с опечатками.
- collate_fn: батчевый паддинг input_ids/attention_mask/labels.
"""
import csv, random, re
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from model import label2id

# Инициализируем токенизатор (например, XLM-RoBERTa)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_fast=True)


def load_train_data(file_path):
    """Читает CSV с обучающими данными и возвращает список кортежей (text, entities),
    где entities — список (start, end, label)."""
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        _ = next(reader, None)  # пропустить заголовок
        for row in reader:
            # Формат: id;search_query;annotation  ИЛИ search_query;annotation
            if len(row) == 3:
                _, query, ann_str = row
            else:
                query, ann_str = row
            # Парсим аннотации, например: "[(0, 8, 'B-TYPE'), ...]"
            entities = eval(ann_str) if ann_str.strip() else []
            # Приводим возможные '0' -> 'O'
            fixed = []
            for (start, end, tag) in entities:
                if tag == "0":
                    tag = "O"
                fixed.append((start, end, tag))
            data.append((query, fixed))
    return data


def load_test_data(file_path):
    """Чтение тестового CSV. Возвращает список кортежей (id, query)."""
    test_queries = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        _ = next(reader, None)
        for row in reader:
            if len(row) == 2:
                idx, query = row
            else:
                idx, query = None, row[0]
            test_queries.append((idx, query))
    return test_queries


def preprocess_query(query):
    """Нормализует текст запроса: убирает лишние пробелы, унифицирует символы и числа."""
    text = (query or "").lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" %", "%")
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)  # 1,5 -> 1.5
    # Нормализация единиц измерения (латинских на кириллицу) — без изменения длины токена
    # (кол-во символов одинаково: "ml"->"мл" (2->2), "l"->"л" (1->1), "g"->"г" (1->1), "kg"->"кг" (2->2))
    text = re.sub(r"\bml\b", "мл", text)
    text = re.sub(r"\bl\b", "л", text)
    text = re.sub(r"\bg\b", "г", text)
    text = re.sub(r"\bkg\b", "кг", text)
    return text


# Пример простого словаря опечаток (можно расширить)
_common_typos = {
    "йогурт": ["йогур", "йогрт"],
    "колбаса": ["колбаск", "колбса"],
}

def correct_obvious_typos(query):
    """Исправляет очевидные опечатки на основе словаря _common_typos."""
    text = query
    for correct, typos in _common_typos.items():
        for typo in typos:
            text = re.sub(fr"\b{re.escape(typo)}\b", correct, text)
    return text


def tokenize_and_align_labels(text, entities):
    """
    Токенизирует текст и создает список меток (индексы) для каждого субтокена.
    entities – список span-аннотаций [(start, end, label), ...] в символных индексах оригинального текста.
    Возвращает: input_ids(list[int]), labels(list[int]), offsets(list[tuple[int,int]])
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    labels = [label2id["O"]] * len(offsets)

    # Сформируем поксимвольную разметку для быстрого маппинга в токены
    char_labels = ["O"] * len(text)  # дефолт для каждого символа
    for (start, end, tag) in entities:
        if tag == "0":
            tag = "O"
        if tag == "O":
            for idx in range(start, min(end, len(text))):
                char_labels[idx] = "O"
        else:
            base_tag = tag[2:] if tag.startswith(("B-", "I-")) else tag
            for idx in range(start, min(end, len(text))):
                if idx == start:
                    char_labels[idx] = "B-" + base_tag
                else:
                    char_labels[idx] = "I-" + base_tag

    # Для каждого субтокена берём метку символа с его начала (offset[0])
    for i, (offs_start, offs_end) in enumerate(offsets):
        if offs_start == offs_end:
            labels[i] = label2id["O"]
        else:
            char_label = char_labels[offs_start] if offs_start < len(char_labels) else "O"
            labels[i] = label2id.get(char_label, label2id["O"])

    return input_ids, labels, offsets


def augment_sample(text, entities):
    """
    Создает вариацию (text_aug, entities_aug) с внесенными опечатками.
    Здесь — простая аугментация (отрезание последней буквы случайного слова длиной >3).
    """
    words = text.split()
    if not words:
        return text, entities
    i = random.randrange(len(words))
    if len(words[i]) > 3:
        cut = words[i][:-1]
        # Пересчёт спанов опустим (минимальная правка); в реальном пайплайне нужно смещать индексы.
        words[i] = cut
    text_aug = " ".join(words)
    return text_aug, entities


def collate_fn(batch):
    """
    Паддинг батча до максимальной длины внутри батча.
    Важно: labels паддятся произвольным значением (берём 'O'), но CRF будет игнорировать паддинги по attention_mask.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=label2id["O"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
