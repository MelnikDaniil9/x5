"""
postprocess.py
Набор функций постпроцессинга для конвертации subtoken-предсказаний модели
в word-level BIO-спаны и их очистки/нормализации.
"""

import re
from typing import List, Tuple

# Типы для ясности
Span = Tuple[int, int, str]  # (start, end, "B-TYPE"/"I-TYPE"/...)
RawSpan = Tuple[int, int, str]  # (start, end, "B-TYPE"/"I-TYPE"/"TYPE")

# Регулярки и хелперы
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")   # слово = буквы/цифры (RU/EN)
NON_WORD_RE = re.compile(r"^\W+$", re.UNICODE)  # только пробелы/пунктуация


def split_entity_into_bio_spans(text: str, start: int, end: int, ent_type: str) -> List[Span]:
    """
    Разбивает (start,end) на word-level спаны по WORD_RE.
    Первый → B-тип, остальные → I-тип.
    Если внутри нет слов, вернёт [].
    """
    chunk = text[start:end]
    word_spans = [(m.start() + start, m.end() + start) for m in WORD_RE.finditer(chunk)]
    if not word_spans:
        return []
    spans: List[Span] = []
    for i, (s, e) in enumerate(word_spans):
        tag = "B" if i == 0 else "I"
        spans.append((s, e, f"{tag}-{ent_type}"))
    return spans


def merge_overlapping_spans(spans: List[Span]) -> List[Span]:
    """
    Сливает пересекающиеся/вложенные спаны одного типа.
    На вход ожидаются уже word-level BIO-спаны, но суффикс типа берём из тега.
    Возвращает спаны с B-<TYPE> тегом (последующий этап заново расставит B/I).
    """
    if not spans:
        return []

    # Приведём к (s,e,type) для слияния
    triples = []
    for s, e, tag in spans:
        ent_type = tag.split("-", 1)[-1]
        triples.append((s, e, ent_type))

    triples.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s, e, t in triples:
        if not merged:
            merged.append([s, e, t])
            continue

        ls, le, lt = merged[-1]
        # вложение или пересечение одного типа → объединяем
        if t == lt and s <= le:
            merged[-1][1] = max(le, e)
        else:
            merged.append([s, e, t])

    return [(s, e, f"B-{t}") for s, e, t in merged]


def build_raw_entities_from_tags(
    text: str,
    pred_tags: List[str],
    offsets: List[Tuple[int, int]]
) -> List[RawSpan]:
    """
    Собирает "сырые" спаны на уровне символов, склеивая последовательные B-/I-одного типа.
    Возвращает спаны в виде (start, end, 'B-<TYPE>').
    """
    raw_entities: List[RawSpan] = []
    current = None  # [start, end, ent_type]

    for tag, (start_char, end_char) in zip(pred_tags, offsets):
        if start_char == end_char:
            continue

        if tag.startswith("B-"):
            if current is not None:
                raw_entities.append((current[0], current[1], f"B-{current[2]}"))
                current = None
            ent_type = tag.split("-", 1)[1]
            current = [start_char, end_char, ent_type]

        elif tag.startswith("I-"):
            ent_type = tag.split("-", 1)[1]
            if current is not None and current[2] == ent_type:
                current[1] = end_char
            else:
                current = [start_char, end_char, ent_type]

        else:  # 'O'
            if current is not None:
                raw_entities.append((current[0], current[1], f"B-{current[2]}"))
                current = None

    if current is not None:
        raw_entities.append((current[0], current[1], f"B-{current[2]}"))

    return raw_entities


def postprocess_to_word_level_bio(text: str, raw_entities: List[RawSpan]) -> List[Span]:
    """
    Превращает subtoken-level сущности (сырые спаны) в word-level BIO с чисткой:
    1) Разбиваем каждый спан на word-level по WORD_RE
    2) Удаляем пустые/пунктуационные куски
    3) Сливаем пересекающиеся/смежные спаны одного типа
    4) Повторно расставляем корректные B/I между соседними словами одного типа
    """
    # 1) Разбиваем каждый спан на word-level
    word_level: List[Span] = []
    for s, e, tag in raw_entities:
        ent_type = tag.split("-", 1)[-1]
        word_level.extend(split_entity_into_bio_spans(text, s, e, ent_type))

    # 2) Фильтрация пустых и пунктуации
    filtered: List[Span] = []
    for s, e, tag in word_level:
        if s >= e:
            continue
        piece = text[s:e]
        if not piece or not piece.strip():
            continue
        if NON_WORD_RE.match(piece):
            continue
        filtered.append((s, e, tag))

    if not filtered:
        return []

    # 3) Слияние пересекающихся/вложенных
    merged = merge_overlapping_spans(filtered)

    # 4) Сортируем и корректируем BIO между словами
    merged.sort(key=lambda x: (x[0], x[1]))
    final: List[Span] = []
    prev_type = None
    for s, e, tag in merged:
        curr_type = tag.split("-", 1)[-1]
        if prev_type == curr_type:
            final.append((s, e, f"I-{curr_type}"))
        else:
            final.append((s, e, f"B-{curr_type}"))
        prev_type = curr_type

    return final


def decode_to_entities(text: str, pred_tags: List[str], offsets: List[Tuple[int, int]]) -> List[Span]:
    """
    Удобный конвейер: теги + offsets → сырые спаны → word-level BIO.
    """
    raw = build_raw_entities_from_tags(text, pred_tags, offsets)
    return postprocess_to_word_level_bio(text, raw)
