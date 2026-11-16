"""Подготовка текста к озвучке. На экран/клиенту уходит оригинал, в Silero — результат normalize_for_tts."""
import json
import logging
import re
from pathlib import Path

from num2words import num2words

logger = logging.getLogger(__name__)

PRONOUNCE_FILE = Path("data/pronounce.json")

# Грубая транслитерация: Silero латиницу не читает, а молча выкидывать слова жалко
_TRANSLIT = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "й", "z": "з",
}


def _load_pronounce() -> list[tuple[re.Pattern, str]]:
    if not PRONOUNCE_FILE.exists():
        logger.warning("pronounce dictionary %s not found", PRONOUNCE_FILE)
        return []
    words = json.loads(PRONOUNCE_FILE.read_text(encoding="utf-8"))
    # Целые слова, без учёта регистра: "тест" не заденет "тесто"
    return [(re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), dst) for src, dst in words.items()]


# Словарь читаем один раз при старте — правишь файл, перезапускаешь сервер
_PRONOUNCE = _load_pronounce()


def _number(value: str) -> str:
    try:
        if "." in value or "," in value:
            return num2words(float(value.replace(",", ".")), lang="ru")
        return num2words(int(value), lang="ru")
    except Exception:
        return value


def normalize_for_tts(text: str) -> str:
    t = text

    # Время 15:30 -> "пятнадцать тридцать"
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b",
               lambda m: f"{_number(m.group(1))} {_number(str(int(m.group(2))))}", t)

    # Знаки и единицы
    t = re.sub(r"(?<![\w\d])[-−–]\s?(?=\d)", "минус ", t)
    t = re.sub(r"(?<![\w\d])\+\s?(?=\d)", "плюс ", t)
    t = t.replace("°C", " градусов").replace("°С", " градусов").replace("°", " градусов")
    t = t.replace("%", " процентов").replace("₽", " рублей").replace("№", " номер ")

    # Числа -> слова
    t = re.sub(r"\d+[.,]\d+|\d+", lambda m: _number(m.group(0)), t)

    # Словарь произношений (ударения через "+")
    for pattern, replacement in _PRONOUNCE:
        t = pattern.sub(replacement, t)

    # Латиница -> кириллица
    t = re.sub(r"[A-Za-z]", lambda m: _TRANSLIT[m.group(0).lower()], t)

    # Всё, что модель не умеет произносить
    t = re.sub(r"[^А-Яа-яЁё+\s.,!?:;—\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_letters(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_for_tts(text: str, limit: int = 800) -> list[str]:
    """Режет текст на куски по границам предложений: у Silero есть предел длины одного вызова."""
    parts, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts
