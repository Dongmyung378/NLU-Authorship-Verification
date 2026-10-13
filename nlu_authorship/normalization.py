"""Shared text cleanup and conservative document identity rules."""
from __future__ import annotations

import re
import unicodedata

MAX_TEXT_CHARACTERS = 1_000_000


def guard_text(text: str, *, allow_empty: bool = False) -> str:
    if not isinstance(text, str):
        raise TypeError("Each passage must be a string")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError("Passage exceeds the supported character limit")
    if "\x00" in text:
        raise ValueError("NUL characters are not supported")
    value = unicodedata.normalize("NFC", text)
    if not allow_empty and not value.strip():
        raise ValueError("Each passage must contain text")
    return value


def _legacy_stylometric(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.]+\.[a-z]{2,}", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", " ", text)
    text = re.sub(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _legacy_roberta(text: str) -> str:
    text = re.sub(r"(From|To|Cc|Subject|Date|Forwarded by)[^\n]*\n", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_roberta_text(text: str) -> str:
    result = _legacy_roberta(guard_text(text))
    if not result:
        raise ValueError("Passage is empty after RoBERTa cleanup")
    return result


def normalize_stylometric_text(text: str) -> str:
    result = _legacy_stylometric(guard_text(text))
    if not result:
        raise ValueError("Passage is empty after stylometric cleanup")
    return result


def identity_views(text: str) -> dict[str, str]:
    # Exact legacy views retain the historical regex behavior. NFC-whitespace
    # additionally links canonical Unicode representations without case folding.
    canonical = guard_text(text, allow_empty=True)
    return {"exact": text,
            "nfc_whitespace": re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip(),
            "legacy_stylometric": _legacy_stylometric(text),
            "legacy_roberta": _legacy_roberta(text),
            "runtime_stylometric": _legacy_stylometric(canonical),
            "runtime_roberta": _legacy_roberta(canonical)}


def minimum_word_count(text_1: str, text_2: str) -> int:
    return min(len(normalize_roberta_text(text_1).split()), len(normalize_roberta_text(text_2).split()))


def length_bin(text_1: str, text_2: str) -> str:
    count = minimum_word_count(text_1, text_2)
    return "under_50" if count < 50 else "50_to_199" if count < 200 else "200_plus"
