"""English-only gate at embed/upsert."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

# Deterministic langdetect across processes.
DetectorFactory.seed = 0

_MIN_CHARS = 40


def detect_language(text: str) -> str | None:
    sample = (text or "").strip()
    if len(sample) < _MIN_CHARS:
        return None
    try:
        return detect(sample[:5000])
    except LangDetectException:
        return None


def is_english(text: str) -> bool:
    return detect_language(text) == "en"
