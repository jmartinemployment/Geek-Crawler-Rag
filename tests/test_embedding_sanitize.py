from __future__ import annotations

from geek_crawler_rag.embedding_sanitize import (
    sanitize_embedding_text,
    sanitize_embedding_texts,
)


def test_strips_null_bytes_and_controls():
    raw = "hello\x00world\x07\tkeep\nme"
    assert sanitize_embedding_text(raw) == "helloworld\tkeep\nme"


def test_strips_zero_width_and_bom():
    raw = "a\u200b\u200c\u200d\ufeff\u00adb"
    assert sanitize_embedding_text(raw) == "ab"


def test_replaces_lone_surrogates():
    raw = "bad\ud800chunk"
    cleaned = sanitize_embedding_text(raw)
    assert "\ud800" not in cleaned
    assert "bad" in cleaned and "chunk" in cleaned


def test_batch_reports_mutations():
    texts = ["clean", "dirty\x00", "also\u200bclean"]
    cleaned, mutated = sanitize_embedding_texts(texts)
    assert mutated == 2
    assert cleaned == ["clean", "dirty", "alsoclean"]


def test_preserves_meaningful_prose_and_html():
    # Sanitizer is encoding hygiene only — does not strip HTML or rewrite text.
    raw = "<p>Partner API docs explain OAuth thoroughly.</p>\nLine two"
    assert sanitize_embedding_text(raw) == raw


def test_none_and_empty():
    assert sanitize_embedding_text(None) == ""
    assert sanitize_embedding_text("") == ""
