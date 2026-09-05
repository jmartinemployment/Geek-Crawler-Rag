"""Token-aware chunking (~500–800 tokens with small overlap)."""

from __future__ import annotations

import tiktoken

_ENCODING_NAME = "cl100k_base"


def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def chunk_text(
    text: str,
    *,
    size_tokens: int = 650,
    overlap_tokens: int = 80,
) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if size_tokens < 1:
        raise ValueError("size_tokens must be >= 1")
    if overlap_tokens < 0 or overlap_tokens >= size_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < size_tokens")

    enc = _encoding()
    tokens = enc.encode(cleaned)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = size_tokens - overlap_tokens
    while start < len(tokens):
        end = min(start + size_tokens, len(tokens))
        piece = enc.decode(tokens[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(tokens):
            break
        start += step
    return chunks
