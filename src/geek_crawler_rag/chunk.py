"""Token-aware chunking: legacy sliding window + parent/child sections."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ParentChildUnit:
    """One searchable child span plus its parent context section."""

    parent_text: str
    child_text: str
    section_title: str | None
    parent_index: int
    child_index: int


def _token_windows(
    text: str,
    *,
    size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    return chunk_text(text, size_tokens=size_tokens, overlap_tokens=overlap_tokens)


def parent_child_units(
    text: str,
    *,
    child_size_tokens: int = 200,
    child_overlap_tokens: int = 40,
    parent_size_tokens: int = 1000,
    parent_overlap_tokens: int = 80,
) -> list[ParentChildUnit]:
    """Build child pinpoint chunks nested under parent windows (~800–1200 tok).

    Sections are not derived here. The input is the flat block projection, which
    carries no structural markers, so windowing it is the only honest split; a
    title would have to be invented. Heading structure lives on the blocks
    (`kind == "heading"`, `level`) and a structural split must take the blocks as
    input rather than re-deriving them from a string.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if child_size_tokens < 1 or parent_size_tokens < 1:
        raise ValueError("chunk sizes must be >= 1")
    if child_overlap_tokens < 0 or child_overlap_tokens >= child_size_tokens:
        raise ValueError("child overlap must be >= 0 and < child size")
    if parent_overlap_tokens < 0 or parent_overlap_tokens >= parent_size_tokens:
        raise ValueError("parent overlap must be >= 0 and < parent size")

    units: list[ParentChildUnit] = []
    parent_index = 0
    child_index = 0

    parents = _token_windows(
        cleaned,
        size_tokens=parent_size_tokens,
        overlap_tokens=parent_overlap_tokens,
    )
    for parent_text in parents:
        children = _token_windows(
            parent_text,
            size_tokens=child_size_tokens,
            overlap_tokens=child_overlap_tokens,
        )
        if not children:
            children = [parent_text]
        for child_text in children:
            units.append(
                ParentChildUnit(
                    parent_text=parent_text,
                    child_text=child_text,
                    section_title=None,
                    parent_index=parent_index,
                    child_index=child_index,
                )
            )
            child_index += 1
        parent_index += 1

    return units
