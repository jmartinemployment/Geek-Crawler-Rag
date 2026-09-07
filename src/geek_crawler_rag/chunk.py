"""Token-aware chunking: legacy sliding window + parent/child sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_ENCODING_NAME = "cl100k_base"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BLANK_RE = re.compile(r"\n{2,}")


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


def _split_heading_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown-ish text on AT headings; fallback to whole doc."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    matches = list(_HEADING_RE.finditer(cleaned))
    if not matches:
        return [(None, cleaned)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        preamble = cleaned[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        title = match.group(2).strip() or None
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = cleaned[start:end].strip()
        if body:
            sections.append((title, body))
        elif title:
            sections.append((title, title))
    return sections or [(None, cleaned)]


def parent_child_units(
    text: str,
    *,
    child_size_tokens: int = 200,
    child_overlap_tokens: int = 40,
    parent_size_tokens: int = 1000,
    parent_overlap_tokens: int = 80,
) -> list[ParentChildUnit]:
    """Build child pinpoint chunks nested under parent sections (~800–1200 tok)."""
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

    for section_title, section_body in _split_heading_sections(cleaned):
        parents = _token_windows(
            section_body,
            size_tokens=parent_size_tokens,
            overlap_tokens=parent_overlap_tokens,
        )
        if not parents:
            continue
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
                        section_title=section_title,
                        parent_index=parent_index,
                        child_index=child_index,
                    )
                )
                child_index += 1
            parent_index += 1

    return units
