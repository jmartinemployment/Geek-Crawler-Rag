"""Token-aware chunking: legacy sliding window + parent/child sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import tiktoken

from geek_crawler_rag.block_text import derive_plaintext_from_blocks

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
    #: Anchors from the blocks of this chunk's own section. A tool name without
    #: its href cites nothing, and attributing anchors to the heading they sit
    #: under is the whole point of sectioning by block rather than by token count.
    section_anchors: tuple[tuple[str, str], ...] = ()


def _token_windows(
    text: str,
    *,
    size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    return chunk_text(text, size_tokens=size_tokens, overlap_tokens=overlap_tokens)


@dataclass(frozen=True)
class BlockSection:
    """One heading and the blocks beneath it, projected to text."""

    title: str | None
    level: int | None
    text: str
    anchors: list[dict[str, str]]


def split_blocks_into_sections(
    blocks: Iterable[dict[str, Any]] | None,
) -> list[BlockSection]:
    """Group blocks into sections at heading boundaries.

    A `heading` block starts a section and supplies its title; the heading stays
    in the body so the words a reader would search for are embedded with the
    prose under them. Blocks before the first heading become a preamble whose
    title is `None` — that is an absent heading, not an invented one.

    Every section's text comes from :func:`derive_plaintext_from_blocks`, the same
    projection that produces the whole-page string and the verification target.
    That is deliberate and load-bearing: a quote is taken from a retrieved chunk
    and then matched against the page projection, so section text has to be the
    page text restricted to those blocks — identical rendering, identical join —
    or correct citations start failing.

    Splitting happens at *every* heading regardless of depth. `level` is carried
    so a future hierarchy can nest them; nothing here interprets it.
    """
    if not blocks:
        return []

    groups: list[tuple[str | None, int | None, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    title: str | None = None
    level: int | None = None

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("kind") == "heading":
            if current:
                groups.append((title, level, current))
            heading_text = (block.get("text") or "").strip()
            title = heading_text or None
            raw_level = block.get("level")
            level = raw_level if isinstance(raw_level, int) else None
            current = [block]
            continue
        current.append(block)

    if current:
        groups.append((title, level, current))

    sections: list[BlockSection] = []
    for section_title, section_level, section_blocks in groups:
        text = derive_plaintext_from_blocks(section_blocks)
        if not text.strip():
            continue
        anchors: list[dict[str, str]] = []
        seen: set[str] = set()
        for block in section_blocks:
            for anchor in block.get("anchors") or []:
                if not isinstance(anchor, dict):
                    continue
                href = str(anchor.get("href") or "").strip()
                label = str(anchor.get("label") or "").strip()
                if not href or href in seen:
                    continue
                seen.add(href)
                anchors.append({"label": label, "href": href})
        sections.append(
            BlockSection(
                title=section_title,
                level=section_level,
                text=text,
                anchors=anchors,
            )
        )
    return sections


def parent_child_units(
    blocks: Iterable[dict[str, Any]] | None,
    *,
    child_size_tokens: int = 200,
    child_overlap_tokens: int = 40,
    parent_size_tokens: int = 1000,
    parent_overlap_tokens: int = 80,
) -> list[ParentChildUnit]:
    """Build child pinpoint chunks nested under parent windows (~800-1200 tok).

    Parents are windowed **within a section**, so a parent never straddles two
    headings and every chunk carries the heading it actually sits under. Before
    this, the only available input was the flat page projection, which has no
    structural markers: every page was one nameless section and `sectionTitle`
    was empty on every chunk in the corpus.

    A section longer than `parent_size_tokens` still splits into several parents —
    headings bound the windows, they do not replace them.
    """
    if child_size_tokens < 1 or parent_size_tokens < 1:
        raise ValueError("chunk sizes must be >= 1")
    if child_overlap_tokens < 0 or child_overlap_tokens >= child_size_tokens:
        raise ValueError("child overlap must be >= 0 and < child size")
    if parent_overlap_tokens < 0 or parent_overlap_tokens >= parent_size_tokens:
        raise ValueError("parent overlap must be >= 0 and < parent size")

    sections = split_blocks_into_sections(blocks)
    if not sections:
        return []

    units: list[ParentChildUnit] = []
    parent_index = 0
    child_index = 0

    for section in sections:
        parents = _token_windows(
            section.text,
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
                        section_title=section.title,
                        parent_index=parent_index,
                        child_index=child_index,
                        section_anchors=tuple(
                            (a["label"], a["href"]) for a in section.anchors
                        ),
                    )
                )
                child_index += 1
            parent_index += 1

    return units
