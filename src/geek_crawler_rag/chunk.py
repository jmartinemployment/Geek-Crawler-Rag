"""Chunking: sliding window + parent/child sections, sized in the embedding model's own tokens.

``tokenizer`` is a required argument throughout, never defaulted. It used to be tiktoken's
``cl100k_base``, chosen implicitly inside this module, while the embedder counted WordPiece -- and
the gap silently truncated 6.9% of parent chunks at inference. A default here is what let those two
disagree without anyone passing anything wrong, so there is no default to fall back to. See
``chunk_tokenizer`` for why chunks are sliced from the original string rather than decoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from geek_crawler_rag.block_text import derive_plaintext_from_blocks
from geek_crawler_rag.chunk_tokenizer import ChunkTokenizer


def chunk_text(
    text: str,
    *,
    tokenizer: ChunkTokenizer,
    size_tokens: int = 650,
    overlap_tokens: int = 80,
) -> list[str]:
    """Sliding token window over ``text``, returning verbatim slices of it.

    Each chunk is ``text[span_of_first_token.start : span_of_last_token.end]`` -- the original
    characters, including whatever punctuation, casing and interior whitespace they had. Nothing is
    reconstructed from tokens.

    ``size_tokens`` counts CONTENT tokens. The model adds ``special_token_overhead`` on top, so a
    caller wanting to stay inside a 512-token model must ask for at most 510. Enforced below rather
    than documented and hoped for.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if size_tokens < 1:
        raise ValueError("size_tokens must be >= 1")
    if overlap_tokens < 0 or overlap_tokens >= size_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < size_tokens")

    limit = tokenizer.sequence_limit
    usable = limit - tokenizer.special_token_overhead if limit > 0 else 0
    if usable > 0 and size_tokens > usable:
        # Fail loudly at the boundary rather than let the model truncate silently at inference.
        # This is a configuration error, not a data condition.
        raise ValueError(
            f"size_tokens={size_tokens} exceeds the model's usable budget {usable} "
            f"(sequence_limit={limit} minus {tokenizer.special_token_overhead} special tokens)"
        )

    spans = tokenizer.token_spans(cleaned)
    if not spans:
        return []

    chunks: list[str] = []
    start = 0
    step = size_tokens - overlap_tokens
    while start < len(spans):
        end = min(start + size_tokens, len(spans))
        piece, end = _fit_window(cleaned, spans, start, end, tokenizer, size_tokens)
        if piece:
            chunks.append(piece)
        if end >= len(spans):
            break
        start += step
    return chunks


def _fit_window(
    cleaned: str,
    spans: list[tuple[int, int]],
    start: int,
    end: int,
    tokenizer: ChunkTokenizer,
    size_tokens: int,
) -> tuple[str, int]:
    """Slice tokens ``[start, end)`` and shrink until the SLICE measures within budget.

    The span arithmetic alone is not sufficient, and assuming it was is how this nearly shipped with
    the bug it was written to remove. A slice covering exactly N tokens of ``cleaned`` can tokenize
    to MORE than N tokens when measured on its own: WordPiece marks word continuations with ``##``,
    so a piece that was ``##tion`` inside ``reconciliation`` becomes the standalone word ``tion``
    when the slice starts there, and a standalone word may split into more pieces than the one it
    came from. Measured on 27,309 live points at a 500-token budget: parents reached 502, a drift of
    +2. Harmless at 500 against a 512 ceiling, and fatal at 510 -- which the caller is allowed to
    ask for.

    So the budget is enforced on the produced string, which is the thing the model actually reads.
    Shrinking is not a fallback for a failed path: it is the correct computation of "N tokens of
    text", done against the tokenizer rather than against an index. It terminates because ``end``
    strictly decreases and stops at ``start + 1``.
    """
    while end > start:
        piece = cleaned[spans[start][0] : spans[end - 1][1]].strip()
        if not piece:
            return "", end
        if len(tokenizer.token_spans(piece)) <= size_tokens:
            return piece, end
        end -= 1
    return "", start + 1


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
    tokenizer: ChunkTokenizer,
    size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    return chunk_text(
        text,
        tokenizer=tokenizer,
        size_tokens=size_tokens,
        overlap_tokens=overlap_tokens,
    )


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
    tokenizer: ChunkTokenizer,
    child_size_tokens: int = 200,
    child_overlap_tokens: int = 40,
    parent_size_tokens: int = 1000,
    parent_overlap_tokens: int = 80,
) -> list[ParentChildUnit]:
    """Build child pinpoint chunks nested under parent windows.

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
            tokenizer=tokenizer,
            size_tokens=parent_size_tokens,
            overlap_tokens=parent_overlap_tokens,
        )
        if not parents:
            continue
        for parent_text in parents:
            children = _token_windows(
                parent_text,
                tokenizer=tokenizer,
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
