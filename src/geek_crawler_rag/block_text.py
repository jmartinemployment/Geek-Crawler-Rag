"""Block to plaintext projection — the single source of truth for page text.

The crawler emits typed blocks (Geek-Crawler-v2 `extract-content.ts`), not
Markdown. Everything downstream that needs a page as a string derives it here:
the chunker, so embeddings are built from it, and citation verification, so
quotes are matched against it.

One module rather than one loop per consumer, deliberately. Chunk text and
verification text must be the same string — a quote comes out of a retrieved
chunk and is then matched against the page projection, so any divergence makes
correct citations fail. Two implementations of "join the blocks" is exactly the
drift that broke this pipeline once already: the crawler migrated off Markdown
and the Library did not, and every page silently classified as empty.
"""

from __future__ import annotations

from typing import Any, Iterable

ROW_CELL_SEPARATOR = " | "
BLOCK_SEPARATOR = "\n\n"


def render_block_text(block: dict[str, Any] | None) -> str:
    """The single, authoritative text projection for one Block.

    Shared by both chunking and verification engines to eliminate drift.

    `row` is the only block kind carrying `cells` instead of `text`; mapping
    `block["text"]` across a page therefore drops every table row, taking tables
    out of the verification target while the chunker still indexes them.
    """
    if not block:
        return ""

    if block.get("kind") == "row":
        # `|` is not whitespace, so it survives citation_verify._normalize_ws and
        # acts as an explicit structural anchor between cells.
        return ROW_CELL_SEPARATOR.join(
            c.strip() for c in (block.get("cells") or []) if c is not None
        )

    return (block.get("text") or "").strip()


def derive_plaintext_from_blocks(blocks: Iterable[dict[str, Any]] | None) -> str:
    """Whole-page projection.

    The join is shared for the same reason the per-block projection is: two
    consumers that agree on rendering but differ on the join still produce
    different strings, and the invariant is equality of the final text rather
    than of its parts.

    The separator itself is cosmetic for matching — `_normalize_ws` collapses
    all whitespace before comparison — so it is chosen for readable chunk text,
    not to affect verification.
    """
    if not blocks:
        return ""
    rendered = (render_block_text(block) for block in blocks)
    return BLOCK_SEPARATOR.join(line for line in rendered if line.strip())
