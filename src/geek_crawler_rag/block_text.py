"""Block to plaintext projection — the single source of truth for page text.

The crawler emits typed blocks (Geek-Crawler-v2 `extract-content.ts`).
Everything downstream that needs a page as a string derives it here: the
chunker, so embeddings are built from it, and citation verification, so quotes
are matched against it.

One module rather than one loop per consumer, deliberately. Chunk text and
verification text must be the same string — a quote comes out of a retrieved
chunk and is then matched against the page projection, so any divergence makes
correct citations fail. Two implementations of "join the blocks" is exactly the
drift that broke this pipeline once already: the crawler changed corpus format
and the Library did not, and every page silently classified as empty.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

ROW_CELL_SEPARATOR = " | "
BLOCK_SEPARATOR = "\n\n"

# Some pages embed HTML inside their own text as unicode escapes — typically a
# meta description or JSON-LD blob authored by hand. Cheerio sees ordinary
# characters, so extraction faithfully preserves them and the chunk reaches the
# embedder carrying `u003ca href=u0022...` as literal tokens.
#
# Observed on freshbooks.com/hub and five stampli.com comparison pages, and the
# stored text carries NO backslash: it is a bare `u003c`, not `\u003c`. Both
# spellings are accepted here because the escaping depends on how the page
# authored it.
#
# Only the five sequences that spell markup are decoded. A general
# `\?u[0-9a-f]{4}` rule would rewrite any product code shaped like `u1234`, and
# str.decode("unicode-escape") is not an option at all: it round-trips through
# latin-1, turning "café" into "cafÃ©" across the whole corpus without raising.
_MARKUP_ESCAPES = {
    "003c": "<",
    "003e": ">",
    "0022": '"',
    "0026": "&",
    "0027": "'",
}
ESCAPED_MARKUP = re.compile(r"\\?u(003c|003e|0022|0026|0027)", re.I)

# Structural tags only, and only after decoding. `a` is included because a
# decoded anchor is exactly what this removes; `code` blocks skip the whole pass
# so a Bearer-header example like `<a token>` is never touched.
INLINE_HTML = re.compile(
    r"</?(?:div|span|p|a|ul|ol|li|table|tr|td|th|h[1-6]|br|img|section|article|nav|header|footer)\b[^>]*>",
    re.I,
)
_WHITESPACE = re.compile(r"\s+")


def clean_prose_text(text: str) -> str:
    """Decode escaped markup a page embedded in its own text, then drop it.

    `Click here u003ca href=u0022#videou0022u003eVideo u003c/au003e` becomes
    `Click here Video`. Leaves every other character untouched — this decodes
    five specific sequences rather than running a general unescape.
    """
    if not text:
        return ""
    if ESCAPED_MARKUP.search(text):
        text = ESCAPED_MARKUP.sub(lambda m: _MARKUP_ESCAPES[m.group(1).lower()], text)
        text = INLINE_HTML.sub(" ", text)
        text = _WHITESPACE.sub(" ", text)
    return text.strip()


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
            clean_prose_text(c) for c in (block.get("cells") or []) if c is not None
        )

    # `code` is exempt: its angle brackets are the subject matter. FreshBooks'
    # API docs carry `<accountId>` placeholders and `Bearer <a token>` headers,
    # and scrubbing those would destroy the documentation being indexed.
    raw = block.get("text") or ""
    if block.get("kind") == "code":
        return raw.strip()

    return clean_prose_text(raw)


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
