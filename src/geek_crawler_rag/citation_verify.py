"""Quote verification against the shared block→plaintext projection."""

from __future__ import annotations

import re

from geek_crawler_rag.block_text import clean_prose_text

_WS = re.compile(r"\s+")

# F-R10. GeekAPI cuts quote candidates from a page's blocks in C# (`GccCorpusBlockMapper`); this
# service projects the same blocks to text in Python (`block_text`). Run over one fixture
# (`contracts/block-projection/v1.json`, pinned by tests on both sides) the two produce the same
# bytes except in exactly two places, and both occur in the corpus:
#
# * Escaped markup in prose -- `u003c u003e u0022 u0026 u0027`, with or without a backslash. Python
#   decodes the five and drops the tags they spell (`clean_prose_text`); C# keeps them raw. 28
#   blocks on 2026-10-04.
# * Empty table cells. Python keeps an empty cell, so its separator appears twice: `A |  | B`. C#
#   drops it: `A | B`. 36,090 rows on 2026-10-04.
#
# `verify_quote` normalises those two and nothing else. No fold table: curly quotes, dashes and
# non-breaking spaces are left exactly as the page has them, because both projections leave them.
_EMPTY_CELL_RUN = re.compile(r"\|(?:\s*\|)+")


def _normalize_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").strip()).lower()


def quote_in_text(quote: str, plain_text: str, blocks: list | None = None) -> bool:
    """True when the quote appears in the page's plaintext projection.

    Short quotes fall through to a whole-cell match. Table cells are routinely
    under the 12-character floor — "$15/month" is 9, "99.9%" is 5 — so the floor
    alone turns correct citations of a price or spec into verification failures.
    A whole-cell match is an exact field match rather than a substring
    coincidence, which is what makes relaxing the floor safe there.
    """
    q = _normalize_ws(quote)
    if not q or not any(ch.isalnum() for ch in q):
        return False

    body = _normalize_ws(plain_text)
    if len(q) >= 12:
        return q in body

    for block in blocks or []:
        if not isinstance(block, dict) or block.get("kind") != "row":
            continue
        for cell in block.get("cells") or []:
            if cell and _normalize_ws(str(cell)) == q:
                return True
    return False


def verify_quote(quote: str, plain_text: str, blocks: list | None = None) -> bool:
    """`quote_in_text` for a quote cut by GeekAPI's C# block projection. Used by `POST /v1/verify`.

    The quote is passed through the same escaped-markup cleaning the Python projection applied to
    the page, and runs of empty-cell separators in the page text collapse to one, so a C# string
    and the Python projection of the same blocks compare equal. A code block's text is not cleaned
    by the Python projection; a quote from one that carries an escape sequence therefore fails
    here. That is a refusal, never a false match.
    """
    return quote_in_text(
        clean_prose_text(quote), _EMPTY_CELL_RUN.sub("|", plain_text), blocks
    )
