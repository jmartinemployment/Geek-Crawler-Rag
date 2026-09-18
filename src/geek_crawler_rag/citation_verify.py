"""Quote verification against the shared block→plaintext projection."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

from geek_crawler_rag.models import GenerateCitation, GenerateSource

_WS = re.compile(r"\s+")


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


def verify_citations(
    citations: list[GenerateCitation],
    sources: list[GenerateSource],
    pages: list[dict[str, Any]],
) -> tuple[list[GenerateCitation], int]:
    """Keep citations only when source identity, digest, and quote agree."""
    pages_by_identity = {
        (
            str(page.get("pageId") or ""),
            str(page.get("url") or "").lower(),
        ): page
        for page in pages
        if page.get("url") and page.get("text")
    }
    allowed_sources = {
        (str(source.page_id or ""), source.url.lower())
        for source in sources
        if source.url
    }
    kept: list[GenerateCitation] = []
    dropped = 0
    for citation in citations:
        url_key = citation.url.lower()
        identity = (str(citation.page_id or ""), url_key)
        page = pages_by_identity.get(identity)
        body = str((page or {}).get("text") or "")
        body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if (
            (allowed_sources and identity not in allowed_sources)
            or page is None
            or not body
            or (
                citation.source_digest is not None
                and not hmac.compare_digest(citation.source_digest, body_digest)
            )
            or not quote_in_text(citation.quote, body, (page or {}).get("blocks"))
        ):
            dropped += 1
            continue
        kept.append(citation)
    return kept, dropped
