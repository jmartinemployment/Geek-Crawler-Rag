"""Markdown quote verification helpers used by library and diagnostic paths."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

from geek_crawler_rag.models import GenerateCitation, GenerateSource

_WS = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").strip()).lower()


def quote_in_markdown(quote: str, markdown: str) -> bool:
    """True when quote (normalized) appears in markdown."""
    q = _normalize_ws(quote)
    if len(q) < 12:
        return False
    body = _normalize_ws(markdown)
    return q in body


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
        if page.get("url") and page.get("markdown")
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
        body = str((page or {}).get("markdown") or "")
        body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if (
            (allowed_sources and identity not in allowed_sources)
            or page is None
            or not body
            or (
                citation.source_digest is not None
                and not hmac.compare_digest(citation.source_digest, body_digest)
            )
            or not quote_in_markdown(citation.quote, body)
        ):
            dropped += 1
            continue
        kept.append(citation)
    return kept, dropped
