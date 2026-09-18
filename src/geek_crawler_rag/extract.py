"""Typed blocks → plain text + title for embedding."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from geek_crawler_rag.block_text import derive_plaintext_from_blocks

_WS = re.compile(r"\s+")


def host_from_origin_or_url(origin: str | None, url: str | None) -> str:
    for candidate in (origin, url):
        if not candidate:
            continue
        raw = candidate.strip()
        if not raw:
            continue
        if "://" not in raw:
            raw = f"https://{raw}"
        try:
            host = urlparse(raw).hostname
            if host:
                return host.lower()
        except Exception:
            continue
    return ""


def extract_text_and_title(html: str | None) -> tuple[str, str | None]:
    if not html or not html.strip():
        return "", None

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    title: str | None = None
    if soup.title and soup.title.string:
        title = _WS.sub(" ", soup.title.string).strip() or None

    body = soup.body or soup
    text = body.get_text(separator=" ", strip=True)
    text = _WS.sub(" ", text).strip()
    return text, title


def page_text_and_title(
    *,
    blocks: list[dict[str, Any]] | None,
    title: str | None,
) -> tuple[str, str | None, bool]:
    """Project a page's typed blocks to plain text.

    The projection is shared with citation verification
    (:mod:`geek_crawler_rag.block_text`) rather than reimplemented, because a
    quote is taken from a retrieved chunk and then matched against this string:
    if the two are built differently, correct citations fail.

    Returns (text, title, has_blocks).
    """
    text = derive_plaintext_from_blocks(blocks)
    if not text:
        return "", title, False

    resolved = title
    if not resolved:
        for block in blocks or []:
            if block.get("kind") == "heading":
                candidate = (block.get("text") or "").strip()
                if candidate:
                    resolved = candidate
                    break
    return text, resolved, True
