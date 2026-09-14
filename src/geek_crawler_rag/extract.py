"""HTML → plain text + title for embedding."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

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
    markdown: str | None,
    title: str | None,
    html: str | None = None,
) -> tuple[str, str | None, bool]:
    """Require crawler Markdown. Do not synthesize from HTML.

    HTML-only pages are deleted (cleanup / index sweeper), not backfilled.
    ``html`` is accepted for call-site compatibility and ignored.

    Returns (text, title, used_markdown).
    """
    del html  # never fall back to HTML extract
    if markdown and markdown.strip():
        md = markdown.strip()
        md_title = title
        if not md_title:
            for line in md.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    md_title = line.lstrip("#").strip() or None
                    break
        return md, md_title, True

    return "", title, False
