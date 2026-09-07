"""Page metadata heuristics + entity domain resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EntityRef:
    entity_id: str | None
    entity_name: str
    source_type: str
    domains: tuple[str, ...] = ()


def normalize_host(host: str | None) -> str:
    if not host:
        return ""
    h = host.strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h


def infer_category(url: str, text: str) -> str:
    path = (urlparse(url).path or "/").lower()
    blob = f"{path} {(text or '')[:500].lower()}"
    rules = (
        ("pricing", ("/pricing", "pricing", "plans and pricing")),
        ("docs", ("/docs", "/documentation", "/api/", "api reference")),
        ("blog", ("/blog", "/news", "/articles", "/resources")),
        ("product", ("/product", "/features", "/platform")),
        ("about", ("/about", "/company", "/team")),
        ("careers", ("/careers", "/jobs")),
        ("legal", ("/privacy", "/terms", "/legal")),
        ("compare", ("/compare", "/vs-", "versus", "alternative")),
    )
    for category, needles in rules:
        if any(n in blob for n in needles):
            return category
    return "page"


def infer_content_intent(url: str, text: str) -> str:
    category = infer_category(url, text)
    mapping = {
        "blog": "thought-leadership",
        "docs": "technical-reference",
        "pricing": "commercial",
        "product": "product-marketing",
        "compare": "competitive",
        "about": "company",
        "careers": "company",
        "legal": "legal",
    }
    return mapping.get(category, "general")


def infer_tags(url: str, title: str | None) -> list[str]:
    path = (urlparse(url).path or "").lower()
    parts = [p for p in path.split("/") if p and p not in {"index.html", "index.htm"}]
    tags: list[str] = []
    for part in parts[:8]:
        cleaned = re.sub(r"[^a-z0-9\-]+", "-", part).strip("-")
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
    if title:
        for tok in _TOKEN.findall(title.lower())[:6]:
            if len(tok) > 2 and tok not in tags:
                tags.append(tok)
    return tags[:12]


def quality_score(*, text: str, title: str | None, has_markdown: bool) -> float:
    length = len((text or "").strip())
    score = 0.35
    if title:
        score += 0.15
    if has_markdown:
        score += 0.15
    if length >= 500:
        score += 0.15
    if length >= 2000:
        score += 0.1
    if length >= 5000:
        score += 0.1
    # Penalize nav-heavy ultra-short pages.
    if length < 200:
        score -= 0.2
    return max(0.0, min(1.0, round(score, 3)))


def is_evergreen(url: str, text: str) -> bool:
    blob = f"{url} {(text or '')[:800]}".lower()
    ephemeral = ("2020", "2021", "2022", "2023", "2024", "2025", "2026", "/news/", "press release")
    if any(x in blob for x in ephemeral):
        return False
    return infer_category(url, text) in {"docs", "product", "pricing", "about", "page"}


def entity_from_crawl(crawl_type: str, host: str) -> EntityRef:
    source = (crawl_type or "unknown").strip().lower() or "unknown"
    name = normalize_host(host) or source
    return EntityRef(entity_id=None, entity_name=name, source_type=source, domains=(name,) if name else ())


def entity_from_doc(doc: dict[str, Any], *, fallback_host: str, crawl_type: str) -> EntityRef:
    entity_id = doc.get("id") or doc.get("Id") or doc.get("_id")
    if entity_id is not None:
        entity_id = str(entity_id)
    name = (
        doc.get("entityName")
        or doc.get("EntityName")
        or doc.get("name")
        or doc.get("Name")
        or normalize_host(fallback_host)
        or crawl_type
    )
    source = (
        doc.get("sourceType")
        or doc.get("SourceType")
        or doc.get("type")
        or doc.get("Type")
        or crawl_type
        or "unknown"
    )
    domains_raw = doc.get("domains") or doc.get("Domains") or []
    domains: list[str] = []
    if isinstance(domains_raw, str):
        domains = [normalize_host(domains_raw)]
    elif isinstance(domains_raw, list):
        domains = [normalize_host(str(d)) for d in domains_raw if d]
    return EntityRef(
        entity_id=entity_id,
        entity_name=str(name),
        source_type=str(source).strip().lower() or "unknown",
        domains=tuple(d for d in domains if d),
    )
