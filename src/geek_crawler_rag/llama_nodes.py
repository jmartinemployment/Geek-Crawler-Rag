"""Build LlamaIndex TextNodes from crawl pages (parent/child + metadata)."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import TextNode

from geek_crawler_rag.chunk import parent_child_units
from geek_crawler_rag.config import Settings
from geek_crawler_rag.extract import host_from_origin_or_url, page_text_and_title
from geek_crawler_rag.language import is_english
from geek_crawler_rag.metadata import (
    EntityRef,
    infer_category,
    infer_content_intent,
    infer_tags,
    is_evergreen,
    quality_score,
)
from geek_crawler_rag.mongo import CrawlPage
from geek_crawler_rag.qdrant_store import point_id


def page_to_nodes(
    *,
    page: CrawlPage,
    run_id: str,
    crawl_type: str,
    entity: EntityRef,
    settings: Settings,
) -> tuple[list[TextNode], str]:
    """Return (nodes, skip_reason). skip_reason is empty on success."""
    text, title, used_markdown = page_text_and_title(
        markdown=page.markdown,
        title=page.title,
        html=page.html,
    )
    if not text:
        return [], "empty"
    if not is_english(text):
        return [], "lang"

    host = host_from_origin_or_url(page.origin, page.url)
    category = infer_category(page.url, text)
    content_intent = infer_content_intent(page.url, text)
    tags = infer_tags(page.url, title)
    qscore = quality_score(text=text, title=title, has_markdown=used_markdown)
    evergreen = is_evergreen(page.url, text)

    units = parent_child_units(
        text,
        child_size_tokens=settings.child_chunk_size_tokens,
        child_overlap_tokens=settings.child_chunk_overlap_tokens,
        parent_size_tokens=settings.parent_chunk_size_tokens,
        parent_overlap_tokens=settings.parent_chunk_overlap_tokens,
    )
    if not units:
        return [], "empty"

    nodes: list[TextNode] = []
    seen_parents: set[int] = set()
    for unit in units:
        if unit.parent_index not in seen_parents:
            seen_parents.add(unit.parent_index)
            nodes.append(
                _node(
                    run_id=run_id,
                    crawl_type=crawl_type,
                    host=host,
                    page=page,
                    title=title,
                    unit_parent=unit.parent_text,
                    unit_child="",
                    section_title=unit.section_title,
                    chunk_index=unit.parent_index,
                    chunk_role="parent",
                    entity=entity,
                    category=category,
                    content_intent=content_intent,
                    tags=tags,
                    quality=qscore,
                    evergreen=evergreen,
                    embed_text=unit.parent_text,
                    point_key=f"parent:{unit.parent_index}",
                )
            )
        nodes.append(
            _node(
                run_id=run_id,
                crawl_type=crawl_type,
                host=host,
                page=page,
                title=title,
                unit_parent=unit.parent_text,
                unit_child=unit.child_text,
                section_title=unit.section_title,
                chunk_index=unit.child_index,
                chunk_role="child",
                entity=entity,
                category=category,
                content_intent=content_intent,
                tags=tags,
                quality=qscore,
                evergreen=evergreen,
                embed_text=unit.child_text,
                point_key=f"child:{unit.child_index}",
            )
        )
    return nodes, ""


def _node(
    *,
    run_id: str,
    crawl_type: str,
    host: str,
    page: CrawlPage,
    title: str | None,
    unit_parent: str,
    unit_child: str,
    section_title: str | None,
    chunk_index: int,
    chunk_role: str,
    entity: EntityRef,
    category: str,
    content_intent: str,
    tags: list[str],
    quality: float,
    evergreen: bool,
    embed_text: str,
    point_key: str,
) -> TextNode:
    metadata: dict[str, Any] = {
        "runId": run_id,
        "crawlType": crawl_type,
        "host": host,
        "url": page.url,
        "finalUrl": page.final_url or page.url,
        "chunkIndex": chunk_index,
        "language": "en",
        "title": title,
        "pageId": page.id,
        "parentText": unit_parent,
        "childText": unit_child,
        "sectionTitle": section_title,
        "chunkRole": chunk_role,
        "sourceType": entity.source_type,
        "entityName": entity.entity_name,
        "entityId": entity.entity_id,
        "category": category,
        "contentIntent": content_intent,
        "tags": tags,
        "qualityScore": quality,
        "isEvergreen": evergreen,
        "lastCrawled": page.crawled_at,
    }
    # Drop Nones — Qdrant/LlamaIndex payload hygiene.
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return TextNode(
        id_=point_id(run_id, page.id, point_key),
        text=embed_text,
        metadata=metadata,
    )
