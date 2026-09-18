"""Build LlamaIndex TextNodes from crawl pages (parent/child + metadata)."""

from __future__ import annotations

import hashlib
from typing import Any

from llama_index.core.schema import TextNode

from geek_crawler_rag.chunk import parent_child_units
from geek_crawler_rag.config import Settings
from geek_crawler_rag.extract import host_from_origin_or_url, page_text_and_title
from geek_crawler_rag.language import is_english
from geek_crawler_rag.metadata import (
    EntityRef,
    infer_category,
    infer_competitor_chunk_kind,
    infer_content_intent,
    infer_feature_tag,
    infer_tags,
    is_evergreen,
    quality_score,
    resolve_source_rights,
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
    text, title, has_blocks = page_text_and_title(
        blocks=page.blocks,
        title=page.title,
    )
    if not text:
        return [], "empty"
    if not is_english(text):
        return [], "lang"

    host = host_from_origin_or_url(page.origin, page.url)
    # Carried so a retrieved chunk can cite its sources: a tool name without its
    # href cites nothing. Chunking works over the flat projection, which cannot
    # attribute an anchor to one chunk, so these are page-level and deduped.
    anchors: list[dict[str, str]] = []
    seen_hrefs: set[str] = set()
    for block in page.blocks:
        for anchor in block.get("anchors") or []:
            if not isinstance(anchor, dict):
                continue
            href = str(anchor.get("href") or "").strip()
            label = str(anchor.get("label") or "").strip()
            if not href or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            anchors.append({"label": label, "href": href})
    category = infer_category(page.url, text)
    content_intent = infer_content_intent(page.url, text)
    tags = infer_tags(page.url, title)
    qscore = quality_score(text=text, title=title, has_blocks=has_blocks)
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
    crawl_norm = (crawl_type or "").strip().lower()
    is_competitor = crawl_norm in {"competitor", "competitors"}
    for unit in units:
        unit_kind = (
            infer_competitor_chunk_kind(
                url=page.url,
                section_title=unit.section_title,
                text=unit.child_text or unit.parent_text,
                category=category,
            )
            if is_competitor
            else None
        )
        feature_tag = (
            infer_feature_tag(unit.section_title, tags) if is_competitor else None
        )
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
                    owner_id=settings.crawler_owner_id,
                    visibility=settings.crawler_visibility,
                    embedding_model=settings.openai_embedding_model,
                    source_rights_consented_hosts=settings.source_rights_consented_hosts,
                    competitor_chunk_kind=unit_kind,
                    feature_tag=feature_tag,
                    anchors=anchors,
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
                owner_id=settings.crawler_owner_id,
                visibility=settings.crawler_visibility,
                embedding_model=settings.openai_embedding_model,
                source_rights_consented_hosts=settings.source_rights_consented_hosts,
                competitor_chunk_kind=unit_kind,
                feature_tag=feature_tag,
                anchors=anchors,
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
    owner_id: str,
    visibility: str,
    embedding_model: str,
    source_rights_consented_hosts: str = "",
    competitor_chunk_kind: str | None = None,
    feature_tag: str | None = None,
    anchors: list[dict[str, str]] | None = None,
) -> TextNode:
    anchors = anchors or []
    metadata: dict[str, Any] = {
        "ownerId": owner_id,
        "visibility": visibility,
        "manifestEligible": False,
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
        "anchors": anchors,
        "sourceDigest": hashlib.sha256(
            (page.content_html or page.html or "").encode("utf-8")
        ).hexdigest(),
        "sourceRights": resolve_source_rights(
            host=host,
            consented_hosts=tuple(
                h.strip()
                for h in (source_rights_consented_hosts or "").split(",")
                if h.strip()
            ),
        ),
        "parserId": "crawler-blocks",
        "parserVersion": "1.0.0",
        "chunkerId": "parent-child-token-window",
        "chunkerVersion": "1.0.0",
        "embeddingModel": embedding_model,
    }
    crawl_norm = (crawl_type or "").strip().lower()
    if crawl_norm in {"competitor", "competitors"}:
        # competitor-extraction §9 — never stamp crawlType partner on rival chunks
        metadata["competitorName"] = entity.entity_name
        if competitor_chunk_kind:
            metadata["competitorChunkKind"] = competitor_chunk_kind
        if feature_tag:
            metadata["featureTag"] = feature_tag
    # Drop Nones — Qdrant/LlamaIndex payload hygiene.
    metadata = {k: v for k, v in metadata.items() if v is not None}
    node_id = point_id(run_id, page.id, point_key)
    metadata["chunkId"] = node_id
    return TextNode(
        id_=node_id,
        text=embed_text,
        metadata=metadata,
    )
