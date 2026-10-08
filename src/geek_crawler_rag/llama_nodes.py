"""Build LlamaIndex TextNodes from crawl pages (parent/child + metadata)."""

from __future__ import annotations

import hashlib
from typing import Any

from llama_index.core.schema import TextNode

from geek_crawler_rag.chunk import is_stub_text, parent_child_units
from geek_crawler_rag.chunk_tokenizer import ChunkTokenizer
from geek_crawler_rag.config import Settings
from geek_crawler_rag.embedding_sanitize import sanitize_embedding_text
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


def page_source_digest(page: CrawlPage) -> str | None:
    """The one definition of a page's `sourceDigest`: sha256 of its `contentHtml`.

    Stamped on every point at index time and returned by `POST /v1/verify`, so a passage and a
    verdict can be matched to the same page version. It is identification, never a verdict.

    None when the page has no `contentHtml`. It used to fall back to the raw `html`, which made the
    value mean two different things depending on the page; a page without the clean fragment has
    no sourceDigest rather than one computed from something else.
    """
    if not page.content_html:
        return None
    return hashlib.sha256(page.content_html.encode("utf-8")).hexdigest()


def text_digest(embed_text: str) -> str:
    """SHA-256 of the exact string the embedder receives, stored on the point as `textDigest`.

    Sanitised first because `embed_and_upsert` sanitises before embedding: two raw strings that
    differ only in a zero-width character are one embedded text and one vector, and this is the
    key the indexer collapses repeats on (`IndexService._admit_page_nodes`).
    """
    return hashlib.sha256(sanitize_embedding_text(embed_text).encode("utf-8")).hexdigest()


def page_to_nodes(
    *,
    page: CrawlPage,
    run_id: str,
    crawl_type: str,
    entity: EntityRef,
    settings: Settings,
    tokenizer: ChunkTokenizer,
) -> tuple[list[TextNode], str, int]:
    """Return (nodes, skip_reason, stubs_skipped). skip_reason is empty on success.

    stubs_skipped counts the units dropped by `chunk.is_stub_text` -- a heading standing alone, a
    breadcrumb, a fragment of fewer than eight words -- which used to be indexed as passages and
    took retrieval slots nothing could be quoted from (plans/retrieval-from-the-brief.md P3). The
    indexer reports the total as `chunksSkippedStub`. A page whose every unit is a stub returns no
    nodes and no skip reason: it is English and was read; it simply holds nothing to index.
    """
    text, title, has_blocks = page_text_and_title(
        blocks=page.blocks,
        title=page.title,
    )
    if not text:
        return [], "empty", 0
    if not is_english(text):
        return [], "lang", 0

    host = host_from_origin_or_url(page.origin, page.url)
    category = infer_category(page.url, text)
    content_intent = infer_content_intent(page.url, text)
    tags = infer_tags(page.url, title)
    qscore = quality_score(text=text, title=title, has_blocks=has_blocks)
    evergreen = is_evergreen(page.url, text)

    # Sections come from the typed blocks, so each chunk carries the heading it
    # actually sits under and the anchors from that heading's own blocks.
    units = parent_child_units(
        page.blocks,
        tokenizer=tokenizer,
        child_size_tokens=settings.child_chunk_size_tokens,
        child_overlap_tokens=settings.child_chunk_overlap_tokens,
        parent_size_tokens=settings.parent_chunk_size_tokens,
        parent_overlap_tokens=settings.parent_chunk_overlap_tokens,
    )
    if not units:
        return [], "empty", 0

    # Once per page, not once per node. This hashes the whole page body, and it
    # used to sit inside _node(), which the loop below calls for every chunk -- so
    # a page producing fifty chunks hashed the entire page fifty times, on the
    # event loop, for a value that is per page by definition.
    source_digest = page_source_digest(page)

    nodes: list[TextNode] = []
    seen_parents: set[int] = set()
    stubs_skipped = 0
    crawl_norm = (crawl_type or "").strip().lower()
    is_competitor = crawl_norm in {"competitor", "competitors"}
    for unit in units:
        # A stub child means a stub parent: children are token windows of their parent, so the
        # first window is full-size unless the parent itself is short. Skipping the unit skips
        # both; a real parent's short tail window is skipped alone and the parent keeps its text.
        if is_stub_text(unit.child_text, unit.section_title):
            stubs_skipped += 1
            continue
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
        unit_anchors = [
            {"label": label, "href": href} for label, href in unit.section_anchors
        ]
        # A parent point only when the parent actually carries more than its child.
        #
        # Measured 2026-09-30 over 3,000 sampled child points: 71.4% have a parentText byte-identical
        # to their childText, because parent_child_units emits a child equal to its parent whenever a
        # heading section is shorter than child_chunk_size_tokens (chunk.py). Those parent points were
        # 45.2% of the collection and pure duplication -- they competed with their own child for the
        # same top-k slots, and under BM25's `idf` modifier they inflated both df and N, depressing
        # the weight of exactly the rare literals the sparse channel exists to bind.
        #
        # The other 28.6% are real: a parent ~3x its child at p50, carrying meaning no 200-token child
        # contains -- pricing matrices, multi-step procedures, a qualifier hundreds of tokens from the
        # feature it qualifies. Dropping the tier outright would have removed that silently, so the
        # test is emptiness of the difference, not the tier.
        parent_adds_context = unit.parent_text.strip() != unit.child_text.strip()
        if unit.parent_index not in seen_parents and parent_adds_context:
            seen_parents.add(unit.parent_index)
            nodes.append(
                _node(
                    run_id=run_id,
                    source_digest=source_digest,
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
                    embedding_model=settings.embedding_model,
                    source_rights_consented_hosts=settings.source_rights_consented_hosts,
                    competitor_chunk_kind=unit_kind,
                    feature_tag=feature_tag,
                    anchors=unit_anchors,
                )
            )
        nodes.append(
            _node(
                run_id=run_id,
                source_digest=source_digest,
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
                embedding_model=settings.embedding_model,
                source_rights_consented_hosts=settings.source_rights_consented_hosts,
                competitor_chunk_kind=unit_kind,
                feature_tag=feature_tag,
                anchors=unit_anchors,
            )
        )
    return nodes, "", stubs_skipped


def _node(
    *,
    run_id: str,
    source_digest: str | None,
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
        "sourceDigest": source_digest,
        "textDigest": text_digest(embed_text),
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
