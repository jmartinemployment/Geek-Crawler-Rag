"""Phase D1 — graph-style theme aggregation over hybrid chunk hits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from geek_crawler_rag.models import ChunkHit, ThemeHit


def build_theme_hits(
    chunks: list[ChunkHit],
    *,
    max_themes: int = 12,
) -> list[ThemeHit]:
    """
    Cluster hybrid parent/child hits into theme nodes + co-occurrence edges.

    Not a full property-graph store — a LlamaIndex/GraphRAG-shaped overlay that
    gives slide/strategy intents entity/theme relationships without a separate DB.
    """
    by_entity: dict[str, list[ChunkHit]] = defaultdict(list)
    by_category: dict[str, list[ChunkHit]] = defaultdict(list)

    for hit in chunks:
        entity = (hit.entity_name or "").strip() or hit.host or "unknown"
        by_entity[entity].append(hit)
        category = (hit.category or "").strip().lower() or "general"
        by_category[category].append(hit)

    themes: list[ThemeHit] = []
    # Entity nodes (strongest scores first).
    entity_rank = sorted(
        by_entity.items(),
        key=lambda kv: max((c.score for c in kv[1]), default=0.0),
        reverse=True,
    )
    for entity, hits in entity_rank[: max_themes // 2 or 1]:
        best = max(hits, key=lambda c: c.score)
        themes.append(
            ThemeHit(
                label=entity,
                relationship="entity",
                entity=entity,
                url=best.final_url or best.url,
                category=best.category,
                crawl_type=best.crawl_type,
                score=best.score,
            )
        )

    # Category themes.
    for category, hits in sorted(
        by_category.items(),
        key=lambda kv: max((c.score for c in kv[1]), default=0.0),
        reverse=True,
    )[: max(2, max_themes // 4)]:
        best = max(hits, key=lambda c: c.score)
        themes.append(
            ThemeHit(
                label=f"theme:{category}",
                relationship="category-theme",
                entity=best.entity_name,
                url=best.final_url or best.url,
                category=category,
                crawl_type=best.crawl_type,
                score=best.score,
            )
        )

    # Co-occurrence edges: entities that share a category within the hit set.
    entities = [e for e, _ in entity_rank[:8]]
    for i, left in enumerate(entities):
        left_cats = {(c.category or "general").lower() for c in by_entity[left]}
        for right in entities[i + 1 :]:
            right_cats = {(c.category or "general").lower() for c in by_entity[right]}
            shared = left_cats & right_cats
            if not shared:
                continue
            shared_cat = next(iter(shared))
            themes.append(
                ThemeHit(
                    label=f"{left} ↔ {right}",
                    relationship=f"co-occurs:{shared_cat}",
                    entity=left,
                    related_entity=right,
                    category=shared_cat,
                    score=min(
                        max((c.score for c in by_entity[left]), default=0.0),
                        max((c.score for c in by_entity[right]), default=0.0),
                    ),
                )
            )
            if len(themes) >= max_themes:
                return themes[:max_themes]

    return themes[:max_themes]


def prefer_parent_for_graph(request_prefer_parent: bool | None, request_prefer_child: bool | None) -> tuple[bool, bool]:
    """Graph mode defaults to parent sections unless caller forced child."""
    if request_prefer_child and not request_prefer_parent:
        return False, True
    return True, False


def graph_warning_if_empty(themes: list[ThemeHit], chunks: list[ChunkHit]) -> str | None:
    if themes or chunks:
        return None
    return "Graph retrieval returned no themes; notify-and-skip or fall back to hybrid."


def payload_entity_key(payload: dict[str, Any]) -> str:
    return str(payload.get("entityName") or payload.get("host") or "unknown").strip()
