"""Phase D2 — few-shot ad template index (separate Qdrant collection)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from llama_index.core.schema import TextNode
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from geek_crawler_rag.config import Settings
from geek_crawler_rag.llama_engine import LlamaIndexEngine
from geek_crawler_rag.models import (
    AdTemplateHit,
    AdTemplateIndexRequest,
    AdTemplateIndexResponse,
    AdTemplateQueryRequest,
    AdTemplateQueryResponse,
    AdTemplateUpsertItem,
)

logger = logging.getLogger(__name__)

_TEMPLATE_NS = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")


def template_point_id(template_id: str) -> str:
    return str(uuid.uuid5(_TEMPLATE_NS, template_id.strip().lower()))


class AdTemplateIndexService:
    """Embed + retrieve operator ad templates. Corpus owned by content-creator-v2."""

    def __init__(self, settings: Settings, llama: LlamaIndexEngine) -> None:
        self._settings = settings
        self._llama = llama
        self._collection = settings.qdrant_ad_templates_collection
        client_kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key
        self._client = AsyncQdrantClient(**client_kwargs)

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._settings.embedding_dimensions,
                    distance=qm.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", self._collection)
        for field in ("channel", "framework", "entityTags"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Template payload index %s skipped", field)

    async def index(self, request: AdTemplateIndexRequest) -> AdTemplateIndexResponse:
        await self.ensure_collection()
        nodes: list[TextNode] = []
        ids: list[str] = []
        for item in request.templates:
            tid = (item.id or "").strip() or str(uuid.uuid4())
            body = (item.body or "").strip()
            if len(body) < 8:
                continue
            pid = template_point_id(tid)
            ids.append(pid)
            text = "\n".join(
                p
                for p in [
                    item.name,
                    f"channel: {item.channel}" if item.channel else "",
                    f"framework: {item.framework}" if item.framework else "",
                    body,
                ]
                if p
            )
            meta = {
                "templateId": tid,
                "name": item.name,
                "channel": (item.channel or "").strip().lower() or None,
                "framework": (item.framework or "").strip().lower() or None,
                "tone": item.tone,
                "entityTags": [t.strip() for t in (item.entity_tags or []) if t.strip()][:12],
                "body": body,
                "text": text,
            }
            nodes.append(TextNode(id_=pid, text=text, metadata=meta))

        if not nodes:
            return AdTemplateIndexResponse(upserted=0, warning="No valid templates to index")

        # Embed via shared OpenAI embed model, upsert into templates collection.
        texts = [n.get_content() for n in nodes]
        embeddings = await self._llama.embed_model.aget_text_embedding_batch(texts)
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=node.metadata)
            for pid, vec, node in zip(ids, embeddings, nodes, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return AdTemplateIndexResponse(upserted=len(points))

    async def query(self, request: AdTemplateQueryRequest) -> AdTemplateQueryResponse:
        await self.ensure_collection()
        query_vec = await self._llama.embed_model.aget_query_embedding(request.need)
        must: list[qm.Condition] = []
        if request.channel:
            must.append(
                qm.FieldCondition(
                    key="channel",
                    match=qm.MatchValue(value=request.channel.strip().lower()),
                )
            )
        if request.framework:
            must.append(
                qm.FieldCondition(
                    key="framework",
                    match=qm.MatchValue(value=request.framework.strip().lower()),
                )
            )
        if request.entity_tags:
            must.append(
                qm.FieldCondition(
                    key="entityTags",
                    match=qm.MatchAny(any=[t.strip() for t in request.entity_tags if t]),
                )
            )
        query_filter = qm.Filter(must=must) if must else None
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=query_vec,
                query_filter=query_filter,
                limit=request.top_k,
                with_payload=True,
            )
        except Exception as ex:
            logger.exception("Ad template query failed: %s", ex)
            return AdTemplateQueryResponse(
                templates=[],
                warning=f"Ad template query failed: {ex}",
            )

        hits: list[AdTemplateHit] = []
        for point in result.points:
            payload = point.payload or {}
            body = str(payload.get("body") or "")
            if not body:
                continue
            hits.append(
                AdTemplateHit(
                    id=str(payload.get("templateId") or point.id),
                    name=str(payload.get("name") or "template"),
                    channel=payload.get("channel"),
                    framework=payload.get("framework"),
                    tone=payload.get("tone"),
                    body=body,
                    score=float(point.score or 0.0),
                    entity_tags=list(payload.get("entityTags") or []),
                )
            )

        warning = None if hits else "No ad templates matched; continue without few-shot exemplars."
        return AdTemplateQueryResponse(templates=hits, warning=warning)


def normalize_upsert_items(raw: list[AdTemplateUpsertItem]) -> list[AdTemplateUpsertItem]:
    out: list[AdTemplateUpsertItem] = []
    seen: set[str] = set()
    for item in raw:
        tid = (item.id or "").strip()
        body = (item.body or "").strip()
        if len(body) < 8:
            continue
        if not tid:
            tid = str(uuid.uuid4())
        if tid.lower() in seen:
            continue
        seen.add(tid.lower())
        out.append(item.model_copy(update={"id": tid, "body": body}))
        if len(out) >= 50:
            break
    return out
