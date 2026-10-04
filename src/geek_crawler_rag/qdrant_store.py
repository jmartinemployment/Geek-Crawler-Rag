"""Qdrant collection ops for geek_crawler_chunks."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from geek_crawler_rag.context_models import ManifestEntry

logger = logging.getLogger(__name__)

# Stable namespace for deterministic point IDs.
_POINT_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

#: Named sparse vector carrying SPLADE token weights for hybrid retrieval. The name is part of the
#: on-disk contract: llama_engine binds its sparse branch to it, and scripts/migrate_sparse_vectors.py
#: backfills it, so all three have to agree or a hybrid query scores against a vector nothing wrote.
SPARSE_VECTOR_NAME = "text-sparse"


def point_id(run_id: str, page_id: str, chunk_key: int | str) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{run_id}:{page_id}:{chunk_key}"))


def _is_missing_collection(exc: BaseException, collection: str) -> bool:
    """True when `exc` is Qdrant saying `collection` is not there.

    Status and message are both checked because neither is sufficient alone: the client raises
    UnexpectedResponse with 404 for a dropped collection, but the wording of the body
    ("Collection `x` doesn't exist!") has changed between server versions, and a transport wrapping
    the error can lose the status while keeping the text.
    """
    if getattr(exc, "status_code", None) == 404:
        return True
    body = getattr(exc, "content", b"") or b""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    lowered = f"{body} {exc}".lower()
    return collection.lower() in lowered and (
        "doesn't exist" in lowered or "does not exist" in lowered or "not found" in lowered
    )


async def find_existing_point_ids(
    client: AsyncQdrantClient,
    collection: str,
    ids: list[str],
    *,
    batch: int = 256,
) -> set[str]:
    """Return the subset of `ids` already present in `collection`.

    Point IDs are deterministic (see `point_id`), so a re-run can skip chunks
    it already committed instead of re-embedding them. Payload and vectors are
    left off so this stays cheap.
    """
    found: set[str] = set()
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        if not chunk:
            continue
        records = await client.retrieve(
            collection_name=collection,
            ids=chunk,
            with_payload=False,
            with_vectors=False,
        )
        found.update(str(rec.id) for rec in records)
    return found


class QdrantStore:
    def __init__(
        self,
        url: str,
        *,
        collection: str = "geek_crawler_chunks",
        api_key: str | None = None,
        vector_size: int = 1536,
        # Seconds before a Qdrant call is abandoned. Defaulted rather than required so the many
        # construction sites in tests stay unchanged; app.py passes the configured value.
        timeout_seconds: int = 60,
    ) -> None:
        self._client = AsyncQdrantClient(
            url=url, api_key=api_key or None, timeout=timeout_seconds
        )
        self._collection = collection
        self._vector_size = vector_size

    async def close(self) -> None:
        await self._client.close()

    async def points_exist(
        self, ids: list[str], *, batch: int = 256
    ) -> set[str]:
        """Subset of `ids` already in this collection. See find_existing_point_ids."""
        return await find_existing_point_ids(
            self._client, self._collection, ids, batch=batch
        )

    async def ping(self) -> bool:
        """Qdrant is reachable. Says nothing about whether OUR collection is there."""
        await self._client.get_collections()
        return True

    async def collection_present(self) -> bool | None:
        """Whether this store's collection exists. None when the question can't be answered.

        Reachability and presence are different facts and `ping` only answers the first, so a
        dropped `geek_crawler_chunks` reported `qdrant: true, status: ok` -- the most damaging
        state in the system looking healthy. It is not hypothetical: the collection was deleted
        out from under a running API on 2026-09-24.

        It matters more now, not less. Until 2026-09-28 a missing collection was at least loud
        by accident, because every purge raised a 500 that GeekAPI turned into a 502.
        `delete_by_run_id` now correctly treats it as a satisfied delete -- there are no vectors
        for the run, which is what the caller asked for -- so the accidental alarm is gone and
        this is what replaces it.

        Silence here would be the worse failure: nothing tells a reader the corpus is empty,
        while `rag_index_jobs` still reports runs `complete` with their old chunk counts and
        every query returns nothing.

        None rather than False on error, because "I could not tell" is not "it is missing", and
        a transport blip must not be reported as a dropped collection.
        """
        try:
            return await self._client.collection_exists(self._collection)
        except Exception:
            logger.warning(
                "Could not determine whether collection %s exists", self._collection
            )
            return None

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._vector_size,
                    distance=qm.Distance.COSINE,
                    # float16 halves vector memory at ~no accuracy cost, and it is CREATION-ONLY:
                    # a PATCH of datatype returns 200 ok and silently changes nothing.
                    datatype=qm.Datatype.FLOAT16,
                    # In RAM. 384-d float16 at 500k points is ~380 MB against a 3 GiB container
                    # limit, so there is nothing to gain by making every search touch NVMe.
                    on_disk=False,
                ),
                # One unified schema, and creation is the only chance to declare it. The dense
                # vector stays unnamed -- Qdrant addresses it as "" and the LlamaIndex store detects
                # that and binds to LEGACY_UNNAMED_VECTOR, so nothing downstream needs to change --
                # while the sparse vector is named because sparse vectors have no unnamed form.
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                        index=qm.SparseIndexParams(on_disk=False),
                        # MANDATORY, and creation-only. fastembed's BM25 returns uniform values --
                        # measured: every one 1.597 -- because IDF is Qdrant's job, applied at query
                        # time through this modifier. Without it BM25 degrades to unweighted term
                        # matching, which discards exactly the rare-term weighting that made it the
                        # right sparse model: a product code like XJ-4420-B would score no higher
                        # than the word "the". Nothing errors; retrieval just quietly gets worse.
                        #
                        # It would be WRONG with a neural sparse model such as SPLADE, which carries
                        # its own learned weights -- IDF on top double-penalises. Modifier and model
                        # are one decision.
                        modifier=qm.Modifier.IDF,
                    ),
                },
                hnsw_config=qm.HnswConfigDiff(
                    m=16,
                    # 200 over the default 100: a one-time build cost for better recall, and recall
                    # here is whether a quote can be verified at all.
                    ef_construct=200,
                    full_scan_threshold=10000,
                    on_disk=False,
                ),
                optimizers_config=qm.OptimizersConfigDiff(
                    # 2 rather than auto (which resolves to the CPU count, 4): fewer, larger segments
                    # mean fewer HNSW graphs to traverse and merge per search. indexing_threshold is
                    # deliberately NOT set to 0 here -- that belongs to a deliberate bulk load, and a
                    # collection created by startup must be immediately searchable rather than
                    # silently unindexed until someone remembers to restore it.
                    default_segment_number=2,
                    max_optimization_threads=2,
                ),
                on_disk_payload=True,
            )
            logger.info(
                "Created Qdrant collection %s (size=%s float16 in RAM, sparse '%s' with IDF)",
                self._collection,
                self._vector_size,
                SPARSE_VECTOR_NAME,
            )
        else:
            await self._verify_collection_matches_intent()
            await self._drop_body_text_indexes()
        await self._ensure_payload_indexes()

    async def _verify_collection_matches_intent(self) -> None:
        """Refuse to run against a collection whose creation-only settings are wrong.

        Replaced ``_ensure_vectors_on_disk``, which forced ``on_disk=True`` on every startup. That was
        right when vectors were 1536-d float32 on a RAM-constrained host; at 384-d float16 it would
        now fight the intended configuration once per boot.

        What matters more is that two settings **cannot be repaired**: ``datatype`` and the sparse
        ``modifier``. A ``PATCH`` of ``datatype`` returns ``200 ok`` and changes nothing, and Qdrant
        will not add or alter a sparse modifier after creation. So a collection created without them
        is not a degraded collection to be nudged into shape -- it is one that has to be rebuilt, and
        the only thing worse than finding that out is not finding out.

        Size and modifier raise. A width mismatch fails every upsert anyway, so failing at startup
        just moves the error to where the cause is legible. A missing IDF modifier is subtler and
        worse: nothing errors, BM25 simply stops weighting rare terms, and the product codes the
        sparse channel exists to bind score no higher than "the". Per ``plans/rules.md`` §3a that is
        exactly what must not degrade silently.

        datatype and on_disk only log. They cost memory or a page-cache hop; they do not change which
        passages come back.
        """
        try:
            info = await self._client.get_collection(self._collection)
        except Exception:
            logger.warning(
                "Could not read %s to verify its configuration", self._collection, exc_info=True
            )
            return

        params = info.config.params
        vectors = params.vectors
        if isinstance(vectors, dict):
            # Named dense vectors -- not this collection's shape. Nothing to compare.
            return

        if vectors is not None and int(vectors.size) != int(self._vector_size):
            raise ValueError(
                f"Qdrant collection {self._collection} declares size={vectors.size} but this "
                f"service embeds at {self._vector_size}. Every upsert would be rejected. The "
                f"collection must be recreated -- size is fixed at creation."
            )

        sparse = params.sparse_vectors or {}
        sparse_params = sparse.get(SPARSE_VECTOR_NAME)
        if sparse_params is None:
            raise ValueError(
                f"Qdrant collection {self._collection} has no '{SPARSE_VECTOR_NAME}' sparse vector. "
                f"Qdrant cannot add one after creation, so hybrid retrieval is impossible on this "
                f"collection and it must be recreated."
            )
        if sparse_params.modifier != qm.Modifier.IDF:
            raise ValueError(
                f"Qdrant collection {self._collection} sparse vector '{SPARSE_VECTOR_NAME}' has "
                f"modifier={sparse_params.modifier!r}, not IDF. BM25 would run unweighted: rare "
                f"terms such as product codes would score no higher than common words, silently. "
                f"The modifier is creation-only, so the collection must be recreated."
            )

        if vectors is not None and vectors.datatype not in (None, qm.Datatype.FLOAT16):
            logger.error(
                "Qdrant collection %s stores vectors as %s, not float16. Twice the memory for no "
                "accuracy gain, and datatype is creation-only -- a PATCH reports success and does "
                "nothing. Fixed only by recreating.",
                self._collection,
                vectors.datatype,
            )
        if vectors is not None and vectors.on_disk:
            logger.info(
                "Qdrant collection %s keeps vectors on disk; at this width they fit in RAM. "
                "Patchable, unlike datatype and modifier.",
                self._collection,
            )

    async def _drop_body_text_indexes(self) -> None:
        """Remove heavy TEXT indexes on long body fields to free RAM/optimizer work."""
        for field in ("text", "childText", "parentText"):
            try:
                await self._client.delete_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                )
                logger.info(
                    "Dropped TEXT payload index %s on %s",
                    field,
                    self._collection,
                )
            except Exception:
                logger.debug(
                    "TEXT index %s absent or drop skipped on %s",
                    field,
                    self._collection,
                )

    async def _ensure_payload_indexes(self) -> None:
        keyword_fields = (
            "runId",
            "crawlType",
            "host",
            "ownerId",
            "visibility",
            "manifestEligible",
            "assetId",
            "assetVersionId",
            "resourceId",
            "resourceDigest",
            "sourceDigest",
            "lifecycle",
            "freshness",
            "parserId",
            "parserVersion",
            "chunkerId",
            "chunkerVersion",
            "embeddingModel",
            "chunkId",
            "language",
            "chunkRole",
            "sourceType",
            "entityName",
            "category",
            "pageId",
        )
        for field in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug(
                    "Payload index %s already present or create skipped", field
                )

        # Do not create TEXT indexes on long body fields (text/childText/parentText).
        # They dominate RAM/optimizer cost; dense retrieval + keyword filters suffice.

        try:
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="qualityScore",
                field_schema=qm.PayloadSchemaType.FLOAT,
            )
        except Exception:
            logger.debug("Payload index qualityScore already present or create skipped")

    async def delete_by_run_id(
        self,
        run_id: str,
        *,
        owner_id: str = "system:crawler",
        visibility: str = "service",
    ) -> None:
        """Delete every point for one run. A missing collection is success; nothing else is.

        Two absences look alike and are not:

        * **No points match the filter.** Already success -- a Qdrant filter delete that
          matches nothing is not an error. This never needed fixing, and believing it did
          is what made the 2026-09-24 diagnosis wrong: eleven runs were reported failed and
          "there are no points for this run" was blamed, but that case returns 204 today
          and did then.
        * **No collection at all.** This is what actually raised. `geek_crawler_chunks` was
          dropped out from under a running API on 2026-09-24; every purge then 500ed, which
          GeekAPI turned into a 502 on the PATCH that was publishing a *different*,
          perfectly good run. With no collection there are no vectors for this run, which
          is precisely the state the caller asked for, so it is success.

        `find_host_index_payload` already reached that conclusion for the same incident.
        This is the same judgement on the write path.

        Everything else re-raises, and that distinction is the whole point. On a read, fail
        closed means answering "not indexed"; on a delete it means reporting failure --
        because a caller told a purge succeeded when it did not will go on to delete the
        pages those surviving vectors cite, and orphaned vectors are citable: every
        retrieval is filtered by a single runId, and `/v1/index/hosts` resolves that runId
        by scrolling Qdrant on host alone. A stale run's points can therefore be selected
        as a host's grounding evidence.
        """
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="runId",
                                match=qm.MatchValue(value=run_id),
                            ),
                            qm.FieldCondition(
                                key="ownerId", match=qm.MatchValue(value=owner_id)
                            ),
                            qm.FieldCondition(
                                key="visibility",
                                match=qm.MatchValue(value=visibility),
                            ),
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            # Broad, then narrowed by _is_missing_collection, because the error arrives as
            # UnexpectedResponse(404) from the client and as a message-only error through a
            # transport that wraps it -- the same two shapes that helper already tests for.
            if _is_missing_collection(exc, self._collection):
                logger.warning(
                    "Collection %s is missing; runId=%s has no points to purge",
                    self._collection,
                    run_id,
                )
                return
            raise
        logger.info("Deleted Qdrant points for runId=%s", run_id)

    async def delete_by_page_id(
        self,
        page_id: str,
        *,
        owner_id: str = "system:crawler",
        visibility: str = "service",
    ) -> None:
        if not page_id:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="pageId",
                            match=qm.MatchValue(value=page_id),
                        ),
                        qm.FieldCondition(
                            key="ownerId", match=qm.MatchValue(value=owner_id)
                        ),
                        qm.FieldCondition(
                            key="visibility",
                            match=qm.MatchValue(value=visibility),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    async def delete_asset_revision(self, entry: ManifestEntry) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="ownerId", match=qm.MatchValue(value=entry.owner_id)
                        ),
                        qm.FieldCondition(
                            key="assetId", match=qm.MatchValue(value=entry.asset_id)
                        ),
                        qm.FieldCondition(
                            key="assetVersionId",
                            match=qm.MatchValue(value=entry.asset_version_id),
                        ),
                        qm.FieldCondition(
                            key="resourceId",
                            match=qm.MatchValue(value=entry.resource_id),
                        ),
                        qm.FieldCondition(
                            key="resourceDigest",
                            match=qm.MatchValue(value=entry.resource_digest),
                        ),
                        qm.FieldCondition(
                            key="sourceDigest",
                            match=qm.MatchValue(value=entry.source_digest),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    async def delete_trusted_asset_revision(
        self, *, owner_id: str, asset_version_id: str, resource_id: str
    ) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="ownerId", match=qm.MatchValue(value=owner_id)
                        ),
                        qm.FieldCondition(
                            key="assetVersionId",
                            match=qm.MatchValue(value=asset_version_id),
                        ),
                        qm.FieldCondition(
                            key="resourceId",
                            match=qm.MatchValue(value=resource_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    async def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must be the same length")
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

    def build_filter(
        self,
        *,
        run_id: str,
        owner_id: str = "system:crawler",
        visibility: str = "service",
        crawl_type: str | None = None,
        host: str | None = None,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
    ) -> qm.Filter:
        must: list[qm.Condition] = [
            qm.FieldCondition(key="runId", match=qm.MatchValue(value=run_id)),
            qm.FieldCondition(key="ownerId", match=qm.MatchValue(value=owner_id)),
            qm.FieldCondition(key="visibility", match=qm.MatchValue(value=visibility)),
            qm.FieldCondition(key="language", match=qm.MatchValue(value="en")),
        ]
        if crawl_type:
            must.append(
                qm.FieldCondition(
                    key="crawlType",
                    match=qm.MatchValue(value=crawl_type),
                )
            )
        if host:
            must.append(
                qm.FieldCondition(
                    key="host",
                    match=qm.MatchValue(value=host.lower().strip()),
                )
            )
        if chunk_role:
            must.append(
                qm.FieldCondition(
                    key="chunkRole",
                    match=qm.MatchValue(value=chunk_role),
                )
            )
        if source_types:
            must.append(
                qm.FieldCondition(
                    key="sourceType",
                    match=qm.MatchAny(
                        any=[s.strip().lower() for s in source_types if s]
                    ),
                )
            )
        if entity_names:
            must.append(
                qm.FieldCondition(
                    key="entityName",
                    match=qm.MatchAny(any=[n.strip() for n in entity_names if n]),
                )
            )
        if categories:
            must.append(
                qm.FieldCondition(
                    key="category",
                    match=qm.MatchAny(any=[c.strip().lower() for c in categories if c]),
                )
            )
        if min_quality is not None:
            must.append(
                qm.FieldCondition(
                    key="qualityScore",
                    range=qm.Range(gte=float(min_quality)),
                )
            )
        return qm.Filter(must=must)

    def build_asset_filter(
        self, *, owner_id: str, entries: list[ManifestEntry]
    ) -> qm.Filter:
        if not entries:
            raise ValueError("A verified manifest must contain allowed entries.")
        exact_entries: list[qm.Condition] = []
        for entry in entries:
            if entry.owner_id != owner_id:
                raise ValueError("Manifest entry owner mismatch.")
            exact_entries.append(
                qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="assetId", match=qm.MatchValue(value=entry.asset_id)
                        ),
                        qm.FieldCondition(
                            key="assetVersionId",
                            match=qm.MatchValue(value=entry.asset_version_id),
                        ),
                        qm.FieldCondition(
                            key="resourceId",
                            match=qm.MatchValue(value=entry.resource_id),
                        ),
                        qm.FieldCondition(
                            key="resourceDigest",
                            match=qm.MatchValue(value=entry.resource_digest),
                        ),
                        qm.FieldCondition(
                            key="sourceDigest",
                            match=qm.MatchValue(value=entry.source_digest),
                        ),
                        qm.FieldCondition(
                            key="visibility",
                            match=qm.MatchValue(value=entry.visibility.value),
                        ),
                    ]
                )
            )
        return qm.Filter(
            must=[
                qm.FieldCondition(key="ownerId", match=qm.MatchValue(value=owner_id)),
                qm.FieldCondition(
                    key="manifestEligible", match=qm.MatchValue(value=True)
                ),
            ],
            should=exact_entries,
            min_should=qm.MinShould(conditions=exact_entries, min_count=1),
        )

    async def search_assets(
        self,
        vector: list[float],
        *,
        owner_id: str,
        entries: list[ManifestEntry],
        top_k: int,
    ) -> list[qm.ScoredPoint]:
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=self.build_asset_filter(owner_id=owner_id, entries=entries),
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)

    async def search_asset_versions(
        self,
        vector: list[float],
        *,
        owner_id: str,
        allowed_versions: list[tuple[str, str]],
        top_k: int,
    ) -> list[qm.ScoredPoint]:
        if not allowed_versions:
            return []
        exact_versions: list[qm.Condition] = [
            qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="assetId", match=qm.MatchValue(value=asset_id)
                    ),
                    qm.FieldCondition(
                        key="assetVersionId",
                        match=qm.MatchValue(value=asset_version_id),
                    ),
                ]
            )
            for asset_id, asset_version_id in allowed_versions
        ]
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="ownerId", match=qm.MatchValue(value=owner_id)
                    ),
                    qm.FieldCondition(
                        key="manifestEligible", match=qm.MatchValue(value=True)
                    ),
                ],
                should=exact_versions,
                min_should=qm.MinShould(conditions=exact_versions, min_count=1),
            ),
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)

    async def search(
        self,
        vector: list[float],
        *,
        run_id: str,
        owner_id: str = "system:crawler",
        visibility: str = "service",
        crawl_type: str | None = None,
        host: str | None = None,
        top_k: int = 8,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
    ) -> list[qm.ScoredPoint]:
        query_filter = self.build_filter(
            run_id=run_id,
            owner_id=owner_id,
            visibility=visibility,
            crawl_type=crawl_type,
            host=host,
            chunk_role=chunk_role,
            source_types=source_types,
            entity_names=entity_names,
            categories=categories,
            min_quality=min_quality,
        )
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)

    async def search_text(
        self,
        text: str,
        *,
        run_id: str,
        owner_id: str = "system:crawler",
        visibility: str = "service",
        crawl_type: str | None = None,
        host: str | None = None,
        top_k: int = 8,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
        text_fields: tuple[str, ...] = ("childText", "text"),
    ) -> list[qm.ScoredPoint]:
        """Keyword/full-text style retrieval via payload TEXT indexes."""
        base = self.build_filter(
            run_id=run_id,
            owner_id=owner_id,
            visibility=visibility,
            crawl_type=crawl_type,
            host=host,
            chunk_role=chunk_role,
            source_types=source_types,
            entity_names=entity_names,
            categories=categories,
            min_quality=min_quality,
        )
        should = [
            qm.FieldCondition(key=field, match=qm.MatchText(text=text))
            for field in text_fields
        ]
        query_filter = qm.Filter(must=list(base.must or []), should=should)
        try:
            # Prefer scroll+filter when no sparse vector; use dummy dense if needed.
            # query_points without vector uses filter-only in recent clients via scroll.
            points, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            scored: list[qm.ScoredPoint] = []
            for i, point in enumerate(points):
                scored.append(
                    qm.ScoredPoint(
                        id=point.id,
                        version=getattr(point, "version", 0) or 0,
                        score=float(top_k - i),
                        payload=point.payload,
                        vector=None,
                    )
                )
            return scored
        except Exception:
            # Re-raised, not turned into []. An empty scroll result and a broken
            # scroll are different answers, and returning [] made them one: the
            # query lost its lexical half, carried on with dense hits alone, and
            # still reported retrieval as hybrid. QueryService already converts a
            # retrieval exception into retrieval="error" with a stated warning,
            # so the clean failure state exists - swallowing here is what kept
            # the boundary from ever seeing it.
            logger.exception("Text search scroll failed for runId=%s", run_id)
            raise

    async def run_text_owners(self, run_id: str) -> tuple[dict[str, str], int]:
        """`textDigest -> pageId` for every point this run already wrote, and how many lacked one.

        This is how a resumed index run rebuilds its repeat set: the set lives in memory during a
        run, and a retry that skips the delete starts with an empty one, so without this it would
        write a second copy of every repeat whose first page was committed before the failure.

        Filtered on runId alone, the same key every retrieval filters on. A point with no
        `textDigest` was written before the field existed; it is counted and returned so the
        caller can say so, not silently treated as unique.

        Raises on any Qdrant failure. An empty answer would mean "nothing was written yet", and
        the indexer would act on that by writing the copies this set exists to prevent.
        """
        owners: dict[str, str] = {}
        missing = 0
        offset: Any = None
        run_filter = qm.Filter(
            must=[qm.FieldCondition(key="runId", match=qm.MatchValue(value=run_id))]
        )
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=run_filter,
                limit=1000,
                offset=offset,
                with_payload=["textDigest", "pageId"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                digest = payload.get("textDigest")
                if not digest:
                    missing += 1
                    continue
                owners.setdefault(str(digest), str(payload.get("pageId") or ""))
            if offset is None:
                return owners, missing

    async def find_host_index_payload(self, host: str) -> dict[str, Any] | None:
        """
        The raw payload of an indexed chunk for a host, if any exists, else None.

        Whether, not how much: one point is fetched, not counted. A count would invite a threshold
        ("is 46 chunks enough?"), which is a different question. A host has one active run at a
        time, so the first match's payload is already the whole answer.

        Returns the full, unfiltered payload rather than a derived bool or a bare id -- one value
        standing in for two different questions ("does it exist" vs "what is it") is how that
        conflation bug gets written. Callers derive both facts explicitly from this one value.

        Filtered on host alone. Crawler-written chunks carry no ownerId or visibility -- both are
        absent on every crawl_pages-derived point -- so adding those conditions, as the delete paths
        do, matches nothing and reports every host unindexed.
        """
        if not host:
            return None

        try:
            points, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="host", match=qm.MatchValue(value=host))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            if _is_missing_collection(exc, self._collection):
                # A dropped collection is not an error to the caller: with no index, no host is
                # indexed, which is exactly what None already means here. Raising instead turned
                # /v1/index/hosts into a 500 and every consumer of it into a 502 on
                # 2026-09-24, when the collection was deleted out from under a running API.
                logger.warning(
                    "Collection %s is missing; reporting host=%s as unindexed",
                    self._collection,
                    host,
                )
                return None
            logger.exception("Host index lookup failed for host=%s", host)
            return None
        except Exception:
            # Same clean "not indexed" state as the missing-collection case, and
            # for the same reason: the caller withholds this host as grounding
            # rather than reaching for something else, and raising here is what
            # turned /v1/index/hosts into a 500 and every consumer into a 502 on
            # 2026-09-24. Asserted by test_any_other_failure_also_fails_closed.
            logger.exception("Host index lookup failed for host=%s", host)
            return None

        return points[0].payload if points else None
