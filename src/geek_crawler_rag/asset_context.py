"""Text asset parsing, deterministic chunking, indexing, and manifest retrieval."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from bs4 import BeautifulSoup

from geek_crawler_rag.chunk import chunk_text
from geek_crawler_rag.config import Settings
from geek_crawler_rag.context_models import (
    AssetDeleteRequest,
    AssetEvidence,
    AssetIndexRequest,
    AssetIndexResponse,
    ManifestEntry,
    ManifestQueryRequest,
    ManifestQueryResponse,
    RuntimeManifestQueryRequest,
    SourceCoordinates,
    TrustedAssetDeleteRequest,
    TrustedAssetIndexRequest,
    verify_manifest,
    verify_persisted_manifest,
)
from geek_crawler_rag.qdrant_store import QdrantStore

PARSER_VERSION = "1.0.0"
CHUNKER_ID = "token-window"
CHUNKER_VERSION = "1.0.0"
RETRIEVAL_POLICY_VERSION = "manifest-dense.v1"
_ASSET_POINT_NS = uuid.UUID("7c294130-c873-4ef4-9193-b51635469d3d")
_WS = re.compile(r"[ \t]+")


class AssetEmbedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class ParsedAsset:
    text: str
    parser_id: str
    parser_version: str


@dataclass(frozen=True)
class AssetChunk:
    id: str
    text: str
    coordinates: SourceCoordinates


def parse_asset(
    content: str, media_type: str, *, max_chars: int = 2_000_000
) -> ParsedAsset:
    """Dispatch only locally parsed UTF-8 text, Markdown, and HTML."""
    if len(content) > max_chars:
        raise ValueError("Asset exceeds the extracted character limit.")
    if "\x00" in content:
        raise ValueError("Asset contains forbidden NUL characters.")
    if media_type == "text/html":
        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        parser_id = "beautifulsoup-lxml"
    elif media_type == "text/markdown":
        text = content
        parser_id = "markdown-utf8"
    elif media_type == "text/plain":
        text = content
        parser_id = "plain-utf8"
    else:
        raise ValueError("Unsupported asset media type.")
    normalized = "\n".join(
        _WS.sub(" ", line).strip() for line in text.replace("\r\n", "\n").splitlines()
    ).strip()
    if not normalized:
        raise ValueError("Asset has no extractable text.")
    return ParsedAsset(normalized, parser_id, PARSER_VERSION)


def deterministic_asset_chunk_id(
    resource_digest: str, coordinates: SourceCoordinates
) -> str:
    coordinate_key = (
        f"{coordinates.kind}:{coordinates.start_char}:{coordinates.end_char}:"
        f"{coordinates.page}:{coordinates.slide}:{coordinates.sheet}:"
        f"{coordinates.cell_range}:{coordinates.image_region}:"
        f"{coordinates.start_ms}:{coordinates.end_ms}"
    )
    return str(uuid.uuid5(_ASSET_POINT_NS, f"{resource_digest}:{coordinate_key}"))


def chunk_asset(
    text: str,
    resource_digest: str,
    *,
    size_tokens: int,
    overlap_tokens: int,
) -> list[AssetChunk]:
    pieces = chunk_text(text, size_tokens=size_tokens, overlap_tokens=overlap_tokens)
    chunks: list[AssetChunk] = []
    search_from = 0
    for piece in pieces:
        start = text.find(piece, search_from)
        if start < 0:
            start = text.find(piece)
        if start < 0:
            raise ValueError("Could not map chunk to source coordinates.")
        end = start + len(piece)
        coordinates = SourceCoordinates(kind="text", startChar=start, endChar=end)
        chunks.append(
            AssetChunk(
                id=deterministic_asset_chunk_id(resource_digest, coordinates),
                text=piece,
                coordinates=coordinates,
            )
        )
        search_from = min(end, start + max(1, len(piece) // 2))
    return chunks


class AssetContextService:
    def __init__(
        self, store: QdrantStore, embedder: AssetEmbedder, settings: Settings
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    def _verify(
        self,
        request: ManifestQueryRequest | AssetDeleteRequest | AssetIndexRequest,
    ) -> None:
        verify_manifest(request.manifest, self._settings.context_manifest_signing_keys)

    async def index(self, request: AssetIndexRequest) -> AssetIndexResponse:
        self._verify(request)
        parsed = parse_asset(request.resource.content, request.resource.media_type)
        if (
            request.resource.parser_id != parsed.parser_id
            or request.resource.parser_version != parsed.parser_version
        ):
            raise ValueError("Declared parser identity does not match parser dispatch.")
        chunks = chunk_asset(
            parsed.text,
            request.resource.resource_digest,
            size_tokens=self._settings.child_chunk_size_tokens,
            overlap_tokens=self._settings.child_chunk_overlap_tokens,
        )
        vectors = await self._embedder.embed_texts([chunk.text for chunk in chunks])
        revision = request.revision
        asset = revision.asset
        payloads = [
            {
                "ownerId": asset.owner_id,
                "visibility": asset.visibility.value,
                "manifestEligible": True,
                "assetId": asset.asset_id,
                "assetVersionId": revision.asset_version_id,
                "resourceId": request.resource.resource_id,
                "resourceDigest": request.resource.resource_digest,
                "sourceDigest": revision.source_digest,
                "sourceReference": request.resource.source_reference,
                "language": revision.language,
                "lifecycle": revision.lifecycle.value,
                "freshness": revision.freshness,
                "parserId": parsed.parser_id,
                "parserVersion": parsed.parser_version,
                "chunkerId": CHUNKER_ID,
                "chunkerVersion": CHUNKER_VERSION,
                "embeddingModel": self._settings.openai_embedding_model,
                "chunkId": chunk.id,
                "coordinates": _source_coordinate(
                    chunk.coordinates, request.resource.source_coordinates
                ).model_dump(by_alias=True, mode="json", exclude_none=True),
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        await self._store.upsert(
            ids=[chunk.id for chunk in chunks], vectors=vectors, payloads=payloads
        )
        return AssetIndexResponse(
            assetVersionId=revision.asset_version_id,
            resourceId=request.resource.resource_id,
            chunksUpserted=len(chunks),
        )

    async def index_trusted(
        self, request: TrustedAssetIndexRequest
    ) -> AssetIndexResponse:
        """Index one GeekAPI-authoritative revision after service authentication."""
        parsed = parse_asset(request.content, request.media_type)
        chunks = chunk_asset(
            parsed.text,
            request.derived_sha256,
            size_tokens=self._settings.child_chunk_size_tokens,
            overlap_tokens=self._settings.child_chunk_overlap_tokens,
        )
        vectors = await self._embedder.embed_texts([chunk.text for chunk in chunks])
        payloads = [
            {
                "ownerId": request.owner_user_id,
                "contextKind": request.context_kind,
                "visibility": request.visibility.value,
                "manifestEligible": True,
                "assetId": request.asset_id,
                "assetVersionId": request.asset_version_id,
                "resourceId": request.resource_id,
                "resourceDigest": request.derived_sha256,
                "sourceDigest": request.source_sha256,
                "sourceReference": request.object_key,
                "language": request.language,
                "lifecycle": request.lifecycle.value,
                "freshness": "unknown",
                "parserId": request.parser_name,
                "parserVersion": request.parser_version,
                "chunkerId": CHUNKER_ID,
                "chunkerVersion": CHUNKER_VERSION,
                "embeddingModel": self._settings.openai_embedding_model,
                "chunkId": chunk.id,
                "coordinates": _source_coordinate(
                    chunk.coordinates, request.source_coordinates
                ).model_dump(by_alias=True, mode="json", exclude_none=True),
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        await self._store.upsert(
            ids=[chunk.id for chunk in chunks], vectors=vectors, payloads=payloads
        )
        return AssetIndexResponse(
            assetVersionId=request.asset_version_id,
            resourceId=request.resource_id,
            chunksUpserted=len(chunks),
        )

    async def delete_trusted(self, request: TrustedAssetDeleteRequest) -> None:
        await self._store.delete_trusted_asset_revision(
            owner_id=request.owner_user_id,
            asset_version_id=request.asset_version_id,
            resource_id=request.resource_id,
        )

    async def delete(self, request: AssetDeleteRequest) -> None:
        self._verify(request)
        entry = _exact_entry(
            request.manifest.entries,
            request.asset_version_id,
            request.resource_id,
        )
        await self._store.delete_asset_revision(entry)

    async def query(self, request: ManifestQueryRequest) -> ManifestQueryResponse:
        self._verify(request)
        vector = await self._embedder.embed_query(request.need)
        dense = await self._store.search_assets(
            vector,
            owner_id=request.manifest.owner_id,
            entries=request.manifest.entries,
            top_k=request.top_k,
        )
        evidence = _build_evidence(dense, request.need)
        for point in dense:
            payload = point.payload or {}
            _assert_payload_in_manifest(payload, request.manifest.entries)
        return ManifestQueryResponse(
            manifestId=request.manifest.manifest_id, evidence=evidence
        )

    async def query_runtime(
        self, request: RuntimeManifestQueryRequest
    ) -> ManifestQueryResponse:
        manifest = verify_persisted_manifest(
            request.manifest, self._settings.context_manifest_signing_keys
        )
        if not manifest.policies.knowledge_search_enabled:
            return ManifestQueryResponse(manifestId=manifest.manifest_id, evidence=[])
        allowed: list[tuple[str, str]] = []
        for entry in manifest.entries:
            if entry.context_kind == "knowledge" and entry.version_id is not None:
                allowed.append((entry.stable_id, entry.version_id))
            elif entry.context_kind == "run_attachment":
                allowed.append((entry.stable_id, entry.stable_id))
        vector = await self._embedder.embed_query(request.need)
        dense = await self._store.search_asset_versions(
            vector,
            owner_id=manifest.owner_user_id,
            allowed_versions=allowed,
            top_k=request.top_k,
        )
        allowed_set = set(allowed)
        for point in dense:
            payload = point.payload or {}
            if (
                payload.get("ownerId") != manifest.owner_user_id
                or (payload.get("assetId"), payload.get("assetVersionId"))
                not in allowed_set
                or payload.get("manifestEligible") is not True
            ):
                raise ValueError(
                    "Retrieved point does not match the verified runtime manifest."
                )
        return ManifestQueryResponse(
            manifestId=manifest.manifest_id,
            evidence=_build_evidence(dense, request.need),
        )


def _exact_entry(
    entries: list[ManifestEntry], asset_version_id: str, resource_id: str
) -> ManifestEntry:
    matches = [
        entry
        for entry in entries
        if entry.asset_version_id == asset_version_id
        and entry.resource_id == resource_id
    ]
    if len(matches) != 1:
        raise ValueError("Asset revision is not an exact manifest entry.")
    return matches[0]


def _source_coordinate(
    chunk: SourceCoordinates, sources: list[SourceCoordinates]
) -> SourceCoordinates:
    start = chunk.start_char or 0
    source = next(
        (
            candidate
            for candidate in sources
            if candidate.start_char is not None
            and candidate.end_char is not None
            and candidate.start_char <= start < candidate.end_char
        ),
        None,
    )
    if source is None:
        return chunk
    return source.model_copy(
        update={"start_char": chunk.start_char, "end_char": chunk.end_char}
    )


def _assert_payload_in_manifest(
    payload: dict[str, object], entries: list[ManifestEntry]
) -> None:
    match = next(
        (
            entry
            for entry in entries
            if entry.asset_version_id == payload.get("assetVersionId")
            and entry.resource_id == payload.get("resourceId")
        ),
        None,
    )
    if (
        match is None
        or match.owner_id != payload.get("ownerId")
        or match.asset_id != payload.get("assetId")
        or match.resource_digest != payload.get("resourceDigest")
        or match.source_digest != payload.get("sourceDigest")
        or match.visibility.value != payload.get("visibility")
        or payload.get("manifestEligible") is not True
    ):
        raise ValueError("Retrieved point does not match the verified manifest entry.")


def _build_evidence(points: list[object], need: str) -> list[AssetEvidence]:
    evidence: list[AssetEvidence] = []
    for rank, point in enumerate(points, 1):
        payload = point.payload or {}  # type: ignore[attr-defined]
        evidence.append(
            AssetEvidence(
                pointId=str(point.id),  # type: ignore[attr-defined]
                chunkId=str(payload["chunkId"]),
                ownerId=str(payload["ownerId"]),
                assetId=str(payload["assetId"]),
                assetVersionId=str(payload["assetVersionId"]),
                resourceId=str(payload["resourceId"]),
                resourceDigest=str(payload["resourceDigest"]),
                sourceDigest=str(payload["sourceDigest"]),
                sourceReference=str(payload["sourceReference"]),
                coordinates=SourceCoordinates.model_validate(payload["coordinates"]),
                text=str(payload["text"]),
                language=str(payload["language"]),
                parserId=str(payload["parserId"]),
                parserVersion=str(payload["parserVersion"]),
                chunkerId=str(payload["chunkerId"]),
                chunkerVersion=str(payload["chunkerVersion"]),
                embeddingModel=str(payload["embeddingModel"]),
                retrievalPolicyVersion=RETRIEVAL_POLICY_VERSION,
                retrievalQuery=need,
                denseScore=float(point.score),  # type: ignore[attr-defined]
                lexicalScore=None,
                rerankScore=None,
                rank=rank,
            )
        )
    return evidence
