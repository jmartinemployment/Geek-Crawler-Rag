"""Governed Knowledge contracts shared with the trusted GeekAPI caller.

The records here describe immutable authority owned by GeekRepository.  This
service verifies and indexes them, but never persists a second authoritative
catalog.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StrictModel(BaseModel):
    model_config = {
        "populate_by_name": True,
        "ser_json_by_alias": True,
        "extra": "forbid",
    }


class AssetVisibility(StrEnum):
    PRIVATE = "private"
    OWNER = "owner"


class AssetLifecycle(StrEnum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


class KnowledgeAsset(StrictModel):
    asset_id: str = Field(..., alias="assetId", min_length=1, max_length=120)
    owner_id: str = Field(..., alias="ownerId", min_length=1, max_length=120)
    name: str = Field(..., min_length=1, max_length=300)
    kind: str = Field(..., min_length=1, max_length=80)
    visibility: AssetVisibility = AssetVisibility.PRIVATE


class KnowledgeAssetRevision(StrictModel):
    asset: KnowledgeAsset
    asset_version_id: str = Field(
        ..., alias="assetVersionId", min_length=1, max_length=120
    )
    version_number: int = Field(..., alias="versionNumber", ge=1)
    source_digest: str = Field(..., alias="sourceDigest", pattern=r"^[0-9a-f]{64}$")
    lifecycle: AssetLifecycle
    language: str = Field("en", min_length=2, max_length=35)
    source_timestamp_utc: datetime | None = Field(None, alias="sourceTimestampUtc")
    freshness: Literal["fresh", "stale", "unknown"] = "unknown"

    @model_validator(mode="after")
    def eligible_revision(self) -> KnowledgeAssetRevision:
        if self.lifecycle == AssetLifecycle.REVOKED:
            raise ValueError("Revoked asset revisions cannot be indexed.")
        return self


class SourceCoordinates(StrictModel):
    kind: Literal["text", "page", "slide", "sheet", "image", "time"] = "text"
    start_char: int | None = Field(None, alias="startChar", ge=0)
    end_char: int | None = Field(None, alias="endChar", ge=0)
    page: int | None = Field(None, ge=1)
    slide: int | None = Field(None, ge=1)
    sheet: str | None = Field(None, max_length=120)
    cell_range: str | None = Field(None, alias="cellRange", max_length=80)
    image_region: str | None = Field(None, alias="imageRegion", max_length=120)
    start_ms: int | None = Field(None, alias="startMs", ge=0)
    end_ms: int | None = Field(None, alias="endMs", ge=0)

    @model_validator(mode="after")
    def ordered_ranges(self) -> SourceCoordinates:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("endChar must be greater than startChar.")
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms <= self.start_ms
        ):
            raise ValueError("endMs must be greater than startMs.")
        return self


class KnowledgeResource(StrictModel):
    resource_id: str = Field(..., alias="resourceId", min_length=1, max_length=120)
    resource_digest: str = Field(..., alias="resourceDigest", pattern=r"^[0-9a-f]{64}$")
    media_type: Literal["text/plain", "text/markdown", "text/html"] = Field(
        ..., alias="mediaType"
    )
    source_reference: str = Field(
        ..., alias="sourceReference", min_length=1, max_length=2_000
    )
    content: str = Field(..., min_length=1, max_length=2_000_000)
    parser_id: str = Field(..., alias="parserId", min_length=1, max_length=120)
    parser_version: str = Field(..., alias="parserVersion", min_length=1, max_length=40)
    source_coordinates: list[SourceCoordinates] = Field(
        default_factory=list, alias="sourceCoordinates", max_length=5_000
    )

    @model_validator(mode="after")
    def digest_matches_content(self) -> KnowledgeResource:
        actual = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(actual, self.resource_digest):
            raise ValueError("Resource content digest mismatch.")
        return self


class ManifestEntry(StrictModel):
    owner_id: str = Field(..., alias="ownerId", min_length=1, max_length=120)
    asset_id: str = Field(..., alias="assetId", min_length=1, max_length=120)
    asset_version_id: str = Field(
        ..., alias="assetVersionId", min_length=1, max_length=120
    )
    resource_id: str = Field(..., alias="resourceId", min_length=1, max_length=120)
    resource_digest: str = Field(..., alias="resourceDigest", pattern=r"^[0-9a-f]{64}$")
    source_digest: str = Field(..., alias="sourceDigest", pattern=r"^[0-9a-f]{64}$")
    lifecycle: AssetLifecycle
    visibility: AssetVisibility

    @model_validator(mode="after")
    def executable_entry(self) -> ManifestEntry:
        if self.lifecycle not in {
            AssetLifecycle.APPROVED,
            AssetLifecycle.DEPRECATED,
        }:
            raise ValueError(
                "Manifest can authorize only approved or deprecated revisions."
            )
        return self


class SignedRunContextManifest(StrictModel):
    schema_version: Literal["run-context-manifest.v1"] = Field(
        ..., alias="schemaVersion"
    )
    manifest_id: str = Field(..., alias="manifestId", min_length=1, max_length=120)
    job_id: str = Field(..., alias="jobId", min_length=1, max_length=120)
    attempt_id: str = Field(..., alias="attemptId", min_length=1, max_length=120)
    owner_id: str = Field(..., alias="ownerId", min_length=1, max_length=120)
    resolved_at_utc: datetime = Field(..., alias="resolvedAtUtc")
    expires_at_utc: datetime | None = Field(None, alias="expiresAtUtc")
    entries: list[ManifestEntry] = Field(..., min_length=1, max_length=500)
    digest: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    signature_key_id: str = Field(
        ..., alias="signatureKeyId", min_length=1, max_length=120
    )

    @model_validator(mode="after")
    def coherent_entries(self) -> SignedRunContextManifest:
        if any(entry.owner_id != self.owner_id for entry in self.entries):
            raise ValueError("Manifest entries must belong to the manifest owner.")
        identities = [
            (entry.asset_version_id, entry.resource_id, entry.resource_digest)
            for entry in self.entries
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Manifest entries must be unique.")
        if self.expires_at_utc and self.expires_at_utc <= self.resolved_at_utc:
            raise ValueError("Manifest expiry must follow resolution.")
        return self


def canonical_manifest_bytes(manifest: SignedRunContextManifest) -> bytes:
    payload = manifest.model_dump(
        by_alias=True,
        mode="json",
        exclude={"digest", "signature"},
    )
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def verify_manifest(
    manifest: SignedRunContextManifest,
    signing_keys: dict[str, str],
    *,
    now: datetime | None = None,
) -> None:
    key = signing_keys.get(manifest.signature_key_id)
    if not key:
        raise ValueError("Unknown manifest signature key.")
    canonical = canonical_manifest_bytes(manifest)
    digest = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(digest, manifest.digest):
        raise ValueError("Manifest digest verification failed.")
    expected = hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, manifest.signature):
        raise ValueError("Manifest signature verification failed.")
    current = now or datetime.now(UTC)
    if manifest.expires_at_utc and manifest.expires_at_utc < current:
        raise ValueError("Manifest has expired.")


class AssetIndexRequest(StrictModel):
    manifest: SignedRunContextManifest
    revision: KnowledgeAssetRevision
    resource: KnowledgeResource

    @model_validator(mode="after")
    def source_is_consistent(self) -> AssetIndexRequest:
        if self.resource.resource_digest != self.revision.source_digest:
            raise ValueError("Resource digest must match revision source digest.")
        matches = [
            entry
            for entry in self.manifest.entries
            if entry.asset_id == self.revision.asset.asset_id
            and entry.asset_version_id == self.revision.asset_version_id
            and entry.resource_id == self.resource.resource_id
            and entry.resource_digest == self.resource.resource_digest
            and entry.source_digest == self.revision.source_digest
            and entry.owner_id == self.revision.asset.owner_id
            and entry.visibility == self.revision.asset.visibility
        ]
        if len(matches) != 1:
            raise ValueError("Indexed revision must exactly match one manifest entry.")
        return self


class AssetIndexResponse(StrictModel):
    asset_version_id: str = Field(..., alias="assetVersionId")
    resource_id: str = Field(..., alias="resourceId")
    chunks_upserted: int = Field(..., alias="chunksUpserted", ge=0)


class TrustedAssetIndexRequest(StrictModel):
    """Service-authenticated immutable revision supplied by authoritative GeekAPI."""

    owner_user_id: str = Field(..., alias="ownerUserId", min_length=1, max_length=120)
    context_kind: Literal["knowledge", "run_attachment"] = Field(
        "knowledge", alias="contextKind"
    )
    asset_id: str = Field(..., alias="assetId", min_length=1, max_length=120)
    asset_version_id: str = Field(
        ..., alias="assetVersionId", min_length=1, max_length=120
    )
    resource_id: str = Field(..., alias="resourceId", min_length=1, max_length=120)
    source_sha256: str = Field(..., alias="sourceSha256", pattern=r"^[0-9a-f]{64}$")
    derived_sha256: str = Field(..., alias="derivedSha256", pattern=r"^[0-9a-f]{64}$")
    content: str = Field(..., min_length=1, max_length=2_000_000)
    object_key: str = Field(..., alias="objectKey", min_length=1, max_length=2_000)
    media_type: Literal["text/plain", "text/markdown", "text/html"] = Field(
        ..., alias="mediaType"
    )
    parser_name: str = Field(..., alias="parserName", min_length=1, max_length=120)
    parser_version: str = Field(..., alias="parserVersion", min_length=1, max_length=40)
    source_coordinates: list[SourceCoordinates] = Field(
        ..., alias="sourceCoordinates", min_length=1, max_length=5_000
    )
    language: str = Field("en", min_length=2, max_length=35)
    lifecycle: AssetLifecycle = AssetLifecycle.DRAFT
    visibility: AssetVisibility = AssetVisibility.OWNER

    @model_validator(mode="after")
    def content_digest_matches(self) -> TrustedAssetIndexRequest:
        actual = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(actual, self.derived_sha256):
            raise ValueError("Derived content digest mismatch.")
        if self.lifecycle == AssetLifecycle.REVOKED:
            raise ValueError("Revoked asset revisions cannot be indexed.")
        return self


class TrustedAssetDeleteRequest(StrictModel):
    owner_user_id: str = Field(..., alias="ownerUserId", min_length=1, max_length=120)
    asset_version_id: str = Field(
        ..., alias="assetVersionId", min_length=1, max_length=120
    )
    resource_id: str = Field(..., alias="resourceId", min_length=1, max_length=120)


class AssetDeleteRequest(StrictModel):
    manifest: SignedRunContextManifest
    asset_version_id: str = Field(..., alias="assetVersionId", min_length=1)
    resource_id: str = Field(..., alias="resourceId", min_length=1)


class ManifestQueryRequest(StrictModel):
    need: str = Field(..., min_length=1, max_length=2_000)
    manifest: SignedRunContextManifest
    top_k: int = Field(8, alias="topK", ge=1, le=50)


class AssetEvidence(StrictModel):
    point_id: str = Field(..., alias="pointId")
    chunk_id: str = Field(..., alias="chunkId")
    owner_id: str = Field(..., alias="ownerId")
    asset_id: str = Field(..., alias="assetId")
    asset_version_id: str = Field(..., alias="assetVersionId")
    resource_id: str = Field(..., alias="resourceId")
    resource_digest: str = Field(..., alias="resourceDigest")
    source_digest: str = Field(..., alias="sourceDigest")
    source_reference: str = Field(..., alias="sourceReference")
    coordinates: SourceCoordinates
    text: str
    language: str
    parser_id: str = Field(..., alias="parserId")
    parser_version: str = Field(..., alias="parserVersion")
    chunker_id: str = Field(..., alias="chunkerId")
    chunker_version: str = Field(..., alias="chunkerVersion")
    embedding_model: str = Field(..., alias="embeddingModel")
    retrieval_policy_version: str = Field(..., alias="retrievalPolicyVersion")
    retrieval_query: str = Field(..., alias="retrievalQuery")
    dense_score: float | None = Field(None, alias="denseScore")
    lexical_score: float | None = Field(None, alias="lexicalScore")
    rerank_score: float | None = Field(None, alias="rerankScore")
    rank: int = Field(..., ge=1)


class ManifestQueryResponse(StrictModel):
    manifest_id: str = Field(..., alias="manifestId")
    evidence: list[AssetEvidence]


class RuntimeManifestEntry(StrictModel):
    context_kind: str = Field(..., alias="contextKind", min_length=1, max_length=80)
    stable_id: str = Field(..., alias="stableId", min_length=1, max_length=120)
    version_id: str | None = Field(None, alias="versionId", max_length=120)
    version_number: int | None = Field(None, alias="versionNumber", ge=1)
    content_sha256: str = Field(..., alias="contentSha256", pattern=r"^[0-9a-f]{64}$")
    lifecycle_decision: str = Field(..., alias="lifecycleDecision")
    permission_decision: Literal["owner_allowed"] = Field(
        ..., alias="permissionDecision"
    )
    freshness_decision: str = Field(..., alias="freshnessDecision")
    selection_source: str = Field(..., alias="selectionSource")
    selected_field_ids: list[str] | None = Field(None, alias="selectedFieldIds")
    source_modified_at_utc: datetime | None = Field(None, alias="sourceModifiedAtUtc")

    @model_validator(mode="after")
    def eligible_at_resolution(self) -> RuntimeManifestEntry:
        if self.lifecycle_decision == "revoked" or self.freshness_decision == "expired":
            raise ValueError("Manifest entry is not eligible for runtime retrieval.")
        if self.context_kind == "knowledge" and not self.version_id:
            raise ValueError(
                "Knowledge manifest entries require an immutable version ID."
            )
        return self


class RuntimeManifestPolicies(StrictModel):
    web_search_enabled: bool = Field(..., alias="webSearchEnabled")
    knowledge_search_enabled: bool = Field(..., alias="knowledgeSearchEnabled")
    retrieval_policy: Literal["manifest-allowlist.v1"] = Field(
        ..., alias="retrievalPolicy"
    )


class RuntimeManifestPayload(StrictModel):
    schema_version: Literal[1] = Field(..., alias="schemaVersion")
    manifest_id: str = Field(..., alias="manifestId", min_length=1, max_length=120)
    job_id: str = Field(..., alias="jobId", min_length=1, max_length=120)
    attempt: int = Field(..., ge=0)
    owner_user_id: str = Field(..., alias="ownerUserId", min_length=1, max_length=120)
    create_id: str = Field(..., alias="createId", min_length=1, max_length=120)
    brief_id: str = Field(..., alias="briefId", min_length=1, max_length=120)
    locale: str = Field(..., min_length=1, max_length=35)
    run_notes: str | None = Field(None, alias="runNotes", max_length=20_000)
    policies: RuntimeManifestPolicies
    agent_snapshot_digest: str | None = Field(None, alias="agentSnapshotDigest")
    skill_snapshot_digest: str | None = Field(None, alias="skillSnapshotDigest")
    resolved_at_utc: datetime = Field(..., alias="resolvedAtUtc")
    entries: list[RuntimeManifestEntry] = Field(..., min_length=1, max_length=500)


class PersistedManifestEnvelope(StrictModel):
    canonical_json: str = Field(..., alias="canonicalJson", min_length=2)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    signing_key_id: str = Field(..., alias="signingKeyId", min_length=1, max_length=120)


def verify_persisted_manifest(
    envelope: PersistedManifestEnvelope, signing_keys: dict[str, str]
) -> RuntimeManifestPayload:
    actual = hashlib.sha256(envelope.canonical_json.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual, envelope.sha256):
        raise ValueError("Manifest digest verification failed.")
    encoded_key = signing_keys.get(envelope.signing_key_id)
    if not encoded_key:
        raise ValueError("Unknown manifest signature key.")
    try:
        key = base64.b64decode(encoded_key, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Manifest signature key is not valid base64.") from error
    if len(key) < 32:
        raise ValueError("Manifest signature key must contain at least 256 bits.")
    expected = hmac.new(
        key, envelope.sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, envelope.signature):
        raise ValueError("Manifest signature verification failed.")
    return RuntimeManifestPayload.model_validate_json(envelope.canonical_json)


class RuntimeManifestQueryRequest(StrictModel):
    need: str = Field(..., min_length=1, max_length=2_000)
    manifest: PersistedManifestEnvelope
    top_k: int = Field(8, alias="topK", ge=1, le=50)
