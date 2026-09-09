from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from geek_crawler_rag.app import require_api_key
from geek_crawler_rag.asset_context import (
    AssetContextService,
    chunk_asset,
    deterministic_asset_chunk_id,
    parse_asset,
)
from geek_crawler_rag.config import Settings
from geek_crawler_rag.context_models import (
    ManifestQueryRequest,
    PersistedManifestEnvelope,
    RuntimeManifestQueryRequest,
    SignedRunContextManifest,
    SourceCoordinates,
    TrustedAssetIndexRequest,
    canonical_manifest_bytes,
    verify_manifest,
    verify_persisted_manifest,
)


def _signed_manifest(*, owner: str = "owner-1") -> SignedRunContextManifest:
    key = "manifest-secret-with-enough-entropy"
    base = {
        "schemaVersion": "run-context-manifest.v1",
        "manifestId": "manifest-1",
        "jobId": "job-1",
        "attemptId": "attempt-1",
        "ownerId": owner,
        "resolvedAtUtc": datetime.now(UTC),
        "expiresAtUtc": datetime.now(UTC) + timedelta(minutes=5),
        "entries": [
            {
                "ownerId": owner,
                "assetId": "asset-1",
                "assetVersionId": "version-1",
                "resourceId": "resource-1",
                "resourceDigest": "a" * 64,
                "sourceDigest": "b" * 64,
                "lifecycle": "approved",
                "visibility": "private",
            }
        ],
        "digest": "0" * 64,
        "signature": "0" * 64,
        "signatureKeyId": "key-1",
    }
    unsigned = SignedRunContextManifest.model_validate(base)
    canonical = canonical_manifest_bytes(unsigned)
    digest = hashlib.sha256(canonical).hexdigest()
    signature = hmac.new(key.encode(), canonical, hashlib.sha256).hexdigest()
    return SignedRunContextManifest.model_validate(
        {**base, "digest": digest, "signature": signature}
    )


def test_service_auth_is_mandatory_and_local_test_mode_is_explicit() -> None:
    with pytest.raises(HTTPException) as missing:
        require_api_key(Settings(api_key=None, local_test_mode=False), None)
    assert missing.value.status_code == 503

    require_api_key(Settings(api_key=None, local_test_mode=True), None)
    require_api_key(Settings(api_key="secret"), "secret")
    with pytest.raises(HTTPException) as invalid:
        require_api_key(Settings(api_key="secret"), "wrong")
    assert invalid.value.status_code == 401


def test_text_markdown_html_dispatch_is_local_and_bounded() -> None:
    plain = parse_asset("alpha\nbeta", "text/plain")
    markdown = parse_asset("# Alpha\n\nBody", "text/markdown")
    html = parse_asset(
        "<html><body><script>attack()</script><h1>Alpha</h1><p>Body</p></body></html>",
        "text/html",
    )
    assert plain.parser_id == "plain-utf8"
    assert markdown.parser_id == "markdown-utf8"
    assert html.parser_id == "beautifulsoup-lxml"
    assert "attack" not in html.text
    with pytest.raises(ValueError, match="Unsupported"):
        parse_asset("%PDF", "application/pdf")
    with pytest.raises(ValueError, match="character limit"):
        parse_asset("1234", "text/plain", max_chars=3)


def test_asset_chunk_ids_are_digest_and_coordinate_deterministic() -> None:
    coordinates = SourceCoordinates(kind="text", startChar=0, endChar=10)
    first = deterministic_asset_chunk_id("a" * 64, coordinates)
    assert first == deterministic_asset_chunk_id("a" * 64, coordinates)
    assert first != deterministic_asset_chunk_id("b" * 64, coordinates)
    chunks = chunk_asset(
        "An English source sentence. " * 100,
        "a" * 64,
        size_tokens=30,
        overlap_tokens=5,
    )
    assert chunks
    assert all(
        chunk.coordinates.end_char > chunk.coordinates.start_char for chunk in chunks
    )


def test_manifest_tamper_and_cross_owner_entries_fail_closed() -> None:
    manifest = _signed_manifest()
    verify_manifest(manifest, {"key-1": "manifest-secret-with-enough-entropy"})
    tampered = manifest.model_copy(update={"job_id": "other-job"})
    with pytest.raises(ValueError, match="digest"):
        verify_manifest(tampered, {"key-1": "manifest-secret-with-enough-entropy"})
    with pytest.raises(ValueError, match="belong"):
        _signed_manifest().model_copy(
            update={
                "entries": [
                    _signed_manifest()
                    .entries[0]
                    .model_copy(update={"owner_id": "other-owner"})
                ]
            }
        ).coherent_entries()


@pytest.mark.asyncio
async def test_asset_query_rejects_tampered_cross_source_payload() -> None:
    manifest = _signed_manifest()

    class Store:
        async def search_assets(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    id="point-1",
                    score=0.9,
                    payload={
                        "chunkId": "chunk-1",
                        "ownerId": "owner-1",
                        "visibility": "private",
                        "manifestEligible": True,
                        "assetId": "asset-1",
                        "assetVersionId": "version-1",
                        "resourceId": "resource-1",
                        "resourceDigest": "f" * 64,
                        "sourceDigest": "b" * 64,
                        "sourceReference": "object://private/source",
                        "coordinates": {
                            "kind": "text",
                            "startChar": 0,
                            "endChar": 10,
                        },
                        "text": "tampered",
                        "language": "en",
                        "parserId": "plain-utf8",
                        "parserVersion": "1.0.0",
                        "chunkerId": "token-window",
                        "chunkerVersion": "1.0.0",
                        "embeddingModel": "text-embedding-3-small",
                    },
                )
            ]

    class Embedder:
        async def embed_query(self, _text):
            return [0.0]

    service = AssetContextService(
        Store(),  # type: ignore[arg-type]
        Embedder(),  # type: ignore[arg-type]
        Settings(
            openai_api_key="test",
            context_manifest_signing_keys={
                "key-1": "manifest-secret-with-enough-entropy"
            },
        ),
    )
    with pytest.raises(ValueError, match="manifest entry"):
        await service.query(ManifestQueryRequest(need="alpha", manifest=manifest))


@pytest.mark.asyncio
async def test_trusted_index_contract_is_digest_bound_and_owner_scoped() -> None:
    content = "Approved normalized source."
    digest = hashlib.sha256(content.encode()).hexdigest()
    captured: dict[str, object] = {}

    class Store:
        async def upsert(self, *, ids, vectors, payloads):
            captured.update(ids=ids, vectors=vectors, payloads=payloads)

    class Embedder:
        async def embed_texts(self, texts):
            return [[0.1] for _ in texts]

    request = TrustedAssetIndexRequest(
        ownerUserId="owner-1",
        assetId="asset-1",
        assetVersionId="version-1",
        resourceId="resource-1",
        sourceSha256="a" * 64,
        derivedSha256=digest,
        content=content,
        objectKey="private/owner-1/version-1.txt",
        mediaType="text/plain",
        parserName="GeekAPI.LocalTextExtractor",
        parserVersion="1",
        sourceCoordinates=[{"kind": "text", "startChar": 0, "endChar": len(content)}],
    )
    service = AssetContextService(
        Store(),  # type: ignore[arg-type]
        Embedder(),  # type: ignore[arg-type]
        Settings(openai_api_key="test"),
    )

    result = await service.index_trusted(request)

    assert result.asset_version_id == "version-1"
    assert result.chunks_upserted == 1
    assert captured["payloads"][0]["ownerId"] == "owner-1"  # type: ignore[index]
    with pytest.raises(ValueError, match="digest"):
        TrustedAssetIndexRequest.model_validate(
            {**request.model_dump(by_alias=True), "derivedSha256": "b" * 64}
        )


def _persisted_manifest() -> tuple[PersistedManifestEnvelope, str]:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/run-context-manifest.v1.json").read_text()
    )
    return (
        PersistedManifestEnvelope(
            canonicalJson=fixture["canonicalJson"],
            sha256=fixture["sha256"],
            signature=fixture["signature"],
            signingKeyId=fixture["keyId"],
        ),
        fixture["keyBase64"],
    )


@pytest.mark.asyncio
async def test_runtime_query_verifies_dotnet_manifest_and_filters_exact_versions() -> (
    None
):
    envelope, encoded_key = _persisted_manifest()
    manifest = verify_persisted_manifest(envelope, {"ctx-fixture-1": encoded_key})
    assert manifest.owner_user_id == "33333333-3333-3333-3333-333333333333"
    captured: dict[str, object] = {}

    class Store:
        async def search_asset_versions(self, _vector, **kwargs):
            captured.update(kwargs)
            return []

    class Embedder:
        async def embed_query(self, _text):
            return [0.1]

    service = AssetContextService(
        Store(),  # type: ignore[arg-type]
        Embedder(),  # type: ignore[arg-type]
        Settings(
            openai_api_key="test",
            context_manifest_signing_keys={"ctx-fixture-1": encoded_key},
        ),
    )
    result = await service.query_runtime(
        RuntimeManifestQueryRequest(need="approved facts", manifest=envelope)
    )
    assert result.manifest_id == "11111111-1111-1111-1111-111111111111"
    assert captured["owner_id"] == "33333333-3333-3333-3333-333333333333"
    assert captured["allowed_versions"] == [
        (
            "66666666-6666-6666-6666-666666666666",
            "77777777-7777-7777-7777-777777777777",
        )
    ]
