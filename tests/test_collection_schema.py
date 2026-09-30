"""The collection's creation-only settings, and the refusal to run without them.

``datatype`` and the sparse ``modifier`` cannot be repaired after creation -- a ``PATCH`` of
``datatype`` returns ``200 ok`` and changes nothing, and Qdrant will not alter a sparse modifier at
all. A collection created without them is not degraded, it is one that has to be rebuilt. These tests
exist because that is silent: BM25 without IDF still returns results, just unweighted.
"""

from __future__ import annotations

import pytest
from qdrant_client import models as qm

from geek_crawler_rag.qdrant_store import SPARSE_VECTOR_NAME, QdrantStore


def _store(**kw) -> QdrantStore:
    return QdrantStore(url="http://localhost:6333", collection="probe", vector_size=384, **kw)


class _FakeClient:
    """Answers get_collection with a configurable schema; records update calls."""

    def __init__(self, *, size=384, datatype=qm.Datatype.FLOAT16, on_disk=False,
                 modifier=qm.Modifier.IDF, has_sparse=True):
        self.updates: list[dict] = []
        sparse = (
            {SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False), modifier=modifier)}
            if has_sparse else {}
        )
        self._info = qm.CollectionInfo(
            status=qm.CollectionStatus.GREEN,
            optimizer_status=qm.OptimizersStatusOneOf.OK,
            segments_count=1,
            payload_schema={},
            config=qm.CollectionConfig(
                params=qm.CollectionParams(
                    vectors=qm.VectorParams(
                        size=size, distance=qm.Distance.COSINE,
                        datatype=datatype, on_disk=on_disk,
                    ),
                    sparse_vectors=sparse,
                ),
                hnsw_config=qm.HnswConfig(m=16, ef_construct=200, full_scan_threshold=10000,
                                          max_indexing_threads=0, on_disk=False),
                optimizer_config=qm.OptimizersConfig(
                    deleted_threshold=0.2, vacuum_min_vector_number=1000,
                    default_segment_number=2, flush_interval_sec=5,
                ),
                wal_config=qm.WalConfig(wal_capacity_mb=32, wal_segments_ahead=0),
            ),
        )

    async def get_collection(self, name):  # noqa: ANN001, ANN201
        return self._info

    async def update_collection(self, **kw):  # noqa: ANN003, ANN201
        self.updates.append(kw)


@pytest.mark.asyncio
async def test_a_matching_collection_is_accepted_and_not_patched() -> None:
    store = _store()
    store._client = _FakeClient()  # type: ignore[assignment]
    await store._verify_collection_matches_intent()
    assert store._client.updates == [], "a correct collection must not be modified on every boot"


@pytest.mark.asyncio
async def test_a_width_mismatch_fails_at_startup() -> None:
    """Every upsert would be rejected anyway; failing here puts the error where the cause is."""
    store = _store()
    store._client = _FakeClient(size=1536)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="declares size=1536"):
        await store._verify_collection_matches_intent()


@pytest.mark.asyncio
async def test_a_missing_idf_modifier_fails_at_startup() -> None:
    """The subtle one. Nothing errors without it -- BM25 just stops weighting rare terms.

    A product code would score no higher than "the", which is the opposite of why BM25 was chosen
    over SPLADE. The modifier is creation-only, so this cannot be nudged into shape.
    """
    store = _store()
    store._client = _FakeClient(modifier=qm.Modifier.NONE)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="not IDF"):
        await store._verify_collection_matches_intent()


@pytest.mark.asyncio
async def test_a_collection_with_no_sparse_vector_fails_at_startup() -> None:
    """Qdrant cannot add a sparse vector after creation, so hybrid retrieval is impossible."""
    store = _store()
    store._client = _FakeClient(has_sparse=False)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no 'text-sparse' sparse vector"):
        await store._verify_collection_matches_intent()


@pytest.mark.asyncio
async def test_a_wrong_datatype_logs_but_does_not_block(caplog) -> None:
    """It costs memory, not correctness -- and it is still unrepairable, so it logs at ERROR."""
    store = _store()
    store._client = _FakeClient(datatype=qm.Datatype.FLOAT32)  # type: ignore[assignment]
    with caplog.at_level("ERROR"):
        await store._verify_collection_matches_intent()
    assert any("not float16" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_vectors_on_disk_is_informational_only(caplog) -> None:
    """Patchable, and it changes latency rather than which passages come back."""
    store = _store()
    store._client = _FakeClient(on_disk=True)  # type: ignore[assignment]
    with caplog.at_level("INFO"):
        await store._verify_collection_matches_intent()
    assert any("on disk" in r.getMessage() for r in caplog.records)
