"""Hybrid retrieval over the named sparse vector, and the backfill that populates it.

These run against an in-process Qdrant, so they cover the parts that can be proven
without the staging collection: that appending sparse values cannot destroy the
dense vector beside them, that the backfill's text selection and resume check
behave, that a sparse-only match surfaces a proper noun the dense vector misses,
and that the store binds to our vector name and our encoder rather than the one it
would pick for itself.

What they deliberately do not cover: the live collection's dense vector naming.
`ensure_collection` creates an unnamed vector while the LlamaIndex store writes a
named `text-dense` on every upsert, and only the running instance can say which is
there. scripts/inspect_qdrant_vectors.py answers that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from geek_crawler_rag.qdrant_store import SPARSE_VECTOR_NAME

_MIGRATION_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_sparse_vectors.py"


def _load_migration():
    """The backfill is a script, not a package member; load it by path to test its helpers."""
    spec = importlib.util.spec_from_file_location("migrate_sparse_vectors", _MIGRATION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()


def _sparse_by_token(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """A deterministic stand-in for SPLADE: one index per distinct lowercased word.

    The real encoder is prithivida/Splade_PP_en_v1 and downloading 500MB of ONNX
    weights is not something a unit test should do. What these tests need from an
    encoder is that identical terms collide and different terms do not, which is
    the property exact-term retrieval rests on; the model's expansion quality is a
    separate question and not one a test can assert.
    """
    indices: list[list[int]] = []
    values: list[list[float]] = []
    for text in texts:
        tokens = sorted({abs(hash(word.lower())) % 100_000 for word in text.split()})
        indices.append(tokens)
        values.append([1.0] * len(tokens))
    return indices, values


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(location=":memory:")


def test_update_vectors_appends_sparse_without_touching_dense(client: QdrantClient) -> None:
    """The property the whole backfill rests on.

    upsert replaces a point, so writing a PointStruct carrying only the sparse
    vector would delete the dense embedding beside it -- 168k of them, each an
    OpenAI call to regenerate. update_vectors writes only what it is given.
    """
    client.create_collection(
        "runs",
        vectors_config={"text-dense": qm.VectorParams(size=4, distance=qm.Distance.COSINE)},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False)),
        },
    )
    client.upsert(
        "runs",
        points=[
            qm.PointStruct(
                id=1,
                vector={"text-dense": [1.0, 0.0, 0.0, 0.0]},
                payload={"text": "Dext receipt capture"},
            )
        ],
    )

    client.update_vectors(
        "runs",
        points=[
            qm.PointVectors(
                id=1,
                vector={SPARSE_VECTOR_NAME: qm.SparseVector(indices=[5, 9], values=[0.9, 0.4])},
            )
        ],
    )

    point = client.retrieve("runs", ids=[1], with_vectors=True, with_payload=True)[0]
    assert sorted(point.vector) == sorted(["text-dense", SPARSE_VECTOR_NAME])
    assert point.vector["text-dense"] == [1.0, 0.0, 0.0, 0.0]
    assert point.payload["text"] == "Dext receipt capture"


def test_payload_text_falls_back_through_the_chunker_keys() -> None:
    assert migration.payload_text({"text": "body", "childText": "child"}) == "body"
    assert migration.payload_text({"childText": "child", "parentText": "parent"}) == "child"
    assert migration.payload_text({"parentText": "parent"}) == "parent"
    # Nothing to encode is not something to invent a body for.
    assert migration.payload_text({"text": "   "}) == ""
    assert migration.payload_text({}) == ""
    assert migration.payload_text(None) == ""


def test_has_sparse_lets_a_rerun_skip_migrated_points() -> None:
    assert migration.has_sparse({SPARSE_VECTOR_NAME: {"indices": [3], "values": [0.5]}}) is True
    # An empty sparse vector is not a migrated point; it still needs values.
    assert migration.has_sparse({SPARSE_VECTOR_NAME: {"indices": [], "values": []}}) is False
    assert migration.has_sparse({}) is False
    assert migration.has_sparse(None) is False
    # An unnamed dense vector comes back as a bare list, which carries no sparse values.
    assert migration.has_sparse([0.1, 0.2]) is False


def test_hybrid_query_surfaces_a_proper_noun_the_dense_vector_misses(client: QdrantClient) -> None:
    """The capability this feature exists for.

    The query embedding is orthogonal to every stored document, so the dense branch
    can rank nothing. Anything that comes back is the sparse branch matching the
    term exactly -- which is what "Dext" and "Zone & Co" need and what semantic
    similarity alone is weakest at.
    """
    store = QdrantVectorStore(
        client=client,
        collection_name="hybrid",
        enable_hybrid=True,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        sparse_doc_fn=_sparse_by_token,
        sparse_query_fn=_sparse_by_token,
        text_key="text",
    )
    store.add(
        [
            TextNode(text="Dext receipt capture for accountants", embedding=[1.0, 0.0, 0.0, 0.0]),
            TextNode(text="Zone and Co NetSuite billing", embedding=[0.0, 1.0, 0.0, 0.0]),
            TextNode(text="generic accounting software overview", embedding=[0.0, 0.0, 1.0, 0.0]),
        ]
    )

    result = store.query(
        VectorStoreQuery(
            query_str="Dext",
            query_embedding=[0.0, 0.0, 0.0, 1.0],
            similarity_top_k=3,
            sparse_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
        )
    )

    contents = [node.get_content() for node in result.nodes]
    assert contents, "hybrid query returned nothing"
    assert "Dext" in contents[0], f"expected the Dext passage first, got {contents}"

    scores = list(result.similarities or [])
    assert scores[0] > 0.0
    # The dense vector matched nothing, so every other document scores zero.
    assert all(score == 0.0 for score in scores[1:])


def test_explicit_encoders_keep_our_sparse_vector_name(client: QdrantClient) -> None:
    """Guards the trap that makes this feature fail silently.

    QdrantVectorStore picks its own sparse encoder when these functions are None,
    and use_old_sparse_encoder() returns True for a collection carrying a vector
    named "text-sparse" -- ours -- which selects
    naver/efficient-splade-VI-BT-large-doc. The backfill writes Splade_PP_en_v1
    weights. Two SPLADE models index different vocabularies, so the mismatch raises
    nothing and simply returns the wrong passages.
    """
    client.create_collection(
        "named",
        vectors_config={"text-dense": qm.VectorParams(size=4, distance=qm.Distance.COSINE)},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False)),
        },
    )

    store = QdrantVectorStore(
        client=client,
        collection_name="named",
        enable_hybrid=True,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        sparse_doc_fn=_sparse_by_token,
        sparse_query_fn=_sparse_by_token,
        text_key="text",
    )

    assert store.sparse_vector_name == SPARSE_VECTOR_NAME
    assert store._sparse_doc_fn is _sparse_by_token
    assert store._sparse_query_fn is _sparse_by_token
