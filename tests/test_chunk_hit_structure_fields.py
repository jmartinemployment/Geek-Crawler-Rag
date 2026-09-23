"""ChunkHit exposes the structural metadata llama_nodes stamps."""

from geek_crawler_rag.models import ChunkHit


def _hit(**overrides):
    base = dict(
        runId="run-1",
        crawlType="partner",
        host="www.medius.com",
        url="https://www.medius.com/ap",
        finalUrl="https://www.medius.com/ap",
        chunkIndex=3,
        language="en",
        text="Get full visibility into invoices.",
        score=0.82,
    )
    base.update(overrides)
    return ChunkHit(**base)


def test_parent_and_child_text_survive_the_response():
    # They are in the Qdrant payload and were not returned, so a retrieved child reached the writer
    # as an isolated fragment while its parent sat unused in the same payload.
    hit = _hit(
        chunkRole="child",
        parentText="Simplify AP by getting rid of paper.",
        childText="Get full visibility into invoices.",
    )

    assert hit.parent_text == "Simplify AP by getting rid of paper."
    assert hit.child_text == "Get full visibility into invoices."


def test_anchors_survive_and_default_to_empty():
    # Anchors are what make anchor-based tool detection possible at all; dropping them at the
    # response boundary is the same structural loss as dropping them at ingest.
    assert _hit(anchors=["Tipalti", "Medius"]).anchors == ["Tipalti", "Medius"]
    assert _hit().anchors == []


def test_the_fields_serialize_under_their_camel_case_aliases():
    # The C# consumer binds on the alias, so a rename here silently empties the fields there.
    dumped = _hit(parentText="p", childText="c", anchors=["a"]).model_dump(by_alias=True)

    assert dumped["parentText"] == "p"
    assert dumped["childText"] == "c"
    assert dumped["anchors"] == ["a"]
