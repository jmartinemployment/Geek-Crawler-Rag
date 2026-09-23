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
    #
    # They are {label, href} objects in the payload, not strings. Typing them list[str] made
    # /v1/query return 500 for every chunk that had any -- this test asserted the wrong shape and
    # passed, because it supplied the wrong shape too.
    hit = _hit(anchors=[{"label": "Tipalti", "href": "https://tipalti.com"}])

    assert [a.label for a in hit.anchors] == ["Tipalti"]
    assert [a.href for a in hit.anchors] == ["https://tipalti.com"]
    assert _hit().anchors == []


def test_an_anchor_missing_a_label_or_href_still_validates():
    # Real crawl payloads are not uniform, and a half-populated anchor must not fail the whole
    # response the way the wrong type just did.
    hit = _hit(anchors=[{"label": "Docs"}, {"href": "https://example.test"}])

    assert len(hit.anchors) == 2


def test_the_fields_serialize_under_their_camel_case_aliases():
    # The C# consumer binds on the alias, so a rename here silently empties the fields there.
    dumped = _hit(
        parentText="p", childText="c", anchors=[{"label": "a", "href": "https://a.test"}]
    ).model_dump(by_alias=True)

    assert dumped["parentText"] == "p"
    assert dumped["childText"] == "c"
    assert dumped["anchors"] == [{"label": "a", "href": "https://a.test"}]
