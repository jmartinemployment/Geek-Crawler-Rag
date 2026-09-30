"""LlamaIndex node builder unit tests."""

from geek_crawler_rag.config import Settings
from geek_crawler_rag.llama_nodes import page_to_nodes
from geek_crawler_rag.metadata import EntityRef
from geek_crawler_rag.mongo import CrawlPage


def _prose_blocks(heading: str, sentence: str, times: int) -> list[dict]:
    return [
        {"kind": "heading", "level": 1, "text": heading, "anchors": []},
        {"kind": "paragraph", "text": sentence * times, "anchors": []},
    ]


def test_page_to_nodes_parent_and_child():
    page = CrawlPage(
        id="p1",
        run_id="r1",
        origin="https://acme.com",
        url="https://acme.com/docs",
        final_url="https://acme.com/docs",
        html="<html><body>ignored</body></html>",
        blocks=_prose_blocks(
            "Docs",
            "This English documentation explains the partner API thoroughly. ",
            40,
        ),
        title="Docs",
    )
    entity = EntityRef(None, "acme.com", "partner", ("acme.com",))
    nodes, skip = page_to_nodes(
        page=page,
        run_id="r1",
        crawl_type="partner",
        entity=entity,
        settings=Settings(openai_api_key="test"),
    )
    assert skip == ""
    assert nodes
    roles = {n.metadata.get("chunkRole") for n in nodes}
    assert "child" in roles
    assert "parent" in roles
    assert all(n.metadata.get("runId") == "r1" for n in nodes)
    assert all(n.metadata.get("runId") == "r1" for n in nodes)


def test_page_to_nodes_builds_text_from_blocks():
    page = CrawlPage(
        id="p2",
        run_id="r1",
        origin="https://acme.com",
        url="https://acme.com/blog",
        final_url="https://acme.com/blog",
        html="<html><body>ignored</body></html>",
        blocks=_prose_blocks(
            "Title",
            "English block content for indexing with enough tokens. ",
            30,
        ),
        title="Title",
    )
    entity = EntityRef(None, "acme.com", "partner", ("acme.com",))
    nodes, skip = page_to_nodes(
        page=page,
        run_id="r1",
        crawl_type="partner",
        entity=entity,
        settings=Settings(openai_api_key="test"),
    )
    assert skip == ""
    assert any("block content" in n.get_content().lower() for n in nodes)


def test_page_to_nodes_rejects_a_page_with_no_blocks():
    page = CrawlPage(
        id="p3",
        run_id="r1",
        origin="https://acme.com",
        url="https://acme.com/docs",
        final_url="https://acme.com/docs",
        html="<html><body>"
        + ("This English documentation explains the partner API thoroughly. " * 40)
        + "</body></html>",
        blocks=[],
    )
    entity = EntityRef(None, "acme.com", "partner", ("acme.com",))
    nodes, skip = page_to_nodes(
        page=page,
        run_id="r1",
        crawl_type="partner",
        entity=entity,
        settings=Settings(openai_api_key="test"),
    )
    assert skip == "empty"
    assert nodes == []


def _page(page_id: str, blocks: list[dict]) -> CrawlPage:
    return CrawlPage(
        id=page_id,
        run_id="r1",
        origin="https://acme.com",
        url=f"https://acme.com/{page_id}",
        final_url=f"https://acme.com/{page_id}",
        html="<html><body>ignored</body></html>",
        blocks=blocks,
        title="T",
    )


def _nodes(page: CrawlPage):
    return page_to_nodes(
        page=page,
        run_id="r1",
        crawl_type="partner",
        entity=EntityRef(None, "acme.com", "partner", ("acme.com",)),
        settings=Settings(),
    )


def test_a_short_section_emits_no_parent_point() -> None:
    """71.4% of the live corpus was this case: a parent byte-identical to its only child.

    parent_child_units emits a child equal to its parent whenever a heading section is shorter than
    child_chunk_size_tokens, so those parent points duplicated their own child -- 45.2% of the
    collection, competing with it for the same top-k slots, and inflating both df and N under BM25's
    idf modifier so that rare literals scored lower than they should.
    """
    nodes, skip = _nodes(_page("short", _prose_blocks(
        "Pricing", "A short English sentence about billing. ", 3)))
    assert skip == ""
    assert nodes
    roles = [n.metadata.get("chunkRole") for n in nodes]
    assert "child" in roles
    assert "parent" not in roles, "a parent identical to its child is pure duplication"
    # The context is not lost -- every child still carries the parent text in payload, which is what
    # query.py reads (payload.get("parentText")).
    child = next(n for n in nodes if n.metadata.get("chunkRole") == "child")
    assert child.metadata.get("parentText")


def test_a_long_section_still_emits_its_parent() -> None:
    """The other 28.6%: a parent ~3x its child at p50, carrying meaning no 200-token child holds.

    Dropping the tier outright would have removed the synthesis window for pricing matrices,
    multi-step procedures and qualifiers sitting far from what they qualify. The test is whether the
    difference is empty, not whether parents exist.
    """
    nodes, skip = _nodes(_page("long", _prose_blocks(
        "Integration", "This English sentence explains the partner integration thoroughly. ", 60)))
    assert skip == ""
    roles = [n.metadata.get("chunkRole") for n in nodes]
    assert "child" in roles
    assert "parent" in roles, "a genuinely wider parent must keep its own point"

    parent = next(n for n in nodes if n.metadata.get("chunkRole") == "parent")
    child = next(n for n in nodes if n.metadata.get("chunkRole") == "child")
    assert parent.get_content().strip() != child.get_content().strip()
