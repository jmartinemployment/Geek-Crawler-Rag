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
