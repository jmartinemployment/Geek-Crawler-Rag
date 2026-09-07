from geek_crawler_rag.graph_retrieve import build_theme_hits
from geek_crawler_rag.models import ChunkHit


def test_build_theme_hits_entities_and_edges():
    chunks = [
        ChunkHit(
            runId="r1",
            crawlType="partner",
            host="acme.com",
            url="https://acme.com/pricing",
            finalUrl="https://acme.com/pricing",
            title="Pricing",
            chunkIndex=0,
            language="en",
            text="pricing plans",
            score=0.9,
            entityName="Acme",
            category="pricing",
        ),
        ChunkHit(
            runId="r1",
            crawlType="competitors",
            host="rival.com",
            url="https://rival.com/pricing",
            finalUrl="https://rival.com/pricing",
            title="Rival pricing",
            chunkIndex=0,
            language="en",
            text="rival plans",
            score=0.8,
            entityName="Rival",
            category="pricing",
        ),
    ]
    themes = build_theme_hits(chunks, max_themes=12)
    assert any(t.relationship == "entity" for t in themes)
    assert any(t.relationship and t.relationship.startswith("co-occurs:") for t in themes)
    assert any(t.label.startswith("theme:") for t in themes)
