"""Library capabilities must not advertise removed /v1/generate."""

from geek_crawler_rag.models import ProducerCapabilities


def test_capabilities_do_not_advertise_removed_generate_endpoint():
    wire = ProducerCapabilities().model_dump(by_alias=True)
    assert wire["product"] == "geek-rag-library"
    assert wire["features"] == ["index", "query", "pages", "templates"]
    assert wire["executionVersions"] == []
    assert wire["skillEnvelopeVersions"] == []
    assert wire["generationStages"] == []
    assert wire["agentGenerationStages"] == []
    assert wire["specialistExecutors"] == []
    assert wire["toolsAllowed"] is False
    assert wire["stageScopedToolsAllowed"] is False
