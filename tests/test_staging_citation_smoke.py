"""Offline tests for the opt-in staging citation smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "staging_citation_smoke.py"
_SPEC = importlib.util.spec_from_file_location("staging_citation_smoke", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SMOKE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SMOKE
_SPEC.loader.exec_module(_SMOKE)

SmokeConfig = _SMOKE.SmokeConfig
SmokeFailure = _SMOKE.SmokeFailure
run_smoke = _SMOKE.run_smoke


class FakeClient:
    def __init__(self, citation_quote: str = "Acme supports SSO for enterprise teams."):
        self.citation_quote = citation_quote
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, path, body))
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/query":
            return {
                "chunks": [
                    {
                        "pageId": "page/one",
                        "text": "Acme supports SSO for enterprise teams.",
                    }
                ]
            }
        if path == "/v1/pages/page%2Fone":
            return {
                "pageId": "page/one",
                "markdown": "# Identity\n\nAcme supports SSO for enterprise teams.",
            }
        if path == "/v1/generate":
            return {
                "content": "Draft",
                "modelUsed": "o3",
                "provenance": {
                    "generationStage": "complete",
                    "modelUsed": "o3",
                    "modelPolicyPreset": "best-quality",
                    "modelPolicyVersion": "content-model-policy.v1",
                    "promptVersion": "citeable-generate.v2",
                    "retrieval": "hybrid",
                    "evidenceIds": ["citation-page"],
                },
                "citations": [
                    {
                        "pageId": "citation-page",
                        "url": "https://example.test/identity",
                        "quote": self.citation_quote,
                    }
                ],
            }
        if path == "/v1/pages/citation-page":
            return {
                "pageId": "citation-page",
                "markdown": "# Identity\n\nAcme supports SSO for enterprise teams.",
            }
        raise AssertionError(f"unexpected request: {method} {path}")


def config() -> SmokeConfig:
    return SmokeConfig(
        base_url="https://rag.staging.example",
        api_key="test-key",
        run_id="run-123",
        query="What identity capabilities are documented?",
        topic="Documented identity capabilities",
        writing_intent="Technical Article",
    )


def test_smoke_is_read_only_and_verifies_exact_quote():
    client = FakeClient()

    result = run_smoke(config(), client)  # type: ignore[arg-type]

    assert result == {"queryChunks": 1, "citationsVerified": 1}
    assert [(method, path) for method, path, _ in client.calls] == [
        ("GET", "/health"),
        ("POST", "/v1/query"),
        ("GET", "/v1/pages/page%2Fone"),
        ("POST", "/v1/generate"),
        ("GET", "/v1/pages/citation-page"),
    ]
    assert all(
        method == "GET" or path in {"/v1/query", "/v1/generate"}
        for method, path, _ in client.calls
    )
    generate_body = next(
        body for method, path, body in client.calls
        if method == "POST" and path == "/v1/generate"
    )
    assert generate_body is not None
    assert generate_body["modelPolicyPreset"] == "best-quality"
    assert generate_body["modelPolicyVersion"] == "content-model-policy.v1"
    assert generate_body["canonicalBrief"]["contentType"] == "tech-article"


def test_smoke_rejects_quote_that_is_not_exact_substring():
    client = FakeClient("Acme supports SSO\nfor enterprise teams.")

    with pytest.raises(SmokeFailure, match="not an exact substring"):
        run_smoke(config(), client)  # type: ignore[arg-type]


def test_smoke_rejects_missing_model_provenance():
    class MissingProvenanceClient(FakeClient):
        def request(
            self, method: str, path: str, body: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            response = super().request(method, path, body)
            if path == "/v1/generate":
                response.pop("provenance")
            return response

    with pytest.raises(SmokeFailure, match="no provenance"):
        run_smoke(config(), MissingProvenanceClient())  # type: ignore[arg-type]


def test_config_rejects_unsafe_url_and_out_of_bounds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STAGING_RAG_URL", "http://rag.staging.example")
    monkeypatch.setenv("STAGING_RAG_API_KEY", "key")
    monkeypatch.setenv("STAGING_RAG_RUN_ID", "run")

    with pytest.raises(SmokeFailure, match="absolute HTTPS"):
        SmokeConfig.from_env()

    monkeypatch.setenv("STAGING_RAG_URL", "https://rag.staging.example")
    monkeypatch.setenv("STAGING_RAG_TOP_K", "11")
    with pytest.raises(SmokeFailure, match="between 1 and 10"):
        SmokeConfig.from_env()
