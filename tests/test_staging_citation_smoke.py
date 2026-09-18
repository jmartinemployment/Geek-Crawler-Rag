"""Offline tests for the opt-in staging library citation smoke."""

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
    def __init__(self, chunk_text: str = "Acme supports SSO for enterprise teams."):
        self.chunk_text = chunk_text
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
                        "text": self.chunk_text,
                    }
                ]
            }
        if path == "/v1/pages/page%2Fone?runId=run-123":
            return {
                "pageId": "page/one",
                "text": "Identity\n\nAcme supports SSO for enterprise teams.",
            }
        raise AssertionError(f"unexpected request: {method} {path}")


def config() -> SmokeConfig:
    return SmokeConfig(
        base_url="https://rag.staging.example",
        api_key="test-key",
        run_id="run-123",
        query="What identity capabilities are documented?",
    )


def test_smoke_is_read_only_and_verifies_chunk_in_page_text():
    client = FakeClient()

    result = run_smoke(config(), client)  # type: ignore[arg-type]

    assert result == {"queryChunks": 1, "chunksVerified": 1}
    assert [(method, path) for method, path, _ in client.calls] == [
        ("GET", "/health"),
        ("POST", "/v1/query"),
        ("GET", "/v1/pages/page%2Fone?runId=run-123"),
    ]
    assert all(
        method == "GET" or path == "/v1/query"
        for method, path, _ in client.calls
    )


def test_smoke_rejects_chunk_that_is_not_exact_substring():
    client = FakeClient("Acme supports SSO\nfor enterprise teams.")

    with pytest.raises(SmokeFailure, match="not an exact substring"):
        run_smoke(config(), client)  # type: ignore[arg-type]


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
