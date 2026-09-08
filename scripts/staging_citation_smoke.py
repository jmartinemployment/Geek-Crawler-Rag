#!/usr/bin/env python3
"""Read-only staging smoke for the citation-backed RAG workflow."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen


class SmokeFailure(RuntimeError):
    """A staging contract assertion or HTTP request failed."""


def _bounded_text(name: str, value: str, minimum: int, maximum: int) -> str:
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        raise SmokeFailure(
            f"{name} must contain between {minimum} and {maximum} characters"
        )
    return value


def _bounded_int(name: str, value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SmokeFailure(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise SmokeFailure(f"{name} must be between {minimum} and {maximum}")
    return parsed


@dataclass(frozen=True)
class SmokeConfig:
    base_url: str
    api_key: str
    run_id: str
    query: str
    topic: str
    writing_intent: str
    model_policy_version: str = "content-model-policy.v1"
    model_policy_preset: str = "best-quality"
    top_k: int = 3
    timeout_seconds: int = 30
    max_response_bytes: int = 2_000_000
    max_citations: int = 10

    @classmethod
    def from_env(cls) -> "SmokeConfig":
        base_url = os.environ.get("STAGING_RAG_URL", "").strip().rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
            raise SmokeFailure("STAGING_RAG_URL must be an absolute HTTPS URL")

        return cls(
            base_url=base_url,
            api_key=_bounded_text(
                "STAGING_RAG_API_KEY",
                os.environ.get("STAGING_RAG_API_KEY", ""),
                1,
                512,
            ),
            run_id=_bounded_text(
                "STAGING_RAG_RUN_ID",
                os.environ.get("STAGING_RAG_RUN_ID", ""),
                1,
                200,
            ),
            query=_bounded_text(
                "STAGING_RAG_QUERY",
                os.environ.get(
                    "STAGING_RAG_QUERY",
                    "What product capabilities are explicitly documented?",
                ),
                3,
                500,
            ),
            topic=_bounded_text(
                "STAGING_RAG_TOPIC",
                os.environ.get(
                    "STAGING_RAG_TOPIC",
                    "Summarize one explicitly documented product capability",
                ),
                3,
                300,
            ),
            writing_intent=_bounded_text(
                "STAGING_RAG_WRITING_INTENT",
                os.environ.get("STAGING_RAG_WRITING_INTENT", "Technical Article"),
                1,
                100,
            ),
            top_k=_bounded_int(
                "STAGING_RAG_TOP_K",
                os.environ.get("STAGING_RAG_TOP_K", "3"),
                1,
                10,
            ),
            timeout_seconds=_bounded_int(
                "STAGING_RAG_TIMEOUT_SECONDS",
                os.environ.get("STAGING_RAG_TIMEOUT_SECONDS", "30"),
                1,
                120,
            ),
            max_response_bytes=_bounded_int(
                "STAGING_RAG_MAX_RESPONSE_BYTES",
                os.environ.get("STAGING_RAG_MAX_RESPONSE_BYTES", "2000000"),
                1_024,
                5_000_000,
            ),
            max_citations=_bounded_int(
                "STAGING_RAG_MAX_CITATIONS",
                os.environ.get("STAGING_RAG_MAX_CITATIONS", "10"),
                1,
                20,
            ),
        )


class JsonHttpClient:
    def __init__(self, config: SmokeConfig):
        self.config = config

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if method not in {"GET", "POST"}:
            raise SmokeFailure(f"unsupported smoke method: {method}")
        url = urljoin(f"{self.config.base_url}/", path.lstrip("/"))
        if not url.startswith(f"{self.config.base_url}/"):
            raise SmokeFailure("request escaped STAGING_RAG_URL")

        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "X-Api-Key": self.config.api_key,
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=payload, headers=headers, method=method)

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(self.config.max_response_bytes + 1)
        except HTTPError as exc:
            detail = exc.read(2_000).decode("utf-8", errors="replace")
            raise SmokeFailure(f"{method} {path} returned {exc.code}: {detail}") from exc
        except URLError as exc:
            raise SmokeFailure(f"{method} {path} failed: {exc.reason}") from exc

        if len(raw) > self.config.max_response_bytes:
            raise SmokeFailure(f"{method} {path} exceeded response byte limit")
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeFailure(f"{method} {path} did not return valid JSON") from exc
        if not isinstance(result, dict):
            raise SmokeFailure(f"{method} {path} must return a JSON object")
        return result


def _required_string(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SmokeFailure(f"{label} must be a non-empty string of at most {maximum} chars")
    return value


def run_smoke(config: SmokeConfig, client: JsonHttpClient) -> dict[str, int]:
    health = client.request("GET", "/health")
    if health.get("status") != "ok":
        raise SmokeFailure(f"health status is not ok: {health.get('status')!r}")

    query_result = client.request(
        "POST",
        "/v1/query",
        {
            "need": config.query,
            "runId": config.run_id,
            "topK": config.top_k,
            "preferParent": True,
        },
    )
    chunks = query_result.get("chunks")
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= config.top_k:
        raise SmokeFailure("query chunks must be non-empty and no larger than topK")
    first_chunk = chunks[0]
    if not isinstance(first_chunk, dict):
        raise SmokeFailure("query chunk must be an object")
    query_page_id = _required_string(first_chunk.get("pageId"), "query pageId", 200)

    page = client.request("GET", f"/v1/pages/{quote(query_page_id, safe='')}")
    if page.get("pageId") != query_page_id:
        raise SmokeFailure("page response pageId does not match query chunk")
    _required_string(page.get("markdown"), "page markdown", config.max_response_bytes)

    generated = client.request(
        "POST",
        "/v1/generate",
        {
            "writingIntent": config.writing_intent,
            "topic": config.topic,
            "partnerRunId": config.run_id,
            "generationStage": "complete",
            "canonicalBrief": {
                "version": "gcc-v2-generation-brief.v1",
                "title": config.topic,
                "targetKeyword": config.query,
                "contentType": "tech-article",
                "primaryIntent": "inform",
                "outputRequirements": {
                    "purpose": "Verify the deployed unified content-generation contract"
                },
            },
            "modelPolicyPreset": config.model_policy_preset,
            "modelPolicyVersion": config.model_policy_version,
        },
    )
    provenance = generated.get("provenance")
    if not isinstance(provenance, dict):
        raise SmokeFailure("generate returned no provenance object")
    if provenance.get("generationStage") != "complete":
        raise SmokeFailure("generate provenance did not confirm the complete stage")
    if provenance.get("modelPolicyVersion") != config.model_policy_version:
        raise SmokeFailure("generate provenance did not confirm the model policy version")
    if provenance.get("modelPolicyPreset") != config.model_policy_preset:
        raise SmokeFailure("generate provenance did not confirm the model policy preset")
    model_used = _required_string(
        provenance.get("modelUsed"), "generate provenance modelUsed", 100
    )
    if model_used != "o3":
        raise SmokeFailure(
            f"best-quality complete generation must use o3, got {model_used!r}"
        )
    if generated.get("modelUsed") != model_used:
        raise SmokeFailure("top-level modelUsed does not match provenance")

    citations = generated.get("citations")
    if not isinstance(citations, list) or not citations:
        raise SmokeFailure("generate returned no citations")
    if len(citations) > config.max_citations:
        raise SmokeFailure("generate returned more citations than the configured bound")

    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            raise SmokeFailure(f"citation {index} must be an object")
        page_id = _required_string(citation.get("pageId"), f"citation {index} pageId", 200)
        cited_quote = _required_string(
            citation.get("quote"), f"citation {index} quote", 2_000
        )
        if len(cited_quote) < 12:
            raise SmokeFailure(f"citation {index} quote is too short")
        source = client.request("GET", f"/v1/pages/{quote(page_id, safe='')}")
        markdown = _required_string(
            source.get("markdown"),
            f"citation {index} page markdown",
            config.max_response_bytes,
        )
        if cited_quote not in markdown:
            raise SmokeFailure(
                f"citation {index} quote is not an exact substring of page Markdown"
            )

    return {"queryChunks": len(chunks), "citationsVerified": len(citations)}


def main() -> int:
    try:
        config = SmokeConfig.from_env()
        result = run_smoke(config, JsonHttpClient(config))
    except SmokeFailure as exc:
        print(f"staging citation smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
