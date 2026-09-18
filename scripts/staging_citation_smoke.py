#!/usr/bin/env python3
"""Read-only staging smoke for library retrieval and page-text citation reads."""

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
    top_k: int = 3
    timeout_seconds: int = 30
    max_response_bytes: int = 2_000_000

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
    chunk_text = _required_string(first_chunk.get("text"), "query chunk text", 50_000)

    page = client.request("GET", f"/v1/pages/{quote(query_page_id, safe='')}?runId={quote(config.run_id, safe='')}")
    if page.get("pageId") != query_page_id:
        raise SmokeFailure("page response pageId does not match query chunk")
    page_text = _required_string(page.get("text"), "page text", config.max_response_bytes)
    if chunk_text not in page_text:
        raise SmokeFailure(
            "query chunk text is not an exact substring of the page text"
        )

    return {"queryChunks": len(chunks), "chunksVerified": 1}


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
