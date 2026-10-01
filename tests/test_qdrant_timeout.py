"""Every Qdrant client must carry an explicit timeout.

qdrant_client defaults to 5 seconds and we never set it. On 2026-10-01 that killed four of the
batch's largest runs in a row -- stripe at 7,461 chunks, liveplan at 22,369, ramp and datarails
before writing anything -- all with the same empty-message ResponseHandlingException wrapping an
httpx.ReadTimeout.

Qdrant was not slow: its access log served those same PUTs in 10-70ms, at 28% memory, status green.
The API container was drawing 299% CPU on ONNX inference, starving the event loop that reads the
response. The deadline expired on our side. Every additional Qdrant call is another draw against
those odds, which is why only large runs failed, and why ramp -- whose resume path makes 110
sequential retrieve calls before any work -- failed having written nothing.

A default that is wrong only under load is the kind that survives every test and fails in
production, so this asserts on construction rather than on behaviour.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from geek_crawler_rag.config import Settings
from geek_crawler_rag.qdrant_store import QdrantStore


def test_the_setting_is_well_above_observed_qdrant_latency():
    # Observed: 10-70ms per write. 5s was the default that failed; 60 leaves three orders of room
    # while still surfacing a genuinely dead Qdrant.
    assert Settings().qdrant_timeout_seconds >= 30


def test_qdrant_store_passes_the_timeout_to_its_client():
    with patch("geek_crawler_rag.qdrant_store.AsyncQdrantClient") as client:
        QdrantStore("http://qdrant:6333", timeout_seconds=60)
    assert client.call_args.kwargs.get("timeout") == 60, (
        "QdrantStore built a client without a timeout; it would inherit the 5s default"
    )


def test_the_engine_builds_its_client_kwargs_with_the_configured_timeout():
    """Asserts on the engine's own code, not on a mock I called myself.

    The first version of this test patched both client classes and then invoked them directly with
    kwargs it had just built. That exercises the test, not llama_engine, and would pass with the
    timeout removed from the source entirely.
    """
    import inspect

    import geek_crawler_rag.llama_engine as le

    source = inspect.getsource(le.LlamaIndexEngine.__init__)
    assert "client_kwargs" in source
    block = source[source.index("client_kwargs"):]
    block = block[: block.index("QdrantClient(")]
    assert '"timeout": settings.qdrant_timeout_seconds' in block, (
        "the engine's client_kwargs no longer carries the configured timeout; both the sync and "
        "async Qdrant clients would silently inherit the 5s default"
    )


def test_no_qdrant_client_is_constructed_without_a_timeout():
    """Grep the source, because a new construction site is how this comes back."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "geek_crawler_rag"
    offenders: list[str] = []
    for path in root.glob("*.py"):
        text = path.read_text()
        for marker in ("QdrantClient(", "AsyncQdrantClient("):
            start = 0
            while (idx := text.find(marker, start)) != -1:
                start = idx + 1
                if text[max(0, idx - 6) : idx].rstrip().endswith("Async") and marker == "QdrantClient(":
                    continue  # matched inside AsyncQdrantClient
                window = text[idx : idx + 400]
                # The call either names timeout directly or splats kwargs that carry it.
                if "timeout" in window or "client_kwargs" in window:
                    continue
                offenders.append(f"{path.name} near offset {idx}")
    assert not offenders, f"Qdrant client built without a timeout: {offenders}"
