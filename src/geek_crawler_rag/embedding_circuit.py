"""Quarantine for embed failures: write the batch to disk, then fail the job closed.

An embedding call is made **once**. There is no retry loop and no bounded retry budget -- an earlier
version of this docstring described one (``*_TRANSIENT_RETRIES``, default 2) and it was removed on
2026-09-28 along with the code it documented, per ``plans/rules.md`` §3a. A retry that succeeds on its
second attempt makes the first failure invisible.

So a batch that cannot be embedded is written here and the job fails. It is not skipped, and the run
does not continue with a hole in the corpus that nothing records. Points already upserted are kept:
the failure is partial by nature and destroying good work does not make it less so.

**Every embed failure quarantines, and that is a change.** The predicates this module used to classify
on -- a provider's HTTP 500, and its 400 for an empty input string -- were specific to a remote API.
Once embeddings moved local (2026-09-30) neither could ever fire again, so
``should_quarantine_embedding_error`` returned ``None`` for everything and the circuit silently stopped
engaging. A fail-closed mechanism that cannot trigger is worse than none, because it reads as
protection. There is no status code to classify on now, so there is nothing to classify: any exception
from the embedder is a batch that was not embedded.

Quarantine lifecycle
--------------------
- Written to EMBEDDING_QUARANTINE_DIR (a Docker volume on Hostinger).
- Retention: keep until an operator deletes it after a successful re-index (default hint 14 days;
  prune with ``find -mtime +14``).
- Not auto-replayed; see ``docs/embedding-circuit-recovery.md``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 500


class EmbeddingCircuitOpen(RuntimeError):
    """Embedding batch failed; quarantine written; job must fail closed (no wipe)."""

    def __init__(
        self,
        message: str,
        *,
        quarantine_path: str,
        reason: str,
        run_id: str | None = None,
        batch_size: int = 0,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.quarantine_path = quarantine_path
        # A slug, not an HTTP status. There is no transport to carry one, and `status_code` invited a
        # reader to look for an API response that does not exist.
        self.reason = reason
        self.run_id = run_id
        self.batch_size = batch_size
        self.detail = detail


def _item_preview(text: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    preview = text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"
    item: dict[str, Any] = {
        "textPreview": preview,
        "textLength": len(text),
    }
    if metadata:
        for key in (
            "runId",
            "pageId",
            "chunkId",
            "chunkRole",
            "host",
            "crawlType",
        ):
            if key in metadata and metadata[key] is not None:
                item[key] = metadata[key]
    return item


def quarantine_embedding_batch(
    *,
    texts: Sequence[str],
    metadata_list: Sequence[dict[str, Any] | None] | None = None,
    quarantine_dir: str | Path,
    model: str,
    token_count: int | None = None,
    run_id: str | None = None,
    reason: str,
    exc_type: str | None = None,
    exc: BaseException | None = None,
) -> tuple[Path, str | None]:
    """Write the failing batch to quarantine JSON; return (file path, detail)."""
    root = Path(quarantine_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc_mkdir:
        logger.error(
            "embedding_circuit_quarantine_dir_unusable dir=%s err=%s",
            root,
            exc_mkdir,
        )
        raise
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"failed_embedding_{stamp}_{uuid.uuid4().hex[:8]}.json"

    meta_seq = list(metadata_list or [])
    items = []
    for i, text in enumerate(texts):
        meta = meta_seq[i] if i < len(meta_seq) else None
        items.append(_item_preview(text, meta))
        if run_id is None and isinstance(meta, dict) and meta.get("runId"):
            run_id = str(meta["runId"])

    detail = _exception_detail(exc) if exc else None

    payload = {
        "quarantinedAtUtc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "excType": exc_type,
        "model": model,
        "tokenCount": token_count,
        "runId": run_id,
        "batchSize": len(texts),
        "retainDaysHint": 14,
        "recovery": "manual_requeue_run_after_inspect",
        "items": items,
    }
    if detail:
        payload["detail"] = detail
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Structured single-line event for log aggregation / Slack alert rules.
    log_event = {
        "event": "embedding_circuit_open",
        "runId": run_id,
        "reason": reason,
        "batchSize": len(texts),
        "tokenCount": token_count,
        "model": model,
        "quarantinePath": str(path),
        "excType": exc_type,
    }
    if detail:
        log_event["detail"] = detail
    logger.error(
        "embedding_circuit_open %s",
        json.dumps(log_event, default=str),
    )
    return path, detail


def open_embedding_circuit(
    *,
    texts: Sequence[str],
    metadata_list: Sequence[dict[str, Any] | None] | None = None,
    quarantine_dir: str | Path,
    model: str,
    token_count: int | None = None,
    run_id: str | None = None,
    reason: str,
    exc_type: str | None = None,
    exc: BaseException | None = None,
) -> EmbeddingCircuitOpen:
    """Quarantine the batch and return EmbeddingCircuitOpen for the caller to raise."""
    path, detail = quarantine_embedding_batch(
        texts=texts,
        metadata_list=metadata_list,
        quarantine_dir=quarantine_dir,
        model=model,
        token_count=token_count,
        run_id=run_id,
        reason=reason,
        exc_type=exc_type,
        exc=exc,
    )
    resolved_run = run_id
    if resolved_run is None:
        for meta in metadata_list or []:
            if isinstance(meta, dict) and meta.get("runId"):
                resolved_run = str(meta["runId"])
                break
    return EmbeddingCircuitOpen(
        f"Embedding failed ({reason}); circuit open. Quarantine: {path}",
        quarantine_path=str(path),
        reason=reason,
        run_id=resolved_run,
        batch_size=len(texts),
        detail=detail,
    )


def _exception_detail(exc: BaseException) -> str | None:
    """A bounded, human-readable description of what failed, walking ``__cause__``.

    ONNX and tokenizer errors put the useful text on the innermost exception, so the outermost
    ``str(exc)`` is often a bare wrapper.
    """
    seen: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < 4:
        text = str(current).strip()
        if text and text not in seen:
            seen.append(f"{type(current).__name__}: {text}")
        nxt = getattr(current, "__cause__", None)
        current = nxt if nxt is not current else None
    if not seen:
        return None
    return " <- ".join(seen)[:500]


EMPTY_INPUT = "empty_input"
INFERENCE_FAILED = "inference_failed"


def classify_embedding_failure(exc: BaseException) -> str:
    """Name the failure. Always quarantines -- there is no "carry on" answer.

    This replaced ``should_quarantine_embedding_error``, which returned an HTTP status or ``None`` and
    decided by matching a remote provider's 500 and its 400 for an empty input string. With embeddings
    computed locally neither could fire, so it answered ``None`` for everything and the circuit never
    engaged. A fail-closed guard that cannot trigger is worse than no guard, because it reads as one.

    The distinction that survives is ``empty_input``, because it has a different cause and a different
    fix: an empty string means the chunker or the sanitizer let something through, not that inference
    broke. ``sanitize_embedding_texts`` and the empty-string guard in ``llama_engine`` are supposed to
    make it unreachable, so seeing it here means one of those has a hole.
    """
    text = str(exc).lower()
    if "empty" in text and ("input" in text or "string" in text or "text" in text):
        return EMPTY_INPUT
    return INFERENCE_FAILED
