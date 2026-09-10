"""Quarantine for non-retryable embed failures (HTTP 500 and empty-input 400).

Fail-closed design (OPENAI_EMBEDDING_MAX_RETRIES=0):
  HTTP 500 from OpenAI is treated as a failure signal (not transient).
  No in-process retry loop; the batch immediately quarantines and job fails.
  Retries are manual, operator-driven after examining the quarantine.

Empty-input HTTP 400:
  -> quarantine dump -> EmbeddingCircuitOpen -> job FAILED, points kept
  -> operator examines, then salvage Yes (fix + requeue) or No (keep + report)

Quarantine lifecycle
--------------------
- Written to EMBEDDING_QUARANTINE_DIR (Docker volume on Hostinger).
- Includes OpenAI diagnostics (requestId, errorType, message) for triage.
- Retention: keep until an operator deletes after successful re-index (default
  retain 14 days; prune with find -mtime +14).
- Not auto-replayed; use docs/embedding-circuit-recovery.md.
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
        status_code: int = 500,
        run_id: str | None = None,
        batch_size: int = 0,
        request_id: str | None = None,
        openai_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.quarantine_path = quarantine_path
        self.status_code = status_code
        self.run_id = run_id
        self.batch_size = batch_size
        self.request_id = request_id
        self.openai_message = openai_message


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
    status_code: int = 500,
    exc_type: str | None = None,
    exc: BaseException | None = None,
) -> tuple[Path, str | None, str | None]:
    """Write failing embed batch to quarantine JSON; return (file path, request_id, message)."""
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

    openai_diag = describe_openai_error(exc) if exc else {}
    request_id = openai_diag.get("requestId")
    message = openai_diag.get("message")

    payload = {
        "quarantinedAtUtc": datetime.now(timezone.utc).isoformat(),
        "statusCode": status_code,
        "excType": exc_type,
        "model": model,
        "tokenCount": token_count,
        "runId": run_id,
        "batchSize": len(texts),
        "retainDaysHint": 14,
        "recovery": "manual_requeue_run_after_inspect",
        "items": items,
    }
    if openai_diag:
        payload["openaiDiagnostics"] = openai_diag
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Structured single-line event for log aggregation / Slack alert rules.
    log_event = {
        "event": "embedding_circuit_open",
        "runId": run_id,
        "statusCode": status_code,
        "batchSize": len(texts),
        "tokenCount": token_count,
        "model": model,
        "quarantinePath": str(path),
        "excType": exc_type,
    }
    if request_id:
        log_event["requestId"] = request_id
    if message:
        log_event["message"] = message
    logger.error(
        "embedding_circuit_open %s",
        json.dumps(log_event, default=str),
    )
    return path, request_id, message


def open_embedding_circuit(
    *,
    texts: Sequence[str],
    metadata_list: Sequence[dict[str, Any] | None] | None = None,
    quarantine_dir: str | Path,
    model: str,
    token_count: int | None = None,
    run_id: str | None = None,
    status_code: int = 500,
    exc_type: str | None = None,
    exc: BaseException | None = None,
) -> EmbeddingCircuitOpen:
    """Quarantine batch and return EmbeddingCircuitOpen for the caller to raise."""
    path, request_id, message = quarantine_embedding_batch(
        texts=texts,
        metadata_list=metadata_list,
        quarantine_dir=quarantine_dir,
        model=model,
        token_count=token_count,
        run_id=run_id,
        status_code=status_code,
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
        f"OpenAI embedding HTTP {status_code}; circuit open. "
        f"Quarantine: {path}",
        quarantine_path=str(path),
        status_code=status_code,
        run_id=resolved_run,
        batch_size=len(texts),
        request_id=request_id,
        openai_message=message,
    )


def describe_openai_error(exc: BaseException) -> dict[str, Any]:
    """Extract OpenAI APIError diagnostic fields, walking __cause__ if needed."""
    request_id = getattr(exc, "request_id", None)
    error_type = getattr(exc, "type", None)
    error_code = getattr(exc, "code", None)
    error_param = getattr(exc, "param", None)
    message = getattr(exc, "message", None)

    if message is None:
        message = str(exc)
    if isinstance(message, str) and len(message) > 300:
        message = message[:300]

    if (
        not request_id
        and not error_type
        and not error_code
        and not error_param
        and (message == str(exc))
    ):
        cause = getattr(exc, "__cause__", None)
        if cause is not None and cause is not exc:
            return describe_openai_error(cause)

    return {
        "requestId": request_id,
        "errorType": error_type,
        "errorCode": error_code,
        "errorParam": error_param,
        "message": message,
    }


def is_openai_http_500(exc: BaseException) -> bool:
    """True for confirmed OpenAI/API HTTP 500 (not 429 or other 4xx)."""
    status = int(getattr(exc, "status_code", 0) or 0)
    if status == 500:
        return True
    name = type(exc).__name__
    if name == "InternalServerError" and (status == 0 or status >= 500):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return is_openai_http_500(cause)
    return False


def is_empty_embedding_input_error(exc: BaseException) -> bool:
    """OpenAI rejects batches that contain '' (HTTP 400 invalid_request_error)."""
    if "input cannot be an empty string" in str(exc).lower():
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return is_empty_embedding_input_error(cause)
    return False


def should_quarantine_embedding_error(exc: BaseException) -> int | None:
    """Return HTTP status to quarantine for, or None if caller should re-raise."""
    if is_openai_http_500(exc):
        return int(getattr(exc, "status_code", 0) or 500)
    if is_empty_embedding_input_error(exc):
        return int(getattr(exc, "status_code", 0) or 400)
    return None
