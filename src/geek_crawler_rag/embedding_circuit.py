"""Circuit breaker for OpenAI embedding HTTP 500s.

State machine (per-job abort on first bad batch)
------------------------------------------------
Job starts
  -> Batch N: 200 OK -> upsert N items (kept in Qdrant)
  -> Batch N+1: HTTP 500 -> quarantine N+1 items -> raise EmbeddingCircuitOpen
  -> Job aborts immediately (no further batches)
  -> Already-upserted points are NOT deleted
  -> Operator inspects quarantine JSON, then deliberately requeues the run
     (deterministic point IDs; attempt>1 skips wipe-on-start)

Recovery is manual/orchestrated (scheduler requeue), not an in-process retry.
Exponential backoff is off by design (OPENAI_EMBEDDING_MAX_RETRIES=0).

Quarantine lifecycle
--------------------
- Written to EMBEDDING_QUARANTINE_DIR (Docker volume on Hostinger).
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
    """OpenAI embedding circuit opened after a confirmed HTTP 500."""

    def __init__(
        self,
        message: str,
        *,
        quarantine_path: str,
        status_code: int = 500,
        run_id: str | None = None,
        batch_size: int = 0,
    ) -> None:
        super().__init__(message)
        self.quarantine_path = quarantine_path
        self.status_code = status_code
        self.run_id = run_id
        self.batch_size = batch_size


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
) -> Path:
    """Write failing embed batch to quarantine JSON; return the file path."""
    root = Path(quarantine_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "embedding_circuit_quarantine_dir_unusable dir=%s err=%s",
            root,
            exc,
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
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Structured single-line event for log aggregation / Slack alert rules.
    logger.error(
        "embedding_circuit_open %s",
        json.dumps(
            {
                "event": "embedding_circuit_open",
                "runId": run_id,
                "statusCode": status_code,
                "batchSize": len(texts),
                "tokenCount": token_count,
                "model": model,
                "quarantinePath": str(path),
                "excType": exc_type,
            },
            default=str,
        ),
    )
    return path


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
) -> EmbeddingCircuitOpen:
    """Quarantine batch and return EmbeddingCircuitOpen for the caller to raise."""
    path = quarantine_embedding_batch(
        texts=texts,
        metadata_list=metadata_list,
        quarantine_dir=quarantine_dir,
        model=model,
        token_count=token_count,
        run_id=run_id,
        status_code=status_code,
        exc_type=exc_type,
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
    )


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
