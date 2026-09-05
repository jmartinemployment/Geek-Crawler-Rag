"""Persist index job status so GET /v1/index/{runId} survives process restarts.

Uses a dedicated Mongo collection (never writes crawl HTML).
"""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from geek_crawler_rag.models import IndexState, IndexStatusResponse

logger = logging.getLogger(__name__)

COLLECTION = "rag_index_jobs"


class IndexStatusStore:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    async def get(self, run_id: str) -> IndexStatusResponse | None:
        doc = await self._col.find_one({"runId": run_id})
        if doc is None:
            return None
        return _from_doc(doc)

    async def save(self, status: IndexStatusResponse) -> None:
        doc = status.model_dump(by_alias=True, mode="json")
        await self._col.replace_one({"runId": status.run_id}, doc, upsert=True)


def _from_doc(doc: dict[str, Any]) -> IndexStatusResponse:
    return IndexStatusResponse(
        run_id=str(doc.get("runId") or ""),
        state=IndexState(str(doc.get("state") or IndexState.FAILED)),
        crawl_type=doc.get("crawlType"),
        mongo_page_count=doc.get("mongoPageCount"),
        pages_seen=int(doc.get("pagesSeen") or 0),
        pages_english=int(doc.get("pagesEnglish") or 0),
        pages_skipped_lang=int(doc.get("pagesSkippedLang") or 0),
        pages_skipped_empty=int(doc.get("pagesSkippedEmpty") or 0),
        chunks_upserted=int(doc.get("chunksUpserted") or 0),
        error=doc.get("error"),
        started_at_utc=doc.get("startedAtUtc"),
        finished_at_utc=doc.get("finishedAtUtc"),
    )
