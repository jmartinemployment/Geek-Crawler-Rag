"""Persist index job status so GET /v1/index/{runId} survives process restarts.

Uses a dedicated Mongo collection (never writes crawl HTML).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from geek_crawler_rag.models import (
    IndexSchedulerStatus,
    IndexState,
    IndexStatusResponse,
)

logger = logging.getLogger(__name__)

COLLECTION = "rag_index_jobs"
SCHEDULER_COLLECTION = "rag_index_scheduler"
SCHEDULER_ID = "index-scheduler"


class IndexStatusStore:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]
        self._scheduler = db[SCHEDULER_COLLECTION]

    async def ensure_indexes(self) -> None:
        await self._col.create_index(
            [("runId", ASCENDING)], name="run_id_unique", unique=True
        )
        await self._col.create_index(
            [("state", ASCENDING), ("leaseUntil", ASCENDING)],
            name="state_lease",
        )

    async def get(self, run_id: str) -> IndexStatusResponse | None:
        doc = await self._col.find_one({"runId": run_id})
        if doc is None:
            return None
        return _from_doc(doc)

    async def save(self, status: IndexStatusResponse) -> None:
        doc = status.model_dump(by_alias=True, mode="python")
        await self._col.update_one({"runId": status.run_id}, {"$set": doc}, upsert=True)

    async def claim(
        self,
        run_id: str,
        *,
        owner: str,
        lease_seconds: int,
        trigger: str,
        force: bool,
    ) -> IndexStatusResponse | None:
        now = datetime.now(timezone.utc)
        claimable: list[dict[str, Any]] = [
            {"state": {"$nin": [IndexState.PENDING, IndexState.RUNNING]}},
            {
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                "leaseUntil": {"$lte": now},
            },
            {
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                "leaseUntil": {"$exists": False},
            },
        ]
        if not force:
            claimable[0] = {
                "state": {
                    "$nin": [
                        IndexState.PENDING,
                        IndexState.RUNNING,
                        IndexState.COMPLETE,
                    ]
                }
            }
        try:
            doc = await self._col.find_one_and_update(
                {"runId": run_id, "$or": claimable},
                {
                    "$set": {
                        "runId": run_id,
                        "state": IndexState.PENDING,
                        "trigger": trigger,
                        "leaseOwner": owner,
                        "leaseUntil": now + timedelta(seconds=lease_seconds),
                        "claimedAtUtc": now,
                        "error": None,
                        "finishedAtUtc": None,
                    },
                    "$inc": {"attempt": 1},
                    "$setOnInsert": {
                        "pagesSeen": 0,
                        "pagesEnglish": 0,
                        "chunksUpserted": 0,
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            return None
        return _from_doc(doc) if doc else None

    async def heartbeat(self, run_id: str, *, owner: str, lease_seconds: int) -> bool:
        now = datetime.now(timezone.utc)
        result = await self._col.update_one(
            {
                "runId": run_id,
                "leaseOwner": owner,
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
            },
            {"$set": {"leaseUntil": now + timedelta(seconds=lease_seconds)}},
        )
        return result.modified_count == 1

    async def release(
        self,
        run_id: str,
        *,
        owner: str,
        retry_seconds: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        status = await self.get(run_id)
        update: dict[str, Any] = {
            "leaseOwner": None,
            "leaseUntil": now,
        }
        if status and status.state == IndexState.FAILED:
            update["nextRetryAtUtc"] = now + timedelta(seconds=retry_seconds)
        await self._col.update_one(
            {"runId": run_id, "leaseOwner": owner},
            {"$set": update},
        )

    async def claim_recoverable(
        self,
        *,
        owner: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> list[IndexStatusResponse]:
        now = datetime.now(timezone.utc)
        cursor = self._col.find(
            {
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                "$and": [
                    {
                        "$or": [
                            {"leaseUntil": {"$lte": now}},
                            {"leaseUntil": {"$exists": False}},
                        ]
                    },
                    {
                        "$or": [
                            {"attempt": {"$lt": max_attempts}},
                            {"attempt": {"$exists": False}},
                        ]
                    },
                ],
            },
            {"runId": 1, "_id": 0},
        ).limit(100)
        recovered: list[IndexStatusResponse] = []
        async for doc in cursor:
            claimed = await self.claim(
                str(doc["runId"]),
                owner=owner,
                lease_seconds=lease_seconds,
                trigger="recovery",
                force=True,
            )
            if claimed is not None:
                recovered.append(claimed)
        return recovered

    async def excluded_run_ids(self, *, max_attempts: int) -> set[str]:
        now = datetime.now(timezone.utc)
        cursor = self._col.find(
            {
                "$or": [
                    {"state": IndexState.COMPLETE},
                    {
                        "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                        "leaseUntil": {"$gt": now},
                    },
                    {"attempt": {"$gte": max_attempts}},
                    {
                        "state": IndexState.FAILED,
                        "nextRetryAtUtc": {"$gt": now},
                    },
                ]
            },
            {"runId": 1, "_id": 0},
        )
        return {str(doc["runId"]) async for doc in cursor if doc.get("runId")}

    async def has_active_job(self) -> bool:
        now = datetime.now(timezone.utc)
        doc = await self._col.find_one(
            {
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                "leaseUntil": {"$gt": now},
            },
            {"_id": 1},
        )
        return doc is not None

    async def scheduler_status(
        self, *, enabled: bool, interval_seconds: int
    ) -> IndexSchedulerStatus:
        now = datetime.now(timezone.utc)
        doc = await self._scheduler.find_one({"_id": SCHEDULER_ID})
        if doc is None:
            await self._scheduler.update_one(
                {"_id": SCHEDULER_ID},
                {
                    "$setOnInsert": {
                        "enabled": enabled,
                        "intervalSeconds": interval_seconds,
                        "nextRunAtUtc": now,
                    }
                },
                upsert=True,
            )
            doc = await self._scheduler.find_one({"_id": SCHEDULER_ID}) or {}
        elif (
            bool(doc.get("enabled")) != enabled
            or int(doc.get("intervalSeconds") or 0) != interval_seconds
        ):
            await self._scheduler.update_one(
                {"_id": SCHEDULER_ID},
                {
                    "$set": {
                        "enabled": enabled,
                        "intervalSeconds": interval_seconds,
                    }
                },
            )
            doc["enabled"] = enabled
            doc["intervalSeconds"] = interval_seconds
        return _scheduler_from_doc(doc, enabled, interval_seconds)

    async def claim_scheduler_due(
        self,
        *,
        owner: str,
        lease_seconds: int,
        enabled: bool,
        interval_seconds: int,
    ) -> bool:
        status = await self.scheduler_status(
            enabled=enabled, interval_seconds=interval_seconds
        )
        now = datetime.now(timezone.utc)
        if not enabled or (status.next_run_at_utc and status.next_run_at_utc > now):
            return False
        doc = await self._scheduler.find_one_and_update(
            {
                "_id": SCHEDULER_ID,
                "enabled": True,
                "nextRunAtUtc": {"$lte": now},
                "$or": [
                    {"leaseUntil": {"$lte": now}},
                    {"leaseUntil": {"$exists": False}},
                ],
            },
            {
                "$set": {
                    "leaseOwner": owner,
                    "leaseUntil": now + timedelta(seconds=lease_seconds),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        return doc is not None

    async def complete_scheduler_tick(
        self,
        *,
        owner: str,
        interval_seconds: int,
        run_id: str | None,
        error: str | None,
    ) -> None:
        now = datetime.now(timezone.utc)
        update: dict[str, Any] = {
            "nextRunAtUtc": now + timedelta(seconds=interval_seconds),
            "lastError": error,
            "leaseOwner": None,
            "leaseUntil": now,
        }
        if run_id:
            update["lastRunId"] = run_id
            update["lastEnqueuedAtUtc"] = now
        await self._scheduler.update_one(
            {"_id": SCHEDULER_ID, "leaseOwner": owner},
            {"$set": update},
        )


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
        pages_deleted_locale=int(doc.get("pagesDeletedLocale") or 0),
        pages_deleted_failure=int(doc.get("pagesDeletedFailure") or 0),
        pages_deleted_empty=int(doc.get("pagesDeletedEmpty") or 0),
        pages_deleted_non_english=int(doc.get("pagesDeletedNonEnglish") or 0),
        chunks_upserted=int(doc.get("chunksUpserted") or 0),
        attempt=int(doc.get("attempt") or 0),
        trigger=str(doc.get("trigger") or "manual"),
        embedding_rate_limit_retries=int(
            doc.get("embeddingRateLimitRetries") or 0
        ),
        embedding_wait_seconds=float(doc.get("embeddingWaitSeconds") or 0.0),
        error=doc.get("error"),
        started_at_utc=doc.get("startedAtUtc"),
        finished_at_utc=doc.get("finishedAtUtc"),
    )


def _scheduler_from_doc(
    doc: dict[str, Any], enabled: bool, interval_seconds: int
) -> IndexSchedulerStatus:
    return IndexSchedulerStatus(
        enabled=bool(doc.get("enabled", enabled)),
        interval_seconds=int(doc.get("intervalSeconds") or interval_seconds),
        next_run_at_utc=doc.get("nextRunAtUtc"),
        last_enqueued_at_utc=doc.get("lastEnqueuedAtUtc"),
        last_run_id=doc.get("lastRunId"),
        last_error=doc.get("lastError"),
    )
