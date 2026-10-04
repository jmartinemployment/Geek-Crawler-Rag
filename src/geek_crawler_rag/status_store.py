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
INTAKE_COLLECTION = "rag_index_intake"
INTAKE_ID = "index-intake"


def lease_expired_or_missing(now: datetime) -> list[dict[str, Any]]:
    """Mongo predicates for a free/expired lease.

    Note: ``null`` leaseUntil does not match ``$lte`` or ``$exists: false``.
    """
    return [
        {"leaseUntil": {"$lte": now}},
        {"leaseUntil": {"$exists": False}},
        {"leaseUntil": None},
    ]


class IndexStatusStore:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]
        self._scheduler = db[SCHEDULER_COLLECTION]
        self._intake = db[INTAKE_COLLECTION]

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

    async def delete(self, run_id: str) -> bool:
        """Drop the job row for a run. True when one was removed.

        Called when a run's vectors are purged. The row outliving the vectors is
        what let a deleted run keep answering GET /v1/index/{runId} with the
        page and chunk counts of a corpus that no longer existed.
        """
        result = await self._col.delete_one({"runId": run_id})
        return result.deleted_count > 0

    async def save(
        self, status: IndexStatusResponse, *, owner: str | None = None
    ) -> bool:
        doc = status.model_dump(by_alias=True, mode="python")
        query: dict[str, Any] = {"runId": status.run_id}
        if owner is not None:
            query["leaseOwner"] = owner
        result = await self._col.update_one(
            query,
            {"$set": doc},
            upsert=owner is None,
        )
        return result.matched_count == 1 or result.upserted_id is not None

    async def is_owned(self, run_id: str, *, owner: str) -> bool:
        now = datetime.now(timezone.utc)
        doc = await self._col.find_one(
            {
                "runId": run_id,
                "leaseOwner": owner,
                "leaseUntil": {"$gt": now},
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
            },
            {"_id": 1},
        )
        return doc is not None

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
                "$or": lease_expired_or_missing(now),
            },
        ]
        if not force:
            claimable[0] = {
                "state": {
                    "$nin": [
                        IndexState.PENDING,
                        IndexState.RUNNING,
                        IndexState.COMPLETE,
                        IndexState.FAILED,
                        IndexState.SKIPPED,
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

    async def reacquire(self, run_id: str, *, owner: str, lease_seconds: int) -> bool:
        """Take the lease for a job this worker is about to execute.

        Distinct from `claim`: no attempt increment and no state change. The job was already
        claimed when it was enqueued, and this covers only the gap between that moment and the
        moment the single worker reaches it -- which, at concurrency 1, is however long everything
        queued ahead of it takes. Counting that wait as a failed attempt would spend the retry
        budget on jobs that never ran.

        Matches a lease that has expired, is absent, or is already ours. A live lease held by a
        different owner is a real conflict and is left untouched, so this cannot take a run that
        another process is indexing.
        """
        now = datetime.now(timezone.utc)
        result = await self._col.update_one(
            {
                "runId": run_id,
                "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                "$or": lease_expired_or_missing(now) + [{"leaseOwner": owner}],
            },
            {
                "$set": {
                    "leaseOwner": owner,
                    "leaseUntil": now + timedelta(seconds=lease_seconds),
                }
            },
        )
        return result.matched_count == 1

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
        # Do not schedule automatic retries for failed jobs — operator must
        # inspect quarantine / re-enqueue manually.
        if status and status.state == IndexState.FAILED:
            update["nextRetryAtUtc"] = None
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
                    {"$or": lease_expired_or_missing(now)},
                    {
                        "$or": [
                            {"attempt": {"$lt": max_attempts}},
                            {"attempt": {"$exists": False}},
                        ]
                    },
                ],
            },
            {"runId": 1, "_id": 0},
        ).sort("claimedAtUtc", ASCENDING).limit(1)
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
                    {"state": IndexState.FAILED},
                    {"state": IndexState.SKIPPED},
                    {
                        "state": {"$in": [IndexState.PENDING, IndexState.RUNNING]},
                        "leaseUntil": {"$gt": now},
                    },
                    {"attempt": {"$gte": max_attempts}},
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

    async def intake_pause_reason(self) -> str | None:
        """The operator's reason if index intake is paused, else None.

        Separate state from the scheduler pause, because they stop different things and an
        operator needs both independently. The scheduler pause stops automatic *selection*
        of runs and deliberately leaves ``POST /v1/index`` working. This stops runs being
        accepted at all, including the ones GeekAPI posts on every crawl commit -- which is
        the traffic that actually fills the queue: all fifteen jobs in flight on 2026-09-28
        were ``trigger=manual`` from GeekAPI, and a scheduler pause would have stopped none
        of them.
        """
        doc = await self._intake.find_one({"_id": INTAKE_ID})
        if doc is None or not doc.get("paused"):
            return None
        return str(doc.get("pausedReason") or "no reason recorded")

    async def set_intake_paused(
        self, *, paused: bool, reason: str | None
    ) -> tuple[bool, str | None]:
        """Pause or resume acceptance of new index jobs. Returns (paused, reason).

        Durable, so a deploy that recreates the container does not lift it -- the same
        reasoning as the scheduler pause, and the reason it is in Mongo rather than in the
        process.

        This does not touch a job that is already queued or running; those are stopped one
        at a time with ``POST /v1/index/{runId}/kill``, or together by stopping the worker.
        It closes the door, it does not clear the room.
        """
        now = datetime.now(timezone.utc)
        update: dict[str, Any] = {"paused": paused}
        if paused:
            update["pausedAtUtc"] = now
            update["pausedReason"] = reason
        else:
            update["pausedAtUtc"] = None
            update["pausedReason"] = None
        await self._intake.update_one(
            {"_id": INTAKE_ID}, {"$set": update}, upsert=True
        )
        logger.warning(
            "Index intake %s by operator%s",
            "PAUSED" if paused else "RESUMED",
            f": {reason}" if paused and reason else "",
        )
        return paused, (reason if paused else None)

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
                        "paused": False,
                        "pausedAtUtc": None,
                        "pausedReason": None,
                    }
                },
                upsert=True,
            )
            doc = await self._scheduler.find_one({"_id": SCHEDULER_ID}) or {}
        elif (
            bool(doc.get("enabled")) != enabled
            or int(doc.get("intervalSeconds") or 0) != interval_seconds
        ):
            # Config wins for enabled/interval, and only for those two. `paused` is
            # deliberately absent from this $set: it is operator state, and re-syncing it
            # from config would silently un-pause the scheduler on the next status read.
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
        if not enabled or status.paused:
            return False
        if status.next_run_at_utc and status.next_run_at_utc > now:
            return False
        # `paused` is in the filter as well as the check above, and that is the half
        # that enforces it: the read and the claim are two round trips, so a pause
        # landing between them would otherwise still let this tick enqueue a run.
        doc = await self._scheduler.find_one_and_update(
            {
                "_id": SCHEDULER_ID,
                "enabled": True,
                "paused": {"$ne": True},
                "nextRunAtUtc": {"$lte": now},
                "$or": lease_expired_or_missing(now),
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

    async def set_scheduler_paused(
        self,
        *,
        paused: bool,
        reason: str | None,
        enabled: bool,
        interval_seconds: int,
    ) -> IndexSchedulerStatus:
        """Pause or resume scheduled enqueues, durably.

        Pausing stops the scheduler claiming its next due tick. It does not touch a job
        already queued or running: those hold their own leases and finish on their own,
        which is the point -- pausing the feed is what an operator wants mid-incident, and
        killing live work is what they do not.

        The state lives in Mongo rather than in the process, so it survives the container
        recreation that a deploy performs. A pause held only in memory would be lifted by
        the very restart an operator paused in order to make safe.

        Resuming does not run a tick immediately: `nextRunAtUtc` is left exactly as it was,
        so the cadence picks up where it left off instead of firing on resume.
        """
        now = datetime.now(timezone.utc)
        update: dict[str, Any] = {"paused": paused}
        if paused:
            update["pausedAtUtc"] = now
            update["pausedReason"] = reason
        else:
            update["pausedAtUtc"] = None
            update["pausedReason"] = None
        # Ensures the document exists before the $set, so a pause on a scheduler that has
        # never ticked is recorded rather than silently matching nothing.
        await self.scheduler_status(
            enabled=enabled, interval_seconds=interval_seconds
        )
        await self._scheduler.update_one({"_id": SCHEDULER_ID}, {"$set": update})
        logger.warning(
            "Index scheduler %s by operator%s",
            "PAUSED" if paused else "RESUMED",
            f": {reason}" if paused and reason else "",
        )
        return await self.scheduler_status(
            enabled=enabled, interval_seconds=interval_seconds
        )

    async def complete_scheduler_tick(
        self,
        *,
        owner: str,
        interval_seconds: int,
        run_id: str | None,
        error: str | None,
        selection_reason: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        update: dict[str, Any] = {
            "nextRunAtUtc": now + timedelta(seconds=interval_seconds),
            "lastError": error,
            "lastSelectionReason": selection_reason,
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
        pages_skipped_unusable=int(doc.get("pagesSkippedUnusable") or 0),
        chunks_upserted=int(doc.get("chunksUpserted") or 0),
        chunks_skipped_repeat=int(doc.get("chunksSkippedRepeat") or 0),
        attempt=int(doc.get("attempt") or 0),
        trigger=str(doc.get("trigger") or "manual"),
        embedding_rate_limit_retries=int(
            doc.get("embeddingRateLimitRetries") or 0
        ),
        embedding_wait_seconds=float(doc.get("embeddingWaitSeconds") or 0.0),
        error=doc.get("error"),
        # Through _as_utc, as _scheduler_from_doc already does. Motor returns naive datetimes,
        # so a rehydrated status carried tz-naive timestamps where a freshly built one is aware.
        # finishedAtUtc is the key GeekAPI's out-of-order guard compares, and a naive value
        # serialises with no offset for C# to read in the server's local zone -- which lets a
        # stale terminal status overwrite a newer one.
        started_at_utc=_as_utc(doc.get("startedAtUtc")),
        finished_at_utc=_as_utc(doc.get("finishedAtUtc")),
    )


def _scheduler_from_doc(
    doc: dict[str, Any], enabled: bool, interval_seconds: int
) -> IndexSchedulerStatus:
    next_run_at = _as_utc(doc.get("nextRunAtUtc"))
    last_enqueued_at = _as_utc(doc.get("lastEnqueuedAtUtc"))
    return IndexSchedulerStatus(
        enabled=bool(doc.get("enabled", enabled)),
        interval_seconds=int(doc.get("intervalSeconds") or interval_seconds),
        next_run_at_utc=next_run_at,
        last_enqueued_at_utc=last_enqueued_at,
        last_run_id=doc.get("lastRunId"),
        last_error=doc.get("lastError"),
        last_selection_reason=doc.get("lastSelectionReason"),
        paused=bool(doc.get("paused", False)),
        paused_at_utc=_as_utc(doc.get("pausedAtUtc")),
        paused_reason=doc.get("pausedReason"),
    )


def _as_utc(value: Any) -> Any:
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
