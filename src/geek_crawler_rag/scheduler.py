"""Durable two-hour, smallest-first crawl indexing scheduler."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from geek_crawler_rag.config import Settings
from geek_crawler_rag.models import IndexSchedulerStatus
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.status_store import IndexStatusStore

logger = logging.getLogger(__name__)


class ScheduledEnqueuer(Protocol):
    async def enqueue_scheduled(self, run_id: str) -> bool: ...


class IndexScheduler:
    def __init__(
        self,
        mongo: MongoCorpus,
        status_store: IndexStatusStore,
        indexer: ScheduledEnqueuer,
        settings: Settings,
        *,
        owner: str,
    ) -> None:
        self._mongo = mongo
        self._status_store = status_store
        self._indexer = indexer
        self._settings = settings
        self._owner = owner
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        await self._status_store.scheduler_status(
            enabled=self._settings.index_scheduler_enabled,
            interval_seconds=self._settings.index_scheduler_interval_seconds,
        )
        if self._settings.index_scheduler_enabled and (
            self._task is None or self._task.done()
        ):
            self._stop.clear()
            self._task = asyncio.create_task(
                self._loop(), name="index-scheduler"
            )
            logger.info(
                "Index scheduler enabled interval=%ss smallest-first",
                self._settings.index_scheduler_interval_seconds,
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def status(self) -> IndexSchedulerStatus:
        return await self._status_store.scheduler_status(
            enabled=self._settings.index_scheduler_enabled,
            interval_seconds=self._settings.index_scheduler_interval_seconds,
        )

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Index scheduler tick failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._settings.index_scheduler_poll_seconds,
                )
            except TimeoutError:
                pass

    async def tick(self) -> str | None:
        claimed = await self._status_store.claim_scheduler_due(
            owner=self._owner,
            lease_seconds=self._settings.index_job_lease_seconds,
            enabled=self._settings.index_scheduler_enabled,
            interval_seconds=self._settings.index_scheduler_interval_seconds,
        )
        if not claimed:
            return None

        run_id: str | None = None
        error: str | None = None
        selection_reason: str | None = None
        try:
            if await self._status_store.has_active_job():
                selection_reason = "active_job"
                logger.info(
                    "Index scheduler skipped due tick because a job is active"
                )
                return None
            excluded = await self._status_store.excluded_run_ids(
                max_attempts=self._settings.index_scheduler_max_attempts
            )
            scan = await self._mongo.find_smallest_markdown_ready_run(
                excluded_run_ids=excluded
            )
            candidate = scan.candidate
            if candidate is None:
                selection_reason = scan.summary()
                logger.info(
                    "Index scheduler found no eligible Markdown-ready run: %s",
                    selection_reason,
                )
                return None
            accepted = await self._indexer.enqueue_scheduled(candidate.id)
            if accepted:
                run_id = candidate.id
                selection_reason = scan.summary()
                logger.info(
                    "Index scheduler enqueued runId=%s pages=%s",
                    candidate.id,
                    candidate.page_count,
                )
            else:
                error = f"Atomic index claim rejected for runId={candidate.id}"
                selection_reason = "claim_rejected"
                logger.warning(error)
            return run_id
        except asyncio.CancelledError:
            error = "Scheduler cancelled during due tick"
            raise
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            await self._status_store.complete_scheduler_tick(
                owner=self._owner,
                interval_seconds=self._settings.index_scheduler_interval_seconds,
                run_id=run_id,
                error=error,
                selection_reason=selection_reason,
            )
