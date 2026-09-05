"""Read-only Mongo access to geek_crawler crawl_pages / crawl_runs.

GeekRepository stores PG-export-shaped documents: PascalCase fields, Guid as
string ("d" format). We never write crawl HTML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrawlRun:
    id: str
    crawl_type: str
    status: str


@dataclass(frozen=True)
class CrawlPage:
    id: str
    run_id: str
    origin: str
    url: str
    final_url: str
    html: str | None


class MongoCorpus:
    def __init__(self, url: str, db_name: str = "geek_crawler") -> None:
        self._client = AsyncIOMotorClient(url)
        self._db: AsyncIOMotorDatabase = self._client[db_name]

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    async def close(self) -> None:
        self._client.close()

    async def ping(self) -> bool:
        await self._client.admin.command("ping")
        return True

    async def get_run(self, run_id: str) -> CrawlRun | None:
        doc = await self._db["crawl_runs"].find_one({"Id": run_id})
        if doc is None:
            return None
        return CrawlRun(
            id=str(doc.get("Id", run_id)),
            crawl_type=str(doc.get("CrawlType") or ""),
            status=str(doc.get("Status") or ""),
        )

    async def count_pages(self, run_id: str) -> int:
        return int(await self._db["crawl_pages"].count_documents({"RunId": run_id}))

    async def iter_pages(
        self,
        run_id: str,
        *,
        batch_size: int = 25,
    ) -> AsyncIterator[list[CrawlPage]]:
        """Paginate pages for a run. Batches Html deliberately (large documents)."""
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "_id": 0,
        }
        cursor = (
            self._db["crawl_pages"]
            .find({"RunId": run_id}, projection)
            .sort("CrawledAtUtc", 1)
            .batch_size(batch_size)
        )

        batch: list[CrawlPage] = []
        async for doc in cursor:
            batch.append(_page_from_doc(doc, run_id))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _page_from_doc(doc: dict[str, Any], run_id: str) -> CrawlPage:
    html = doc.get("Html")
    if html is not None and not isinstance(html, str):
        html = str(html)
    return CrawlPage(
        id=str(doc.get("Id") or ""),
        run_id=str(doc.get("RunId") or run_id),
        origin=str(doc.get("Origin") or ""),
        url=str(doc.get("Url") or ""),
        final_url=str(doc.get("FinalUrl") or doc.get("Url") or ""),
        html=html,
    )
