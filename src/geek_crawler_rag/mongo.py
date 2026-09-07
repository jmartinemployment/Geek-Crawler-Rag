"""Mongo access to geek_crawler crawl_pages / crawl_runs / entities.

GeekRepository stores PG-export-shaped documents: PascalCase fields, Guid as
string ("d" format). Corpus reads are primary; deletes remove unusable pages
that leaked past the crawler (locale / failure / extract-empty).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from geek_crawler_rag.metadata import EntityRef, entity_from_crawl, entity_from_doc, normalize_host

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
    markdown: str | None = None
    title: str | None = None
    crawled_at: str | None = None
    failure_reason: str | None = None
    robots_allowed: bool | None = None


class MongoCorpus:
    def __init__(self, url: str, db_name: str = "geek_crawler") -> None:
        self._client = AsyncIOMotorClient(url)
        self._db: AsyncIOMotorDatabase = self._client[db_name]
        self._entity_cache: list[dict[str, Any]] | None = None

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
        """Paginate pages for a run. Batches Html/Markdown deliberately (large documents)."""
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "Markdown": 1,
            "markdown": 1,
            "Title": 1,
            "title": 1,
            "CrawledAtUtc": 1,
            "FailureReason": 1,
            "RobotsAllowed": 1,
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

    async def resolve_entity(self, *, host: str, crawl_type: str) -> EntityRef:
        """Match Mongo entities.domains to host; else crawlType + host fallback."""
        host_n = normalize_host(host)
        fallback = entity_from_crawl(crawl_type, host_n)
        if not host_n:
            return fallback
        try:
            entities = await self._load_entities()
        except Exception:
            logger.exception("Failed loading entities collection; using crawlType fallback")
            return fallback

        for doc in entities:
            domains_raw = doc.get("domains") or doc.get("Domains") or []
            if isinstance(domains_raw, str):
                domains = [domains_raw]
            elif isinstance(domains_raw, list):
                domains = [str(d) for d in domains_raw if d]
            else:
                domains = []
            for domain in domains:
                d = normalize_host(domain)
                if not d:
                    continue
                if host_n == d or host_n.endswith("." + d):
                    return entity_from_doc(doc, fallback_host=host_n, crawl_type=crawl_type)
        return fallback

    async def _load_entities(self) -> list[dict[str, Any]]:
        if self._entity_cache is not None:
            return self._entity_cache
        names = await self._db.list_collection_names()
        if "entities" not in names:
            self._entity_cache = []
            return self._entity_cache
        cursor = self._db["entities"].find({}).limit(5000)
        self._entity_cache = [doc async for doc in cursor]
        logger.info("Loaded %s entities for domain matching", len(self._entity_cache))
        return self._entity_cache

    async def get_page(self, page_id: str) -> CrawlPage | None:
        """Load one page by Id (Guid string). Includes Markdown for citation reads."""
        if not page_id:
            return None
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "Markdown": 1,
            "markdown": 1,
            "Title": 1,
            "title": 1,
            "Excerpt": 1,
            "CrawledAtUtc": 1,
            "FailureReason": 1,
            "RobotsAllowed": 1,
            "_id": 0,
        }
        doc = await self._db["crawl_pages"].find_one({"Id": page_id}, projection)
        if doc is None:
            return None
        return _page_from_doc(doc, str(doc.get("RunId") or ""))

    async def get_page_by_url(self, *, run_id: str, url: str) -> CrawlPage | None:
        """Lookup by run + Url or FinalUrl (exact match)."""
        if not run_id or not url:
            return None
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "Markdown": 1,
            "markdown": 1,
            "Title": 1,
            "title": 1,
            "Excerpt": 1,
            "CrawledAtUtc": 1,
            "FailureReason": 1,
            "RobotsAllowed": 1,
            "_id": 0,
        }
        doc = await self._db["crawl_pages"].find_one(
            {
                "RunId": run_id,
                "$or": [{"Url": url}, {"FinalUrl": url}],
            },
            projection,
        )
        if doc is None:
            return None
        return _page_from_doc(doc, run_id)

    async def delete_page(self, page_id: str) -> int:
        """Delete one crawl_pages doc and its crawl_links. Returns links removed."""
        if not page_id:
            return 0
        link_res = await self._db["crawl_links"].delete_many({"PageId": page_id})
        await self._db["crawl_pages"].delete_one({"Id": page_id})
        return int(link_res.deleted_count)


def _page_from_doc(doc: dict[str, Any], run_id: str) -> CrawlPage:
    html = doc.get("Html")
    if html is not None and not isinstance(html, str):
        html = str(html)
    markdown = doc.get("Markdown")
    if markdown is None:
        markdown = doc.get("markdown")
    if markdown is not None and not isinstance(markdown, str):
        markdown = str(markdown)
    title = doc.get("Title")
    if title is None:
        title = doc.get("title")
    if title is not None and not isinstance(title, str):
        title = str(title)
    crawled = doc.get("CrawledAtUtc")
    crawled_at = None
    if crawled is not None:
        crawled_at = crawled.isoformat() if hasattr(crawled, "isoformat") else str(crawled)
    failure = doc.get("FailureReason")
    failure_reason = failure.strip() if isinstance(failure, str) and failure.strip() else None
    robots = doc.get("RobotsAllowed")
    robots_allowed = robots if isinstance(robots, bool) else None
    return CrawlPage(
        id=str(doc.get("Id") or ""),
        run_id=str(doc.get("RunId") or run_id),
        origin=str(doc.get("Origin") or ""),
        url=str(doc.get("Url") or ""),
        final_url=str(doc.get("FinalUrl") or doc.get("Url") or ""),
        html=html,
        markdown=markdown.strip() if isinstance(markdown, str) and markdown.strip() else None,
        title=title.strip() if isinstance(title, str) and title.strip() else None,
        crawled_at=crawled_at,
        failure_reason=failure_reason,
        robots_allowed=robots_allowed,
    )
