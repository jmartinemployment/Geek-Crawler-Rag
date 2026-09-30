from __future__ import annotations

from datetime import datetime, timezone

import pytest

from geek_crawler_rag.mongo import MongoCorpus


def D(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def limit(self, _limit):
        return self

    def __aiter__(self):
        async def rows():
            for doc in self.docs:
                yield doc

        return rows()


class FakeRuns:
    def __init__(self, docs=None):
        self.query = None
        self.projection = None
        self.docs = docs or [
            {"Id": "large", "ContentReadyAt": D(4)},
            {"Id": "small", "ContentReadyAt": D(3)},
            {"Id": "done", "ContentReadyAt": D(2)},
            {"Id": "zero", "ContentReadyAt": D(1)},
        ]

    async def count_documents(self, _query):
        return 7

    def find(self, query, projection, **_kwargs):
        self.query = query
        self.projection = projection
        return FakeCursor(self.docs)


class FakePages:
    async def count_documents(self, query):
        return {"large": 60_000, "small": 12, "done": 2, "zero": 0}[query["RunId"]]


@pytest.mark.asyncio
async def test_run_level_readiness_selects_oldest_ready_and_reports_exclusions():
    runs = FakeRuns()
    corpus = MongoCorpus.__new__(MongoCorpus)
    corpus._db = {"crawl_runs": runs, "crawl_pages": FakePages()}

    scan = await corpus.find_smallest_content_ready_run(
        excluded_run_ids={"done"},
        maximum_pages=50_000,
    )

    assert runs.query["ContentReadyAt"] == {
        "$exists": True,
        "$nin": [None, ""],
    }
    # "large" is ready LAST (day 4) and is over the page cap; "done" is excluded; "zero" has no
    # pages. Of what remains, oldest-ready wins -- which here is "small".
    assert scan.candidate is not None
    assert scan.candidate.id == "small"
    assert scan.candidate.page_count == 12
    assert "ContentReadyAt" in runs.projection, "ordering needs the field it sorts on"
    assert scan.missing_ready_marker == 7
    assert scan.zero_pages == 1
    assert scan.safety_cap == 1
    assert scan.excluded == 1


@pytest.mark.asyncio
async def test_it_picks_the_oldest_ready_run_not_the_smallest():
    """Smallest-first silently reordered the corpus against the order the operator crawled in.

    That was chosen so a pipeline fault surfaced on the cheapest run rather than after the most
    expensive one -- a debugging property, not an operating one. Jeff, 2026-09-30: "I never liked it
    picking smallest first, as I am trying to produce content in a different order." FIFO on
    ContentReadyAt makes the order controllable by the thing the operator already controls: when they
    crawl.
    """

    class Pages:
        async def count_documents(self, query):
            return {"first": 900, "second": 10}[query["RunId"]]

    runs = FakeRuns([
        {"Id": "second", "ContentReadyAt": D(9)},   # smaller, but ready later
        {"Id": "first", "ContentReadyAt": D(8)},    # bigger, but ready first
    ])
    corpus = MongoCorpus.__new__(MongoCorpus)
    corpus._db = {"crawl_runs": runs, "crawl_pages": Pages()}

    scan = await corpus.find_smallest_content_ready_run(
        excluded_run_ids=set(), maximum_pages=50_000
    )
    assert scan.candidate.id == "first", "oldest ready wins even though it is 90x larger"


@pytest.mark.asyncio
async def test_a_missing_ready_marker_orders_last_rather_than_first():
    """Defensive only -- the filter already requires ContentReadyAt, so this cannot normally occur.

    It matters because `None` sorting first would quietly promote a run to the front of the queue.
    """

    class Pages:
        async def count_documents(self, query):
            return 5

    runs = FakeRuns([
        {"Id": "nomarker", "ContentReadyAt": None},
        {"Id": "dated", "ContentReadyAt": D(20)},
    ])
    corpus = MongoCorpus.__new__(MongoCorpus)
    corpus._db = {"crawl_runs": runs, "crawl_pages": Pages()}

    scan = await corpus.find_smallest_content_ready_run(
        excluded_run_ids=set(), maximum_pages=50_000
    )
    assert scan.candidate.id == "dated"
