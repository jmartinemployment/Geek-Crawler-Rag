from __future__ import annotations

import pytest

from geek_crawler_rag.mongo import MongoCorpus


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
    def __init__(self):
        self.query = None

    async def count_documents(self, _query):
        return 7

    def find(self, query, _projection, **_kwargs):
        self.query = query
        return FakeCursor(
            [{"Id": "large"}, {"Id": "small"}, {"Id": "done"}, {"Id": "zero"}]
        )


class FakePages:
    async def count_documents(self, query):
        return {"large": 60_000, "small": 12, "done": 2, "zero": 0}[query["RunId"]]


@pytest.mark.asyncio
async def test_run_level_readiness_selects_smallest_and_reports_exclusions():
    runs = FakeRuns()
    corpus = MongoCorpus.__new__(MongoCorpus)
    corpus._db = {"crawl_runs": runs, "crawl_pages": FakePages()}

    scan = await corpus.find_smallest_markdown_ready_run(
        excluded_run_ids={"done"},
        maximum_pages=50_000,
    )

    assert runs.query["MarkdownReadyAt"] == {
        "$exists": True,
        "$nin": [None, ""],
    }
    assert scan.candidate is not None
    assert scan.candidate.id == "small"
    assert scan.candidate.page_count == 12
    assert scan.missing_ready_marker == 7
    assert scan.zero_pages == 1
    assert scan.safety_cap == 1
    assert scan.excluded == 1
