"""A sitemap is a link index, not a page.

netsuite.com/portal/sitemap.shtml is 143,292 characters across 3,092 blocks -- three times the next
largest page in that 907-page crawl, and the only one over 60k. Chunked and flushed, its upsert body
stalled the socket write and failed the run five times: httpx.WriteTimeout inside
_send_request_body, while Qdrant sat at 0.17% CPU answering every other request in milliseconds.

It is also worthless as corpus. A sitemap carries no prose a citation could come from, and it is the
one page shape whose size scales with the whole site rather than with its own content -- so the
bigger the site, the worse it gets.

Two earlier fixes did not help, and the reason is instructive: raising the Qdrant client timeout
from 5s to 60s and halving the ONNX thread count both addressed contention, and this was never
contention. The body was too large to write.
"""

from __future__ import annotations

from geek_crawler_rag.unusable import classify_unusable_page, is_sitemap_url


#: Every sitemap page in the live corpus on 2026-10-02, found by scanning all 47 crawls. Listed
#: verbatim because the first pattern matched only 9 of them: `product-sitemap` and `html-sitemap`
#: put the qualifier BEFORE the word, and a rule written from one example missed both.
REAL_SITEMAPS = (
    "https://www.avalara.com/us/en/sitemap.html",
    "https://www.avidxchange.com/sitemap/",
    "https://www.sage.com/en-us/sitemap/",
    "https://www.highradius.com/product-sitemap/",
    "https://www.highradius.com/sitemap/",
    "https://www.netsuite.com/portal/sitemap.shtml",
    "https://stripe.com/sitemap",
    "https://www.accountingseed.com/html-sitemap/",
    "https://www.netsuite.com/portal/fr/sitemap.shtml",
    "https://www.netsuite.com/portal/es/sitemap.shtml",
    "https://www.liveplan.com/sitemap",
)


def test_every_sitemap_in_the_live_corpus_is_matched():
    missed = [u for u in REAL_SITEMAPS if not is_sitemap_url(u)]
    assert not missed, f"real sitemap pages not matched: {missed}"


def test_the_sitemap_shapes_that_appear_in_real_crawls():
    for url in (
        "https://www.netsuite.com/portal/sitemap.shtml",
        "https://example.com/sitemap.xml",
        "https://example.com/sitemap_index.xml",
        "https://example.com/sitemap-1.xml",
        "https://example.com/sitemap",
        "https://example.com/SiteMap.XML",
    ):
        assert is_sitemap_url(url), url


def test_pages_that_merely_mention_sitemaps_are_kept():
    """The rule matches the segment that NAMES the page, not the word anywhere in the path.

    An article about building a sitemap is ordinary citable content. `sitemapping-guide` is the
    case that caught the first pattern out: `sitemap` is a prefix of it, so `^sitemap[\\w.-]*$`
    matched and would have deleted a guide.
    """
    for url in (
        "https://example.com/blog/how-to-build-a-sitemap",
        "https://example.com/resources/sitemapping-guide",
        "https://example.com/sitemap/products/widget",
        "https://example.com/pricing",
        "https://example.com/",
    ):
        assert not is_sitemap_url(url), url


def test_a_sitemap_is_refused_even_though_it_has_plenty_of_blocks():
    """The check has to precede the content checks, because a sitemap passes all of them.

    3,092 blocks clears every prose floor in the pipeline. Ordering this after the block check
    would leave the rule unreachable for exactly the page that motivated it.
    """
    blocks = [{"kind": "listItem", "text": f"Link {i}"} for i in range(3092)]
    assert classify_unusable_page(
        url="https://www.netsuite.com/portal/sitemap.shtml", blocks=blocks
    ) == "sitemap"


def test_an_ordinary_page_with_the_same_blocks_is_kept():
    blocks = [{"kind": "paragraph", "text": "Real prose about accounts payable automation."}]
    assert classify_unusable_page(url="https://example.com/pricing", blocks=blocks) is None


def test_the_reason_is_its_own_and_not_folded_into_no_content():
    """A sitemap is excluded by policy, not because extraction failed.

    Reporting it as `no_content` would read as a crawler fault on a page that extracted perfectly.
    """
    reason = classify_unusable_page(
        url="https://example.com/sitemap.xml", blocks=[{"kind": "listItem", "text": "x"}]
    )
    assert reason == "sitemap"
    assert reason != "no_content"


def test_a_skipped_sitemap_is_not_counted_as_empty():
    """It has 3,092 blocks. Counting it as skipped-empty would report a fault that did not occur.

    No new wire field is added: pagesSkippedEmpty / Lang / Unusable are pinned by
    contracts/rag-index-status/webhook.v1.json, so a sitemap counts only in the unusable total,
    alongside the other policy exclusions.
    """
    from unittest.mock import MagicMock

    from geek_crawler_rag.indexer import IndexService
    from geek_crawler_rag.models import IndexState, IndexStatusResponse

    status = IndexStatusResponse(run_id="r", state=IndexState.RUNNING)
    IndexService._skip_unusable(MagicMock(), MagicMock(), "sitemap", status)

    assert status.pages_skipped_unusable == 1
    assert status.pages_skipped_empty == 0
    assert status.pages_skipped_lang == 0
