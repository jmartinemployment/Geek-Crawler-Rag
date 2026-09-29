"""The contract is checked against the payload `notify()` really posts, not a copy of it.

GeekAPI binds this payload by name over JSON. Neither repo can see the other's types, so a
field the sender adds is bound by nothing and System.Text.Json discards it without an error, a
log, or a default worth reading. Five fields sat in that state until 2026-09-29 --
`pagesSkippedUnusable`, `attempt`, `trigger`, `embeddingRateLimitRetries`,
`embeddingWaitSeconds` -- and the costly one was `pagesSkippedUnusable`: GeekAPI bound both
benign skip reasons and dropped the one that says the crawl stored pages it should not have, so
a corpus gutted by 4xx bodies reported identically to a clean one.

**The first version of this file re-implemented the producer and was therefore decorative in
four separate ways.** It computed the payload itself -- `{f.alias or name}` plus a hand-typed
`payload["eventType"] = ...` -- which is the "two implementations of one projection" drift
CLAUDE.md 1a names as what broke this pipeline before. All four were confirmed by mutation:

- Deleting `payload["eventType"] = "rag_index"` from webhook.py: **5 passed**, including the
  test named for eventType, whose assertions never touched webhook.py.
- Adding an unaliased multi-word field (`pages_skipped_robots: int = 0`, matching the style of
  the already-unaliased `attempt`/`trigger`/`error`, which are single words and so happen to be
  camelCase already): **5 passed**, and the real wire key was `pages_skipped_robots`. GeekAPI
  matches case-insensitively but does not strip underscores, so that binds to nothing -- the
  exact defect this file exists to prevent, blessed by a green test.
- `serialization_alias` beats `alias` in `model_dump(by_alias=True)`, so `f.alias` computed the
  wrong key for any field that set it.
- Every timestamp was None in the one serialising test, so `mode="json"` was never exercised;
  dropping it passes here and raises TypeError in production on a populated status, where
  webhook.py's deliberately bare `except Exception` swallows it and delivery stops silently.

So the payload now comes from `IndexStatusWebhook.notify()` through a fake transport. There is
one projection, and it is the real one.

This pins the sender only. GeekBackend.Tests/GeekCrawler/RagIndexStatusWebhookContractTests.cs
pins the receiver against a byte-identical copy of the same file, and the cross-repo workflow
diffs the two copies.
"""

from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime, timezone

import pytest

from geek_crawler_rag.models import IndexState, IndexStatusResponse
from geek_crawler_rag.webhook import IndexStatusWebhook

CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "contracts"
    / "rag-index-status"
    / "webhook.v1.json"
)

# GeekAPI binds with PropertyNamingPolicy = CamelCase and matches case-insensitively, but it
# does not strip underscores or hyphens. So this, and not "is it valid JSON", is the shape a key
# has to have to be bindable at all.
BINDABLE_KEY = re.compile(r"^[a-z][a-zA-Z0-9]*$")


def _contract() -> dict:
    """Parse the contract, refusing a duplicate key rather than silently picking one.

    `json.loads` keeps the last occurrence of a repeated key while GeekBackend's
    `JsonDocument...EnumerateObject()` yields both, so a duplicate makes the two repos read
    different documents out of files CI has certified byte-identical -- precisely the drift this
    file was created to close. object_pairs_hook is where that is catchable.
    """

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        seen: set[str] = set()
        for key, _value in pairs:
            if key in seen:
                raise AssertionError(
                    f"{CONTRACT.name} repeats the key {key!r}. Python keeps the last "
                    "occurrence and C# keeps both, so the two repos would disagree about a "
                    "file the cross-repo job reports as identical."
                )
            seen.add(key)
        return dict(pairs)

    return json.loads(CONTRACT.read_text(), object_pairs_hook=no_duplicates)


async def _posted_body(*, minimal: bool = False) -> dict:
    """The actual bytes-worth of JSON that goes to GeekAPI.

    A populated status, not a defaults-only one: both timestamps are set so `mode="json"` is
    exercised, and the result is round-tripped through `json.dumps`/`loads` because that is what
    httpx does with `json=` -- a payload holding a raw datetime passes an in-memory dict
    comparison and raises TypeError on the wire.
    """
    posted: dict = {}

    class FakeResponse:
        status_code = 202
        text = ""

    class FakeClient:
        async def post(self, url, json=None, headers=None):
            posted["json"] = json
            return FakeResponse()

        async def aclose(self):
            return None

    webhook = IndexStatusWebhook("https://api.example/webhook", "secret-key")
    webhook._client = FakeClient()  # type: ignore[assignment]

    if minimal:
        # Only the two required fields. Everything else takes its model default, which is the
        # shape a freshly claimed job posts and the shape that exposes a None default.
        status = IndexStatusResponse(
            run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", state=IndexState.PENDING
        )
        await webhook.notify(status)
        return json.loads(json.dumps(posted["json"]))

    status = IndexStatusResponse(
        run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        state=IndexState.COMPLETE,
        crawl_type="partner",
        mongo_page_count=506,
        pages_seen=506,
        pages_english=46,
        pages_skipped_lang=0,
        pages_skipped_empty=0,
        pages_skipped_unusable=460,
        chunks_upserted=214,
        attempt=2,
        trigger="scheduler",
        embedding_rate_limit_retries=3,
        embedding_wait_seconds=12.5,
        error=None,
        started_at_utc=datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc),
        finished_at_utc=datetime(2026, 9, 29, 10, 4, tzinfo=timezone.utc),
    )
    await webhook.notify(status)

    # json.dumps is the assertion, not decoration: it fails on anything httpx could not send.
    return json.loads(json.dumps(posted["json"]))


@pytest.mark.asyncio
async def test_the_posted_payload_matches_the_contract_exactly():
    body = await _posted_body()
    contracted = set(_contract()["fields"])
    sent = set(body)

    missing = sent - contracted
    extra = contracted - sent
    assert not missing, (
        f"The webhook posts these and {CONTRACT.name} does not list them: {sorted(missing)}. "
        "Add them to the contract, copy the file to GeekBackend unchanged, and bind each one "
        "in RagIndexStatusWebhookRequest -- an unbound field is discarded on arrival."
    )
    assert not extra, (
        f"{CONTRACT.name} lists these and the webhook does not post them: {sorted(extra)}. "
        "A receiver binding one of these holds its default forever, which reads as real data."
    )


@pytest.mark.asyncio
async def test_every_posted_key_is_one_geekapi_can_bind():
    """The hole that let a green test bless an unbindable wire key.

    A field declared without `alias=` serialises under its snake_case Python name. GeekAPI's
    camelCase policy matches case-insensitively but does not strip underscores, so the field
    arrives and is dropped -- and because the old check derived the expected key from the same
    attribute, the contract simply recorded the broken name and passed.
    """
    unbindable = sorted(k for k in await _posted_body() if not BINDABLE_KEY.match(k))
    assert not unbindable, (
        f"These wire keys cannot bind to a C# property: {unbindable}. Give each field an "
        'explicit camelCase alias, e.g. Field(0, alias="pagesSkippedRobots").'
    )


@pytest.mark.asyncio
async def test_the_event_type_the_sender_injects_is_on_the_wire():
    # eventType is not a model field -- notify() sets it after model_dump -- so this can only be
    # checked by running notify(). Asserting it against the model (as the first version did)
    # passes even after the line that produces it is deleted.
    body = await _posted_body()
    assert body["eventType"] == "rag_index"
    assert "eventType" not in {
        f.alias or n for n, f in IndexStatusResponse.model_fields.items()
    }


@pytest.mark.asyncio
async def test_the_reject_count_is_posted_as_a_number_under_its_contracted_name():
    # The field whose absence was the defect, pinned by name and by JSON type so neither a
    # rename nor a nullable-ing can quietly reopen the gap.
    body = await _posted_body()
    assert body["pagesSkippedUnusable"] == 460
    assert isinstance(body["pagesSkippedUnusable"], int)
    assert _contract()["fields"]["pagesSkippedUnusable"] == "int"


@pytest.mark.asyncio
async def test_the_contracted_type_of_every_field_matches_what_is_posted():
    """The type column compared to something, instead of to nothing.

    The previous version's only type assertion compared the checked-in file to a literal typed
    in the same commit. Meanwhile a real drift loses the whole payload, not one field: turning
    `pages_seen` nullable puts `"pagesSeen": null` on the wire, which cannot bind to a
    non-nullable `public int`, so ASP.NET rejects all 18 fields with a 400 that webhook.py
    swallows as a warning -- every Rag* field stays stale forever with green CI.
    """
    body = await _posted_body()
    allowed = {
        "string": (str,),
        "string?": (str, type(None)),
        "int": (int,),
        "int?": (int, type(None)),
        "double": (int, float),
        "double?": (int, float, type(None)),
        "datetime": (str,),
        "datetime?": (str, type(None)),
    }
    wrong: list[str] = []
    for key, declared in _contract()["fields"].items():
        assert declared in allowed, f"{CONTRACT.name}: unknown type {declared!r} for {key}"
        # bool first: it subclasses int, so a bool would satisfy (int,) unnoticed.
        value = body[key]
        if isinstance(value, bool) and declared not in ("bool", "bool?"):
            wrong.append(f"{key}: declared {declared}, posted a bool")
        elif not isinstance(value, allowed[declared]):
            wrong.append(f"{key}: declared {declared}, posted {type(value).__name__}")
    assert not wrong, (
        "Contracted types disagree with the posted payload: "
        + "; ".join(wrong)
        + ". A nullable-ing or a widened type here 400s the entire webhook, not one field."
    )


@pytest.mark.asyncio
async def test_no_non_nullable_field_can_be_posted_as_null():
    """Checked against a DEFAULTS-ONLY status, which is the case that actually bites.

    The populated fixture above sets every field, so it cannot see a field becoming optional:
    making `pages_seen` nullable was mutation-tested against the first version of this test and
    **passed**, because the fixture still posted 506. The production shape is a status carrying
    only what has been filled in so far, and that is where a `None` default reaches the wire.

    A null into a non-nullable C# property is not a dropped field -- `JsonException` makes
    ASP.NET reject all 18, and webhook.py's bare `except Exception` logs a warning and moves on,
    so every Rag* field on the run stays stale forever while CI is green.
    """
    minimal = await _posted_body(minimal=True)
    nulls = sorted(
        key
        for key, declared in _contract()["fields"].items()
        if not declared.endswith("?") and minimal.get(key) is None
    )
    assert not nulls, (
        f"These are declared non-nullable and a defaults-only status posts null for them: "
        f"{nulls}. Either give the model field a non-null default or mark the contract entry "
        "nullable on both copies -- and remember a null 400s the entire payload, not one field."
    )


@pytest.mark.asyncio
async def test_a_defaults_only_status_still_carries_every_contracted_key():
    # Key set must not depend on how much of the status is filled in; a field omitted when unset
    # (exclude_none, or a model_dump kwarg change) is a field the receiver silently defaults.
    assert set(await _posted_body(minimal=True)) == set(_contract()["fields"])
