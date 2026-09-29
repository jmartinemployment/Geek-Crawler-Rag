"""Every field the index-status webhook sends must be in the contract file.

GeekAPI binds this payload into `RagIndexStatusWebhookRequest` by name. Neither repo can see
the other's types at compile time, so a field added here is bound by nothing until somebody
notices -- and on 2026-09-29 five of them were in exactly that state, sent on every webhook
and read by no one:

    pagesSkippedUnusable  attempt  trigger  embeddingRateLimitRetries  embeddingWaitSeconds

`pagesSkippedUnusable` was the costly one. It is the count of pages the Library refused as not
citable -- a 4xx/5xx body, robots-denied, a non-English locale path, a recorded fetch failure --
and GeekAPI bound both of its benign siblings (`pagesSkippedLang`, `pagesSkippedEmpty`) while
dropping it. So a run reporting pagesSeen=506 pagesEnglish=46 with both bound skip counts at 0
looked the same whether the site was small or 460 pages of error bodies had been thrown away.

This test fails when a field is added or removed without updating the contract, which is the
signal for the GeekBackend side to bind it. It does not prove GeekAPI binds anything -- that is
GeekBackend.Tests/GeekCrawler/RagIndexStatusWebhookContractTests.cs, against a byte-identical
copy of the same file, and the cross-repo workflow diffs the two copies.
"""

from __future__ import annotations

import json
import pathlib

from geek_crawler_rag.models import IndexState, IndexStatusResponse

CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "contracts"
    / "rag-index-status"
    / "webhook.v1.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT.read_text())


def _sent_aliases() -> set[str]:
    """Exactly what goes on the wire.

    `webhook.notify` does `status.model_dump(by_alias=True, mode="json")` and then sets
    `eventType`, so the payload is the model's aliases plus that one key.
    """
    aliases = {f.alias or name for name, f in IndexStatusResponse.model_fields.items()}
    aliases.add("eventType")
    return aliases


def test_the_contract_lists_every_field_that_is_sent():
    missing = _sent_aliases() - set(_contract()["fields"])
    assert not missing, (
        f"These fields are sent on the webhook and absent from {CONTRACT.name}: "
        f"{sorted(missing)}. Add them to the contract AND bind them in GeekAPI's "
        "RagIndexStatusWebhookRequest -- an unbound field is silently discarded."
    )


def test_the_contract_lists_nothing_that_is_not_sent():
    extra = set(_contract()["fields"]) - _sent_aliases()
    assert not extra, (
        f"{CONTRACT.name} lists fields the webhook does not send: {sorted(extra)}. "
        "A receiver binding one of these gets a default forever."
    )


def test_eventtype_is_in_the_contract_although_it_is_not_on_the_model():
    # The one key that is added after model_dump. It is on the wire, so it is in the contract;
    # deriving the contract from the model alone would miss it.
    assert "eventType" in _contract()["fields"]
    assert "eventType" not in {
        f.alias or n for n, f in IndexStatusResponse.model_fields.items()
    }


def test_the_reject_count_is_on_the_wire_and_under_contract():
    # The specific field that was dropped, pinned by name so a rename cannot quietly undo it.
    assert "pagesSkippedUnusable" in _sent_aliases()
    assert _contract()["fields"]["pagesSkippedUnusable"] == "int"


def test_a_real_payload_serialises_with_every_contracted_key():
    # The end-to-end shape, not just the field list: build a response and dump it the way the
    # webhook does, then check the JSON itself.
    status = IndexStatusResponse(
        runId="11111111-1111-1111-1111-111111111111", state=IndexState.COMPLETE
    )
    payload = status.model_dump(by_alias=True, mode="json")
    payload["eventType"] = "rag_index"
    assert set(payload) == set(_contract()["fields"])
