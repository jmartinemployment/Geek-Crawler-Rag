# Retire the two permanently-zero throttle fields on the wire

What survives `go-local-embeddings.md`, deleted 2026-10-01 once every criterion in it verified.
Nothing here is a consequence of that plan failing; these are the pieces it deliberately left for a
cross-repo window.

## The problem

`embeddingRateLimitRetries` and `embeddingWaitSeconds` are pinned in
`contracts/rag-index-status/webhook.v1.json:21-22`, with a byte-matching GeekBackend copy that
`.github/workflows/cross-repo-citation-contract.yml` diffs on every push.

There is no throttle any more and no metered API to pace, so both can only ever be zero. A field
that is always zero is the same shape as the defect the contract exists to catch — a reader cannot
tell "no retries happened" from "nothing reports retries". That makes it a deviation, not a
non-issue.

Verified: zero references in `content-creator-v2`, and the contract's own `$notes.persisted`
confirms neither reaches `crawl_runs`. They stop at SignalR and the `rag-index` projection.

## The change

- **Delete `embeddingRateLimitRetries`.** No local analogue exists.
- **Rename `embeddingWaitSeconds` → `embeddingSecondsTotal`**, carrying real dense-inference seconds
  per run. Renamed, not repurposed: reporting inference time under `...WaitSeconds` is a quiet lie
  the next reader takes at face value. Already plumbed end to end — `Index timing` logs
  `embedSeconds` per run (measured 358.5s of 395.8s total on dext, 2026-10-01).
- **Add `truncatedInputs`.** `indexer._sync_embedding_stats` already computes it as a per-run delta
  and deliberately does not set it on `IndexStatusResponse`, because adding a key reds the contract
  diff until the receiver binds it. It belongs in this same window.

Removal is runtime-safe in either merge order: an absent field reads as zero, an unknown member is
ignored by the receiver.

## Sites

Sender: `models.py`, `indexer.py`, the three `embedding_stats` mocks, `README.md:284-285`.
Receiver: `GeekCrawlerRagWebhookController.cs:125-126,180,182`, `GeekCrawlerController.cs:220-221`,
`HttpGeekCrawlerRagClient.cs:212-213,342-343,474`, and its contract test.

While in there: `$notes.persisted` in both copies describes the receiver swallowing a persist failure
and returning `Accepted`. GeekBackend `2123a1b` makes that false.

## Verification

`python3 scripts/compare_webhook_contract.py <sender.json> <receiver.json>` must agree on the field
set, and `uv run pytest` must hold at its current count.

## Also outstanding, unrelated to the wire

- **Rotate the exposed OpenAI key.** Required regardless of OpenAI being gone from this repo; the key
  is live for GeekAPI content generation. New key → Railway `LlmProviders__OpenAi__ApiKey` →
  `GeekBackend/.env` → then revoke the old one.
- **Confirm what the existing backup covers.** Single Qdrant node, one volume, no Qdrant-side
  snapshots.
