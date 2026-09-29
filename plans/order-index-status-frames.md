# Order the index-status frames, and make the receiver admit when it did not persist

## Context

The field-drift work landed on 2026-09-29: five fields were posted on every index-status webhook and
bound by nothing, so a corpus gutted by 4xx error pages reported identically to a clean one. While
tracing the receiver for that change, the out-of-order guard turned out to have no ordering key for
most of the frames it guards.

GeekAPI's `GeekCrawlerRagWebhookController` gates persistence on:

```csharp
if (run is null || run.RagIndexedAtUtc is null || body.FinishedAtUtc > run.RagIndexedAtUtc)
```

`finishedAtUtc` is **null until a run reaches a terminal state**, so for every mid-run progress
frame the guard reduces to "`RagIndexedAtUtc` is null" — which is true for the whole run. Every
mid-run frame therefore persists in arrival order with no comparison at all. A delayed earlier frame
overwrites a newer one, and whichever frame happens to land last before the terminal frame is what
sticks in `RagChunksUpserted`, `RagPagesEnglish` and `RagPagesSkippedUnusable`.

I deliberately left this out of the field-drift change rather than picking an ordering design at the
end of a long edit — choosing that badly reintroduces the silent-overwrite class we had just spent
the day removing.

## Two defects in the same block

**1. No ordering key for mid-run frames.** As above. `finishedAtUtc` structurally cannot order them.

**2. The receiver reports success for a persist it did not do.** The `UpdateRagIndexStatusAsync`
call is wrapped in `catch (Exception) { LogWarning }` and the action still returns `Accepted`. That
is `plans/rules.md` §3a **No Fallbacks** verbatim — *"Do not turn a required-operation failure into
apparent success by … skipping persistence, swallowing exceptions"*.

These belong in one change. The sender now logs at ERROR on 401/403 and 400 (done in the field-drift
work) so a dead integration is visible — but a **persist** failure returns 202, so the sender still
cannot learn about the one failure mode that loses data. Fixing ordering while leaving that open
means the new guard can silently reject or silently fail and look identical from the sender's side.

## Design — `sentAtUtc`, stamped at POST time

Add one field, `sentAtUtc`, stamped by `IndexStatusWebhook.notify()` at the moment of the POST, and
gate persistence on the greatest `sentAtUtc` seen for the run.

**Why not a per-run sequence counter.** A sequence needs somewhere monotonic to live, which means
the RAG's job row holds a counter, `status_store.save` writes it, and `_from_doc` reads it back.
`_from_doc` is a hand-maintained field list on the notify path — the one place where forgetting a
field puts it on the wire at its default forever. The receiver would then have to arbitrate between
sequence and timestamp, and getting that wrong either drops good updates or admits stale ones,
silently.

`sentAtUtc` avoids all of it:

- One field, one clock, monotonic per process by construction.
- Stamped in `notify()` after `model_dump`, exactly as `eventType` already is — so it is **not a
  model field**, and `status_store` / `_from_doc` / `IndexStatusResponse` do not change.
- It orders mid-run frames, which is precisely what `finishedAtUtc` cannot do.
- It supersedes the terminal comparison rather than competing with it, so there is one rule.

**`RagIndexedAtUtc` is not repurposed.** It carries a separate, load-bearing meaning — null means
"not indexed", and `scripts/trigger_manual_index.py` writes null to force a re-index. The ordering
key gets its own persisted field, `RagStatusSentAtUtc`.

## Changes

*Verification of the persisted-field chain and the `RagIndexedAtUtc` reader list is in flight; the
lists below are provisional until it lands.*

### Sender — Geek-Crawler-Rag
- `src/geek_crawler_rag/webhook.py`: stamp `payload["sentAtUtc"]` beside the existing `eventType`
  line, from `utc_now()`, ISO-8601 with an offset.
- `contracts/rag-index-status/webhook.v1.json`: add `"sentAtUtc": "datetime"` to `fields`, and a
  `$notes` entry saying it is injected by `notify()` rather than carried on the model, and that it
  is the receiver's ordering key.
- `tests/test_index_status_webhook_contract.py`: the existing `_posted_body()` picks it up through
  the real `notify()`, so the key-set and type tests cover it automatically. Add one assertion that
  two successive `notify()` calls produce strictly increasing `sentAtUtc`.

### Receiver — GeekBackend
- `RagIndexStatusWebhookRequest`: bind `SentAtUtc`.
- The guard becomes: persist when `run.RagStatusSentAtUtc is null || body.SentAtUtc > run.RagStatusSentAtUtc`.
  A frame with no `SentAtUtc` (an older sender mid-deploy) must be treated explicitly — see the open
  question below.
- Persist `RagStatusSentAtUtc` with the other `Rag*` fields.
- **Stop swallowing the persist failure**: on exception, log and return a non-2xx (502, matching the
  existing `PushRagIndexAsync` failure path) so the sender's ERROR branch fires.
- Add `sentAtUtc` to the SignalR frame and the `crawls/{runId}/rag-index` projection, so the two
  agree field-for-field as they now do for everything else.
- Persisted-field chain: entity, both `PatchRagIndexStatusCommand` records, the
  `UpdateRagIndexStatusAsync` signature/interface/`$set`, and `GeekCrawlerRunDto` — **append**, never
  insert, because that record has positional callers.

### Tests
- `PatchRagIndexStatusHopTests` covers the new member automatically (it reflects over the record).
- New receiver tests: a stale frame does not overwrite a newer one; an equal `sentAtUtc` does not
  re-persist; a persist failure returns non-2xx; the SignalR push still fires for a rejected frame
  (it is outside the guard today — confirm and keep that deliberate).

## Open question for Jeff

**What should the receiver do with a frame carrying no `sentAtUtc`?** This happens for the length of
one deploy window, since GeekBackend must merge first. Two defensible answers: accept it (preserves
today's behaviour, so a mid-deploy frame is unordered exactly as now), or reject it (fails closed,
but drops every frame until the RAG deploys). I lean **accept, with a WARNING naming the sender as
out of date** — rejecting would discard real progress for a mismatch that is expected and
self-resolving, and the WARNING makes the window visible. Flagging it because "fail closed" is the
house default and this is a deliberate exception to it.

## Verification

1. **Mutation** — make `notify()` stamp a constant instead of `utc_now()`; the monotonicity test must
   fail. Revert and confirm green.
2. **Mutation** — revert the guard to `FinishedAtUtc`; the stale-frame test must fail.
3. **Mutation** — make the persist throw; the non-2xx test must fail if the `catch` still returns
   `Accepted`.
4. `uv run pytest` (392 today) and `dotnet test GeekBackend.Tests` (1,239 today) both up, none down.
5. `python3 scripts/compare_webhook_contract.py` against both copies — must pass only after both are
   updated, which is the merge-order signal.
6. **Merge order is mandatory**: GeekBackend first (it has no PR CI), then this repo. Expect a red
   window on `main` here between the two merges.
7. **Deploys**: the RAG needs one (`webhook.py` is under `src/`) — never while indexing is active.
   GeekAPI needs one. Neither is optional: a sender stamping `sentAtUtc` against a receiver that does
   not bind it is harmless, but a receiver gating on `RagStatusSentAtUtc` against a sender that does
   not send it hits the open question above on every frame.

## Out of scope

- The RAG runs a single API process today, so a sender-stamped clock is monotonic. If it is ever
  scaled out, `sentAtUtc` stops being a total order and a real sequence becomes necessary.
- `_persist`'s `bool` return is still ignored at 11 call sites. Found during the field-drift work and
  deliberately not fixed there: either it should gate its callers or it should go, and changing
  control flow at 11 sites is a behaviour decision rather than a cleanup.
- GeekBackend still runs no CI on pull requests.
