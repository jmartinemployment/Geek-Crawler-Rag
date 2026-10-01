# `_persist`'s bool return is ignored at 11 call sites

All that remains of `order-index-status-frames.md`, deleted 2026-10-01 once its other two items
closed. Kept separate because this one is a behaviour decision, not a cleanup.

## The item

`IndexService._persist` returns whether the status was written. Eleven call sites discard it. Either
it should gate its callers — a frame that did not persist is a frame the receiver has no record of —
or it should return `None` and stop implying a check nobody makes. Changing control flow at eleven
sites is the part that needs a decision rather than a patch.

## What closed, so it is not re-investigated

**`sentAtUtc` frame ordering — rejected, and deliberately.** The receiver's guard

```csharp
if (run is null || run.RagIndexedAtUtc is null || body.FinishedAtUtc > run.RagIndexedAtUtc)
```

reduces to "`RagIndexedAtUtc` is null" for every mid-run frame, since `finishedAtUtc` is null until a
run is terminal. That reads as an out-of-order hole. It cannot be reached: the index worker runs at
concurrency 1 and the webhook POST is a plain `await self._webhook.notify(status)`, so the next frame
is not sent until the previous returns. The heartbeat renews the Mongo lease and emits no frames, and
`status_store.save` rejects a stale write on the sender's side too. Two frames for one run are never
in flight together.

The guard has a real hole and nothing can get through it. Adding `sentAtUtc` would be machinery
guarding a path that does not exist, at the cost of a two-repo contract change. Revisit only if the
sender ever emits frames concurrently; the comment in `GeekCrawlerRagWebhookController.cs` says so.

**The silent-persist defect — fixed and deployed.** GeekBackend `2123a1b` (pushed 2026-09-30) refuses
an absent run 404 before any write and answers 404 or 502 instead of falling through to Accepted;
`UpdateOneAsync` has no upsert, so a runId with no document previously matched nothing, raised
nothing, and was answered 202. Geek-Crawler-Rag `406a77c` (deployed 2026-10-01) gives 404 and 5xx
their own ERROR branch naming that `crawl_runs` still carries stale numbers.

**`$notes.persisted` corrected in both contract copies, 2026-10-01.** It had described the receiver
swallowing a persist failure and returning Accepted, which `2123a1b` made false — the documentation
half of the defect it described. Both copies now state that a 202 means the five fields were written.
Verified byte-matching at 18 fields by `scripts/compare_webhook_contract.py`.

## Out of scope

- `attempt` and `trigger` are bound and pushed but unused by the guard. They carry real values, so
  they are unused telemetry rather than dead fields.
- GeekBackend still runs no CI on pull requests.
