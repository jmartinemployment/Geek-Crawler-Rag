# The index-status receiver's persist failure

## What this file used to claim, and why it was wrong

It described an out-of-order defect: GeekAPI's guard is

```csharp
if (run is null || run.RagIndexedAtUtc is null || body.FinishedAtUtc > run.RagIndexedAtUtc)
```

and `finishedAtUtc` is null until a run is terminal, so for every mid-run frame the comparison reduces
to "`RagIndexedAtUtc` is null" — true for the whole run. The conclusion drawn was that a delayed
earlier frame overwrites a newer one, and the proposed fix was a sender-stamped `sentAtUtc` ordering
key plus a cross-repo contract change.

**That cannot happen.** Verified 2026-09-30: the index worker runs at concurrency 1, and the webhook
POST is a plain `await self._webhook.notify(status)` — the next frame is not *sent* until the previous
one has returned. The heartbeat task (`indexer.py:457`) only renews the Mongo lease and emits no
frames. `status_store.save` rejects a stale write on the sender's side as well. Two frames for one run
are never in flight together, so nothing can overtake anything.

The guard has a real hole. Nothing can get through it.

`sentAtUtc` is therefore not being added. It would be machinery guarding a path that does not exist,
and it would cost a two-repo contract change and a deploy window to do it. Revisit only if the sender
ever emits frames concurrently — at which point the comment now in
`GeekCrawlerRagWebhookController.cs` is what says so.

## The real defect, and it is addressed

The same block swallowed its persist failure:

```csharp
catch (Exception ex) { _logger.LogWarning(...); }   // then fell through to Accepted
```

`plans/rules.md` §3a verbatim. The sender logs at ERROR on 401/403 and 400, so a dead integration was
visible — but a failed *persist* answered 202, so the one failure mode that loses
`RagChunksUpserted`, `RagPagesEnglish` and `RagPagesSkippedUnusable` was the only one the sender could
never learn about. Those three are what `GccDeclaredUrlEvidence` reads.

Fixing only the exception would have documented a property no check enforced, because the exception was
not the only silent route: `UpdateOneAsync` has no upsert, so a runId with no document matched nothing,
raised nothing, and was answered 202. That is the realistic case — the crawler purges a failed run from
GeekAPI while the Library still holds an index job for it.

**Both sides are written and awaiting a deploy window:**

- **GeekBackend `2123a1b`** — an absent run is refused 404 before any write;
  `UpdateRagIndexStatusAsync` returns whether a document matched and GeekRepository answers 404 when it
  did not; the persist catch returns 404 or 502 instead of falling through, logs at Error, and excludes
  `OperationCanceledException`. Also corrects the guard's comment, which claimed out-of-order
  protection it does not provide.
- **This repo `406a77c`** — 404 and 5xx get their own ERROR branch on the sender, naming that
  `crawl_runs` still carries stale numbers. An ordinary 4xx stays a warning, pinned by a test.

Receiver first. No contract change, so the two copies still agree at 18 fields.

## What remains

1. **Deploy both.** Receiver before sender. Listed as step 4 of `go-local-embeddings.md`, since that
   change needs the same window.
2. **Correct `$notes.persisted` in both contract copies.** They currently say the receiver *"wraps its
   persist in `catch { LogWarning }` and still returns Accepted, so a persist failure is not visible to
   the sender — durability here is best-effort, not guaranteed."* `2123a1b` makes that false. Leaving it
   is the documentation half of the defect it describes.
3. **`_persist`'s `bool` return is ignored at 11 call sites.** Found during the field-drift work and
   still open: either it should gate its callers or it should go, and changing control flow at 11 sites
   is a behaviour decision rather than a cleanup.

## Out of scope

- `attempt` and `trigger` are bound and pushed but unused by the guard. They carry real values, so they
  are unused telemetry rather than dead fields — a different question from the two throttle fields,
  which `go-local-embeddings.md` handles.
- GeekBackend still runs no CI on pull requests.
