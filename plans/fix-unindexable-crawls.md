# Make GeekAPI stop crawling: it produces runs nothing can index

## Context

The "Crawl it" button, the schedule panel, and stall recovery all drive a crawler that lives inside
GeekAPI and produces runs the Library can never index. Verified by reading code and live config, not
inferred.

**The path.** `start-crawl-form.tsx:33` → `POST /api/geek-crawler/crawls` →
`GeekCrawlerController.StartCrawl:69` → `GeekCrawlerService.StartCrawlAsync` → `_wake.Wake(run.Id)`
(`:145`) → `GeekCrawlerWorker.cs:36` consumes the channel → `ExecuteRunAsync` → `SameOriginBfsCrawler`
→ `MobilePageFetcher` driving headless Chromium (`MobilePageFetcher.cs:77`, `:183`). It is a real
crawler, and `Dockerfile:23` installs Chromium so it works in the deployed image.

**Reason 1 — no corpus body.** `GeekCrawlerPageBatchWriter.cs:23-30` passes seven positional fields
and leaves `Title`, `Excerpt`, `ContentHtml`, `Blocks` at null. The DTO comment admits it outright
(`GeekCrawlerDtos.cs:83-86`): *"Optional so the internal SameOriginBfsCrawler path, **which runs no
extractor**, keeps compiling and behaving unchanged."* The Library is block-based end to end, so an
`Html`-only row has no corpus.

**Reason 2 — no readiness stamp.** `ExecuteRunAsync`'s completion patch
(`GeekCrawlerService.cs:396-404`) sets `Status`, `HostProgressJson`, `CompletedAtUtc` — not
`ContentReadyAt`, not `CrawlReportJson`. Every `ContentReadyAt` write in GeekAPI is in
`GeekCrawlerIngestController` (`:378`), the external route. `mongo.find_smallest_content_ready_run`
filters on it, so such a run is invisible to the RAG scheduler forever.

**It is live in production.** Railway `GeekAPI` production has `GEEK_CRAWLER_WORKER_COUNT=5`,
`GEEK_CRAWLER_SEEDS_ONLY=false`, no `ASPNETCORE_ENVIRONMENT` (so Production, and the local safety
valves are force-disabled there by `GeekCrawlerOptions.cs:37-39`). Five workers, plus stall recovery
scanning every 2 minutes (`GeekCrawlerStallRecoveryHostedService.cs:78`) and the schedule service
scanning every minute and calling `StartCrawlAsync` (`:80`). Registration is unconditional
(`Program.cs:155` → `ServiceRegistration.cs:83`); the default `WorkerCount` is 1, so it would run
even unset.

**The failure is silent, and one log line asserts the opposite of the truth.**
`GeekCrawlerService.cs:566-571` logs *"the run is crawled and **content-ready** but unindexed"* — for
runs that are never content-ready. `GeekCrawlerEventMapper.cs:10-22` omits `contentReadyAt` from the
SignalR frame entirely, so the UI cannot show it even as null. The defect surfaces only much later,
as *"its crawl extracted no content"* from `GccDeclaredUrlEvidence.cs:82`.

**A second defect, worse than the first.** `GeekCrawlerService.cs:174-185` takes a run the external
crawler owns, forces it to `pending`, and wakes the .NET worker. The `external` status exists
precisely so that cannot happen (`GeekCrawlerRunStatuses.cs:7`: *"Owned by an external crawler
(Crawlee); ignored by GeekCrawlerWorker"*), and the isolation is one-way only — re-starting a slot
from the UI **converts a good, indexable external run into an unindexable in-process one.**

## Correcting the premise in the report

The report cited a 2026-09-17 note giving two allowed shapes, one being *"the crawler polls for queued
runs."* **That note does not exist, and that half is the opposite of the written rule.**

- `Geek-Crawler-v2/plans/move-crawl-reads-to-geekapi.md:140-142`: *"**Not an option: the crawler
  polling GeekAPI for queued runs.** That is the prohibited pattern, and this repo's `external` run
  status exists precisely so GeekAPI's own worker does not claim v2 runs."*
- `Geek-Crawler-v2/.cursor/rules/description-prohibit-polling.mdc:47` bans polling outright.
- That file frames start as an **open decision with three candidates** (hub wake / crawl in GeekAPI /
  CLI only), recommending a hub wake. It is unresolved, not settled.

**The two shapes that actually work today** are both operator-local
(`Geek-Crawler-v2/README.md:159-170`): `npm run crawl -- --seed … --type …`, or `POST` to the loopback
serve API on `127.0.0.1:8787`. Geek-Crawler-v2 **does not poll and has no claim endpoint** — its only
GeekAPI calls are the ingest routes plus one run-presence `GET`. And
`POST /api/geek-crawler/ingest/runs` creates a run in `external` state that **nothing crawls**; v2
must already be running locally and pushing into it.

## Measured blast radius: zero surviving runs, but that proves less than it looks

Live `geek_crawler` Mongo, 2026-09-29: 18 runs, all `status=complete`, **all 18 with
`ContentReadyAt`** — so every surviving run came through the external ingest route, none from the
in-process path.

Two caveats that matter. The corpus was wiped earlier today and Dext was purged this evening, so
these 18 are all post-wipe — this is evidence about the current store, **not** proof the button was
never used. And with five workers live in production, the exposure is ongoing regardless of what
survived.

## Decision

Jeff chose: **make the in-process crawler unreachable, keep the code.** Not deleted (large, and some
pieces are shared), not taught to extract (that would put a second implementation of the corpus
projection in C#, which CLAUDE.md §1a forbids by name).

`GeekCrawlerOptions.cs:28-32` already has the intended switch, with a comment naming this exact case:
*"0 is intentional: idle this GeekAPI instance's crawl workers (local Mac / Railway handoff)."*

## Plan

**1. Turn the workers off in production.** Set `GEEK_CRAWLER_WORKER_COUNT=0` on the Railway `GeekAPI`
production service. This is a config change, and it is the only supported off switch. Do it first: it
stops new damage immediately and independently of any code landing.

**2. Do not let the create path hang silently.** With no workers, `StartCrawlAsync` still creates a
run and writes to the wake channel, so it would sit `pending` forever with no error — trading a silent
bad result for a silent no result. `StartCrawl` must refuse with an actionable message naming the two
working shapes, rather than accepting work nothing will do. This is the §3a rule: fail closed, with
diagnostic detail.

**3. Stop the external-run hijack.** `GeekCrawlerService.cs:174-185` must not convert an `external` run
to `pending`. That path silently destroys the indexability of a run the external crawler produced
correctly, and it defeats the isolation `GeekCrawlerRunStatuses.cs:7` exists to provide.

**4. Stop schedules from starting in-process crawls.** `GeekCrawlerScheduleHostedService:80` calls
`StartCrawlAsync` every minute with no environment guard. Whatever item 2 does, this path must reach
the same outcome — a schedule must not be a back door to the crawler the UI can no longer reach.

**5. Fix the log line that lies.** `GeekCrawlerService.cs:566-571` claims "content-ready" for runs
that are not. Per CLAUDE.md §1a, a surviving claim is read as evidence.

**6. Make the gap visible.** Add `contentReadyAt` to `GeekCrawlerEventMapper.cs:10-22` and the SignalR
frame so the UI can render "content ready: no" beside `status: complete`. `ToSnapshot` already carries
it, so `GET /crawls/{id}` returns it — the live frame is the only hole.

**7. A guard that fails closed.** A run must not read `complete` while carrying no `ContentReadyAt`.
The ingest route already refuses a page with no extracted content (`GeekCrawlerIngestController.cs:647`
— *"pages carry no extracted content (contentHtml + blocks required)"*); the equivalent belongs
wherever a run is marked complete, so this class cannot recur behind a config flag.

## Verification

1. **Before/after in Mongo**: count `complete` runs with no `ContentReadyAt`. Zero now; must stay zero.
2. `railway variables` on the production `GeekAPI` service shows `GEEK_CRAWLER_WORKER_COUNT=0`, and the
   startup `GeekCrawlerConfigLogger` line confirms zero workers after redeploy.
3. A test that `StartCrawl` refuses rather than queueing uncrawlable work, and that its message names
   the CLI and loopback shapes.
4. A test that an `external` run stays `external` through the requeue path — the regression that
   silently downgrades a good run.
5. `dotnet test GeekBackend.Tests` (1,239 today), `uv run pytest` (392), frontend `tsc` + build.
6. **End to end through the supported shape**: `npm run crawl -- --seed … --type partner` on the
   operator box → run reaches `complete` with `ContentReadyAt` → RAG picks it up →
   `GET /v1/pages/{id}` returns text.

## Open question for Jeff

**Item 2: refuse, or create an `external` run the local crawler can adopt?** Refusing is honest and
simple but the UI loses its start button entirely. Creating an `external` run keeps the button
meaningful — it becomes "reserve a run for me to crawl locally" — but only works if you then run the
CLI against that run id, which is not a shape v2 supports today (it creates its own runs). I lean
**refuse with a message naming the two working shapes**, because the alternative invents a fourth
intake shape while `move-crawl-reads-to-geekapi.md` still lists the start design as undecided.

## Out of scope

- Resolving that undecided start design (hub wake vs GeekAPI-crawls vs CLI-only). This plan removes a
  broken path; choosing the replacement is a separate decision, and the plan file for it already
  exists in the crawler repo.
- Deleting the in-process crawler, per the decision above.
- `plans/order-index-status-frames.md` — unrelated and smaller.
