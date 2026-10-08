# Handoff — Geek-Crawler-Rag, 2026-10-08

For whoever picks this repository up next. Checked on 2026-10-06 against the repositories, the VPS
and live Mongo/Qdrant, and on 2026-10-08 for the hybrid fix; where something was not checked, it
says so. Authority for rules is
`/Users/jeffmartin/development/.claude/CLAUDE.md`. The plan is in `content-creator-v2/plans/`:
`fix-overview.md` (rules, decisions, wave order) and `fix-geek-crawler-rag.md` (this repo's stages;
read its **Status** section first). This file says where things stand and what to do next.

## 1. What this repository is

The Library: retrieval and quote verification over the crawl corpus. **It never generates.** It
indexes a crawl run (Mongo `crawl_pages` blocks → chunks → Qdrant `geek_crawler_chunks`), answers
`POST /v1/query` for GeekAPI, serves page text, and answers `POST /v1/verify`. Production is the
Hostinger VPS `root@2.24.101.90` (key `~/.ssh/hostinger_rag_ed25519`); the live compose is
`/docker/geek-crawler-rag/docker-compose.yml`, not `deploy/`.

**Pushing to `main` deploys** (build → GHCR → VPS) and recreates the API container, which kills any
index job in flight. There are no branches.

## 2. Rules that shape the work, by where each came from

**Jeff, in the 2026-10-04 and 2026-10-06 sessions (his words quoted).**

1. **Indexing a run replaces it; it never merges.** Every attempt deletes the run's points, then
   writes the run whole. There is no resume. *"Never have two URLs the same, delete or update as
   appropriate. RAG Indexing seems to me as a delete."*
2. **Hybrid always.** *"HYBRID ALWAYS?"* Retrieval is dense (meaning) + sparse BM25 (keyword) in one
   Qdrant query; there is no dense-only mode and no setting to make one. **Until 2026-10-08 the
   query omitted `query_str`, the one argument that makes LlamaIndex run the keyword half, so every
   production query ran dense-only while labelled `llamaindex-hybrid`.** `query_str` was never in any
   commit: `448b25c` (built hybrid, 09-24) and `529a5df` (hybrid always, 10-04) both set
   `mode=HYBRID` without it, and the hybrid test called the store directly with `query_str`, so it
   proved LlamaIndex, not `dense_query`. Fixed 2026-10-08 by Jeff's instruction ("fix");
   `tests/test_dense_query_is_hybrid.py` goes through `dense_query` and asserts the fusion ran.
   Every retrieval measurement before that date (R2's before/after, the earlier near-copy numbers)
   was taken on dense-only retrieval.
3. **Log before any fallback, so a failure can be fully understood** (2026-10-06). Every failure
   path of hybrid retrieval is logged, including the silent one where a half returns nothing (§8).
   Nothing may ever be built that degrades a path without first logging what failed.
4. **Jeff queues re-indexes himself, one site at a time.** *"I will queue sites to be re-index as i
   need them no shotgun approach."* (A 62-run re-index was queued on 2026-10-04 and killed.)
5. **The scheduler stays on** (2026-10-06: "put scheduler default back on"). It is the only automatic
   catch-up when GeekAPI's index enqueue is lost (GeekAPI fails closed and only logs; on 2026-09-24
   eleven crawls completed unindexed that way). Every 300s it indexes the oldest content-ready run
   that has no job row. It also re-picks a SIGKILLed job once its lease lapses (bounded by
   `INDEX_SCHEDULER_MAX_ATTEMPTS`); it skips any run with a `complete`/`failed`/`skipped` row. Note:
   deleting a run's job row (`DELETE /v1/index/runs/{id}`) makes that run a candidate again within 5
   minutes. It defaults on in `config.py` and the repo compose, and production sets it on. **If it
   is ever to be stopped, that is Jeff's call, and the method is `POST /v1/index-scheduler/pause`
   with a reason** (recorded in Mongo, shown by `GET /v1/index-scheduler`, undone by `/resume`, no
   restart). Editing the env flag recreates the container and kills any job in flight.

**The plans writer's instruction list (2026-10-04, relayed by Jeff).**

6. Re-index only by R1's rule: one run at a time, only runs a saved project declares, each checked at
   `GET /v1/index/{run_id}` before the next is posted. Never the corpus, never several at once,
   never the `requeue-stranded.sh` cron.
7. No push while any index job is pending or running. Check `rag_index_jobs` first (§6).
8. Do not touch the production scheduler (see 5).
9. **Measurements ask what GeekAPI asks.** A testing rule, not an app requirement; it changes nothing
   in the app. **Since GeekBackend `bfd99c9` (2026-10-08) GeekAPI sends the keyword alone** —
   `GccGroundingResolver.BuildNeed` returns `GccTopic.KeywordOf(topic)` (the part after the
   `descriptor:` colon, or the whole topic without one), trimmed, capped at 150 — with `topK` 32 and
   `crawlType` `partner`; the competitor query is unchanged. From `684cc59` (10-02) to `bfd99c9`
   it appended 22 fixed words (`" -- the cost, delay and error rate of the manual or status-quo
   way, the capability that removes it, and measured outcomes"`), a stand-in for the brief's niche
   framing that the method never read; once the keyword half ran, every one of those words was a
   search term, and Jeff had them deleted. Every measurement in this file dated before `bfd99c9`
   that says "GeekAPI question" or "`BuildNeed`" used that 25-word text. Copy the text from the
   current GeekBackend code; if `BuildNeed` changes, the measurement follows it.
10. Commit to `main`. No branches.

**The session's own inference, not stated as a rule.**

11. Do exactly what was asked; when a plan or a measurement suggests more, report and ask. Do not
    override a recorded plan decision on your own judgement.

**Standing rules, older than this week.**

12. Fail closed, no fallbacks, no Markdown — `CLAUDE.md` §1a, §2; `fix-overview.md` §0.
13. One session per repository — content-creator-v2 `AGENTS.md`. Read other repos, change nothing
    there. A contract change is committed on both sides together (the plan accepts that).

## 3. What is deployed right now

| Repository | `main` | Deployed | Checked |
|---|---|---|---|
| Geek-Crawler-Rag | the hybrid fix commit of 2026-10-08 (`query_str=need`), pushed | VPS, deploys on push; after it, a live `/v1/query` must leave a `hybrid_halves runId=… dense=N sparse=N fused=N` line in `docker compose logs api` — that line is the proof the keyword half ran, nothing else is | 2026-10-08 |
| GeekBackend | `5671288`, working tree clean | Railway, not checked from here | 2026-10-06 |
| content-creator-v2 | origin `67456d3`; 2 local unpushed commits from another session | — | — |

This repo's commits this week, in order: `1a25095` (R1/R4/R5 first cut), `03b02c7` (always delete,
resume removed, status round-trip fix), `529a5df` (hybrid always), `9a0c901` (R4 fixed, F-R10),
`5e622b6` (R2, scroll deleted), `2cd443f` (scheduler default on, hybrid half logging). GeekBackend
`1c553fc` (webhook field `chunksSkippedRepeat`), `a8e284d` (C# half of the block-projection
fixture). content-creator-v2 `9f41697` (the six re-indexed runs, in the Rag plan's status).

## 4. This repo's stages (`fix-geek-crawler-rag.md`)

| Stage | Wave | State |
|---|---|---|
| **R1** one point per distinct text per run | 1 | **Done.** `textDigest` on every point; `indexer._admit_page_nodes`; `chunksSkippedRepeat` in status and webhook. Met on the old Ramp run (33,728 → 26,683 points). |
| **R4** verify route | 1 | **Code done** (`9a0c901`): one digest `sha256(contentHtml)`, `verify_citations` deleted, `found` the only verdict, F-R10 fixture. **Done-when waits on GeekAPI A1 full** (Wave 2): its verify pass must call `/v1/verify` and keep no comparison of its own. |
| **R5** readiness fail-closed | 1 | **Done.** `POST /v1/index` and the scheduler's entrance refuse a run without `ContentReadyAt` (409, no job row). Scheduler default: on (rule 5). |
| **R2** keyword scroll deleted (D15) | 2 | **Done early** (`5e622b6`). Distinct pages before/after on the old Ramp run with `BuildNeed` questions: 20/19/28/25/23, identical — **measured on dense-only retrieval**, before the 2026-10-08 hybrid fix; re-measure on Ramp `4563f7ec`. On 2026-10-08, before the fix, the five "test" partners returned 32 passages from 19–23 distinct pages each. |
| **R3** collapse before the cut; measure near-copies | 2 | **Selection rule done 2026-10-08** (Jeff: "proceed to fully implement this"): `_select_ranked_candidates` ranks passages then selects pages — every page's best first, then second-best, until `topK`; text dedupe and sibling collapse unchanged. Before/after on the live box is in §9. The near-copy measurement (same paragraph, merchant name swapped) is still open. |
| **R6** tests | 2 | Done: cross-page collapse, verify route, readiness, F-R10, hybrid half logging, `dense_query` hybrid, hosts crawl type, **the flooded-pool test** (`tests/test_page_diverse_selection.py`, through the service). "Ranked lexical list" is moot now the list is deleted. |
| **R7** README drift | 3 | Open. Memory limit, upsert delay and retry wording still disagree with code. |

**The Rag plan's Status section is stale on R4:** it lists the three R4 defects as open; all three
are fixed in `9a0c901`. It also predates `2cd443f` (scheduler on, logging). The plan writer owns it.

## 5. Outstanding steps in the whole plan (`fix-overview.md` §4), as of 2026-10-06

Read from each project plan and the repos' logs on 2026-10-06. "Committed" means a commit names the
stage; it does not mean its done-when was checked. Each plan's own status section is the authority.

**The gate, next.** Wave 1 is built in every project, and persistence P0 has shipped (GeekBackend
`6bef275`…`aefc443`, `a2a559d`; frontend `HANDOFF.md`). The proof as the plan words it: one
Generate on the Accounts Payable project ("test"), all seven live types (five tool pages, a pillar,
a blog, one cold email, one social piece, one image-prompt set, one ads set), or each refused by
name; Jeff reads them; every quote on every page is found on the page it cites. **Where it stands
(Jeff, 2026-10-06):** Jeff ran two Generates on "test" on 2026-10-05 — the first with serious
errors, the second (18:24 UTC) confirming the fixes and exposing an unwanted drafts history, since
removed (GeekBackend `b578480`). Both selected only tool, blog and pillar; he had not realised cold
email and social had to be selected, and the ads set was not selected either. He questions the
standalone image prompt as a proof type (he sees it as an on-demand retry, not a Generate output) —
a plan change for the plan writer. **He will run a third Generate with the missed types once the
projects reach a holding stage.** The two failed runs of 2026-10-06 10:05/10:06 UTC (OpenAI
credits; Anthropic empty body, F-A19) were started by another session and are not his. Wave 2 work
in this repo waits on his read of the third run.

| Wave | Project | Stage | State |
|---|---|---|---|
| 1 | Rag | R1, R4, R5 | Built (R4's done-when waits on A1 full) |
| 1 | Crawler-v2 | C1+C2, C3, C5 | Done (`195e2df`…`106b8b9`, `d8e4341`) |
| 1 | GeekAPI | A1 interim, A2, A5, A6, A7, A16 | Committed (`185df80`, `aab048f`, `3087efa`, `2775051`, `8917237`, `ebbaea9`, `f420d7b`) |
| 1 | GeekRepository | D1 | Committed (`db918bb`) |
| 1 | Frontend | F4, F5, F6 | Done (`3b3134e`…`683f186`) |
| **1** | **all** | **Accounts Payable Generate, read by Jeff** | **Not run** |
| 2 | Rag | R3 (measure), R6 rest | Open (R2 done early) |
| 2 | Crawler-v2 | C4; re-crawl ramp, bill, a third declared partner | C4 done early; ramp `4563f7ec`, bill `e17ef3c0`, lightyear `84f4f34f` re-crawled and indexed 2026-10-05. Open: the per-directory breakdown of `refused.directoryCap` needs a decision (ledger change) |
| 2 | GeekAPI | **A1 full** (verify pass calls `/v1/verify`), A4, A13, A14 | A13 committed (`687242a`); A14 kept advisory (`ebbaea9`); **A1 full and A4 open** |
| 2 | GeekRepository | D2, D3, D4 | D2 committed (`a117eec`); D3, D4 not found in the log |
| 2 | Frontend | F2, F3 | Done early |
| 3 | Rag | R7 | Open |
| 3 | GeekAPI | A3, A8, A9, A10, A11, A12; also A17, A18, A19 | A11 partial (`5248c22`, latest-run read), A12 committed (`62efbad`); A3, A8, A9, A10, A17, A18, A19 not found in the log |
| 3 | Frontend | F1; F7–F11 | F1 half done (second half needs A11); F7–F11 held, each needing a GeekAPI route that does not exist yet (frontend `HANDOFF.md` §5) |
| 4 | GeekAPI | A15 (delete the dormant v2 cluster) | Open, last |
| P1 | Persistence | GR4, GR5, GA3, GA4, GA5 | Open (`fix-project-persistence.md`) |
| P2 | Persistence | GR6, GA6, GF5 (remove the create rows and routes) | Open, only after the report is empty and the proof is read |

Also open, not in a wave: the GeekAPI plan's review items for A6 (figure grammar, strict-subset
retry rule) and A13 (two types in parallel record only their own models) — check its status section
before assuming them done.

## 6. How to check things

```bash
# Is anything indexing? (run before every push)
ssh -i ~/.ssh/hostinger_rag_ed25519 root@2.24.101.90 \
  'cd /docker/geek-crawler-rag && docker compose exec -T api python -' <<'EOF'
import os, asyncio
from motor.motor_asyncio import AsyncIOMotorClient
async def m():
    db = AsyncIOMotorClient(os.environ["MONGO_CRAWLER_URL"])[os.environ.get("MONGO_CRAWLER_DB","geek_crawler")]
    print([d async for d in db["rag_index_jobs"].find({"state":{"$in":["pending","running"]}},{"_id":0,"runId":1,"state":1})])
asyncio.run(m())
EOF
```

- One run's job: `GET /v1/index/{runId}` (header `X-API-Key: $API_KEY` inside the container; the
  image has no `curl` — use `python -c` with `httpx`).
- A run's points: Qdrant `count` with `runId` = run, and again with `textDigest` present; both equal
  to `chunksUpserted` means it was rewritten whole on current code.
- Hybrid health: `docker compose logs api | grep -E "hybrid_halves|hybrid_half_empty"` (§8).
- `uv run pytest` (≈497 tests). Cross-repo CI: `.github/workflows/cross-repo-citation-contract.yml`.
- Do not print a range of the VPS compose: `MONGO_CRAWLER_URL` carries the Mongo password. Grep keys.
- `docker compose exec -T` reads stdin: two in one ssh line and the second gets nothing. One per call.
- A regex over `crawl_pages.Origin` scans the whole collection (minutes). Prefer `RunId`.
- The GitHub API times out intermittently from here; re-list runs rather than trusting one watch.

## 7. Where the code is

| Concern | File |
|---|---|
| Index a run: delete, page loop, repeat collapse, readiness gate | `src/geek_crawler_rag/indexer.py` (`_index_run`, `_admit_page_nodes`, `readiness_refusal`) |
| Chunks → nodes, `textDigest`, `sourceDigest` | `src/geek_crawler_rag/llama_nodes.py` (`text_digest`, `page_source_digest`) |
| Embedding, hybrid query, hybrid half logging | `src/geek_crawler_rag/llama_engine.py` (`dense_query`, `logged_relative_score_fusion`) |
| Query: hybrid candidates (Qdrant fusion) → pool cut → (Cohere off) → page-diverse select. The in-process BM25 re-rank + RRF is **disabled 2026-10-08**, kept commented in `_query_hybrid` with restore instructions | `src/geek_crawler_rag/query.py` (`_query_hybrid`, `_select_ranked_candidates`, `_page_key`) |
| Verify route, page reads, shared refusal helper | `src/geek_crawler_rag/app.py` (`verify_quotes`, `_citable_page_text`) |
| Quote check and the F-R10 normalisation | `src/geek_crawler_rag/citation_verify.py` (`verify_quote`) |
| Block → text, the one projection | `src/geek_crawler_rag/block_text.py` |
| Scheduler, and its pause | `src/geek_crawler_rag/scheduler.py`; routes `/v1/index-scheduler*` in `app.py` |
| Job rows (Mongo `rag_index_jobs`) | `src/geek_crawler_rag/status_store.py` |
| Wire contracts, copied byte-identically into GeekBackend | `contracts/rag-index-status/webhook.v1.json`, `contracts/block-projection/v1.json` |

## 8. Logging, and known gaps

**What a hybrid failure leaves in the log (`2cd443f`):**

| Failure | Log |
|---|---|
| Hybrid query raises (Qdrant down, keyword model error) | `Query failed for runId=…` with traceback (`query.py`); caller gets `retrieval="error"` |
| `query_str` missing again (the hybrid branch not taken) | **Nothing** — LlamaIndex's dense-only fall-through calls no fusion, so no `hybrid_*` line appears. A query with no `hybrid_halves` / `hybrid_half_empty` line ran dense-only. `test_dense_query_is_hybrid.py` pins the argument. |
| One half returns nothing | `WARNING hybrid_half_empty runId=… dense=N sparse=0 fused=N -- this answer used the meaning half only` (or the keyword half, or neither). Ranking is unchanged; this only records it. |
| Both halves answer | `INFO hybrid_halves runId=… dense=N sparse=N fused=N`, one line per query |
| Keyword model fails to load | `Sparse (BM25 keyword) encoder failed to load model=… vector=…` with traceback, then startup stops |
| Collection lacks the keyword vector, or it is not IDF | startup stops with a message naming it (`qdrant_store.py`) |
| Keyword vectors fail while indexing | `Index failed for runId=…` with traceback; the job is `failed` with the error |

At startup: `Hybrid retrieval: dense model=…, sparse (BM25) model=… on vector 'text-sparse'`.

**Known and open.**
- Cross-repo CI is red on four GeekBackend tests (`RagClientContractTests` retry-model, 404s). They
  failed before this work and are GeekBackend's.
- `diagnostics.py:629` checks a caller-declared `sourceDigest` over a supplied document's text — a
  second meaning of the name, for non-corpus documents. Left as is; flagged to the plan writer.
- `lexicalScore` on `ChunkHit` is always null; kept because it is on the wire to GeekAPI.
- Curly vs straight quotes are not folded, by instruction: both projections keep the page's
  characters. A model that straightens an apostrophe gets `found: false`.
- `CLAUDE.md` §1a names `mongo.find_smallest_content_ready_run`; the code is
  `find_oldest_content_ready_run`.
- `plans/partner-evidence-reaches-the-writer.md` is untracked here and not this session's.
- Cohere rerank is off in production (no key), so ranking is RRF over hybrid + BM25 (F-R7).
- Ramp's passages are mostly `/blog` (21–27 of 32) on the old run; the re-crawl (`4563f7ec`) follows
  off-sitemap links (C1) and has not been measured.

## 9a. Page-diverse selection — before and after (2026-10-08, live box, topK 32, crawlType partner)

Measured on the same two runs, same questions, the morning's deploy (`0eb139c`, hybrid on,
rank-order selection) against `3092ccd` (page-diverse selection). "GeekAPI question" is the
25-word `BuildNeed` text that was live until GeekBackend `bfd99c9` later the same day; "bare" is
the keyword alone, **which is what GeekAPI sends now**. Both were taken with the in-process BM25
re-rank still on; it was disabled after (see §7 and §10).

| Run | Question | Before: pages / max per page | After: pages / max per page |
|---|---|---|---|
| tipalti.com partner `39bbce59` | GeekAPI question | 12 / 7 | **24 / 2** |
| tipalti.com partner `39bbce59` | bare keyword | 21 / 6 | **30 / 2** |
| stampli.com `ab551881` | GeekAPI question | 14 / 6 | **32 / 1** |
| stampli.com `ab551881` | bare keyword | 16 / 13 | **32 / 1** |

32 passages every time; the ranked pool is unchanged (64), only the cut is. The first slot is
still the best-ranked passage overall (Stampli, GeekAPI question: the automated-invoice-approval
blog post; bare: `/dynamic-approval-workflows/`). Not measured yet: whether the extra pages lift
GeekAPI's category count (the Avidxchange 2-of-20 refusal) — that is GeekAPI's extraction over
these pages and needs a readiness run.

## 9. Corpus state (2026-10-06)

- 104 crawl runs; no index job pending or running.
- Collapsed on current code (every point carries `textDigest`): ramp.com `4563f7ec` (2,150 pages,
  28,150 points, 7,339 repeats skipped), bill.com `e17ef3c0` (11,720 / 2,956), lightyear.cloud
  `84f4f34f` (1,727 / 593), and from the 2026-10-04 six: airbase `c60dc645`, fnshiftsolutions
  `d880fb46`, highnote `324af3f2`, dost `dbd75d19`, invoiced `67ac7054` (lightyear `44ba341c` is
  superseded). Runs indexed after `03b02c7` are collapsed too; not enumerated here.
- The old Ramp run `f8a3aa8c`, on which R1 and R2 were measured, no longer exists. Further Ramp
  measurement uses `4563f7ec`.
- Older runs keep their duplicate points until Jeff re-queues them.

## 10. Pitfalls recorded this week

- **A host is not a run.** tipalti.com is on a project's partner list and its competitor list,
  with one complete, indexed run for each. `/v1/index/hosts` resolved the host alone and answered
  the competitor run; GeekAPI probed it with `crawlType: partner`, got nothing, and excluded the
  partner ("its crawl finished, but a search of the index finds nothing from it"). Fixed
  2026-10-08 on both sides: this repo `0eb139c` — the route takes `crawlType` and resolves
  `(host, crawlType)`; an untyped host indexed under more than one type is refused with the
  types named. GeekBackend `2d7420b` — `HostsIndexedAsync(urls, crawlType, ct)` sends the type
  from all four callers and refuses a blank one before any request. **Proven 2026-10-08
  11:31:56 UTC:** Jeff reloaded the app, re-entered https://tipalti.com/ and saved; this service's
  log shows the typed hosts lookup, then `hybrid_halves runId=39bbce59…` (the partner run) for the
  probe, then the query answered; `5cbbb85b` never appears, and the UI reported Tipalti usable.
  Two traps: the project form asks the index only for URL strings it has no answer for in React
  state (`ProjectForm.tsx:218`), so re-entering an answered URL on an open form sends nothing — a
  page reload or "Re-check the index" re-asks; and a filtered `grep` over `docker compose logs`
  twice returned nothing while the lines were present — dump raw `--since/--until` windows before
  concluding a log is empty.
- **Every word sent to `/v1/query` is a search term.** A question written for meaning-only
  retrieval ("…the cost, delay and error rate of the manual or status-quo way…") becomes noise
  the moment the keyword half runs: on Melio those words put an accounts-receivable article in
  slot two, while the operator's own `nicheFraming.coreProblem` pulled the pricing page and five
  case studies. The 22 words were deleted (GeekBackend `bfd99c9`); the brief's framing still does
  not reach retrieval, and feeding it to the meaning half with the keyword on the keyword half is
  an open proposal Jeff wants to test before deciding.
- **Two keyword passes, by accident.** The in-process BM25 + RRF (`cfdb36e`, 09-07) was the only
  keyword signal while retrieval was meaning-only; once the hybrid's own keyword half ran
  (`7c47fa1`) it scored the keyword a second time, about 3:1 keyword over meaning with no single
  knob. Disabled 2026-10-08 at Jeff's instruction, kept commented in `_query_hybrid` because he
  intends to turn one keyword engine back on for comparison. It cannot replace the hybrid's half
  (it only ever ran on top of retrieval); re-enabling it means two passes again.
- **A claim read from code is not a fact about production.** Check Mongo/Qdrant/the VPS first.
- **Names hide behaviour.** `query.py` calls the hybrid query `dense_query` / `DenseRetriever`; the
  plan's audit missed that hybrid BM25 existed and decided to build a second keyword search.
- **A document's claim spreads into a plan.** The README called the scheduler deprecated; the plan
  took it as fact and R5 turned it off by default; nobody weighed what it does. Check the code.
- **`attempt` rises on every claim**, including a re-post of a completed run. That is why the old
  "skip delete on attempt > 1" path merged instead of replacing; it is gone, do not bring it back.
- **The job store reads fields back one by one** (`status_store._from_doc`); a new status field must
  be added there too. `test_the_job_store_reads_back_every_status_field` pins it.
- **A test that calls the library directly proves the library, not our code.** The hybrid test
  passed `query_str` to the store itself; `dense_query` never did, and production ran dense-only.
  Test through the function production calls.
- **A label is not evidence.** `retrieval="llamaindex-hybrid"` was stamped from the requested mode,
  not from what ran. The log line is what showed the truth.
- **zsh:** `echo ===` is command expansion and aborts the line.
