# Handoff — Geek-Crawler-Rag, 2026-10-06

For whoever picks this repository up next. Checked on 2026-10-06 against the repository, the VPS
and live Mongo/Qdrant; where something was not checked, it says so. Authority for rules is
`/Users/jeffmartin/development/.claude/CLAUDE.md`; the plan is
`content-creator-v2/plans/fix-geek-crawler-rag.md` (read its **Status** section first, then the
stages). This file says where things stand and what to do next.

## 1. What this repository is

The Library: retrieval and quote verification over the crawl corpus. **It never generates.** It
indexes a crawl run (Mongo `crawl_pages` blocks → chunks → Qdrant `geek_crawler_chunks`), answers
`POST /v1/query` for GeekAPI, serves page text, and answers `POST /v1/verify`. Production is the
Hostinger VPS `root@2.24.101.90` (key `~/.ssh/hostinger_rag_ed25519`); the live compose is
`/docker/geek-crawler-rag/docker-compose.yml`, not `deploy/`.

**Pushing to `main` deploys** (build → GHCR → VPS) and recreates the API container, which kills any
index job in flight. There are no branches.

## 2. Rules that shape the work (Jeff's, this week)

- **Indexing a run replaces it; it never merges.** Every attempt deletes the run's points, then
  writes the run whole. There is no resume. "Never have two URLs the same." (2026-10-04)
- **Hybrid always.** Retrieval is dense + sparse BM25 in one Qdrant query; there is no dense-only mode
  and no setting to make one.
- **Re-index only per R1:** one run at a time, only runs a saved project declares, each checked at
  `GET /v1/index/{run_id}` before the next is posted. **Jeff queues sites himself.** Never the corpus,
  never several at once, never the `requeue-stranded.sh` cron. (A 62-run re-index was queued on
  2026-10-04 and killed; do not repeat it.)
- **No push while any index job is pending or running.** Check `rag_index_jobs` first (§6).
- **Do not touch the production scheduler.** It is on (`INDEX_SCHEDULER_ENABLED: "true"` in the VPS
  compose); the repo default is off. Turning it off is Jeff's change.
- **Every measurement asks what GeekAPI asks:** `GccGroundingResolver.BuildNeed` in GeekBackend —
  the topic's keyword (whole string if no `descriptor: keyword` colon), trimmed, capped at 150, then
  `" -- the cost, delay and error rate of the manual or status-quo way, the capability that removes
  it, and measured outcomes"`, `topK` 32, `crawlType` `partner`. Never the bare keyword.
- **Do exactly what was asked.** When a plan or a measurement suggests more, report and ask. Do not
  override a recorded plan decision on your own judgement.
- **Fail closed, no fallbacks, no Markdown** (CLAUDE.md §1a, §2). Commit to `main`.
- **One session per repository** (content-creator-v2 `AGENTS.md`). A contract change is committed on
  both sides together (the plan accepts that); otherwise read other repos, do not change them.

## 3. What is deployed right now

| Repository | `main` | Deployed | Checked |
|---|---|---|---|
| Geek-Crawler-Rag | `5e622b6` | VPS, API container created 2026-10-04 14:58 UTC | 2026-10-06 |
| GeekBackend | `5671288` (carries `1c553fc`, `a8e284d` from this work) | Railway, not checked from here | — |
| content-creator-v2 | origin `67456d3`; local has 2 unpushed commits from another session | — | — |

Commits from this work, in order: `1a25095` (R1/R4/R5 first cut), `03b02c7` (always delete, resume
removed, status round-trip fix), `529a5df` (hybrid always), `9a0c901` (R4 fixed, F-R10), `5e622b6`
(R2, scroll deleted). GeekBackend `1c553fc` (webhook field `chunksSkippedRepeat`), `a8e284d` (C#
half of the block-projection fixture). content-creator-v2 `9f41697` (the six re-indexed runs).

## 4. The plan's stages, as they actually stand

| Stage | State |
|---|---|
| **R1** one point per distinct text per run | **Done.** `textDigest` on every point; `indexer._admit_page_nodes`; `chunksSkippedRepeat` in status and webhook. Done-when met on the old Ramp run (33,728 → 26,683 points). |
| **R2** keyword scroll deleted (D15) | **Done** (`5e622b6`). Distinct pages before/after on the old Ramp run, with `BuildNeed` questions: 20/19/28/25/23 → identical. |
| **R3** collapse before the cut; measure near-copies | **Not started. Wave 2 — held.** Earlier near-copy numbers used the bare keyword and do not count. |
| **R4** verify route | **Code done** (`9a0c901`): one digest `sha256(contentHtml)` (`llama_nodes.page_source_digest`), `verify_citations` deleted, `found` documented as the only verdict, F-R10 fixture. **Done-when not met:** GeekAPI's A1 must call `/v1/verify` with no comparison of its own — not built (GeekBackend's). |
| **R5** readiness fail-closed | **Done.** `POST /v1/index` and the scheduler's entrance refuse a run without `ContentReadyAt` (409, no job row). |
| **R6** tests | Partial. Done: cross-page collapse, verify route, readiness, F-R10. Open: flooded-pool test. The "ranked lexical list" test is moot now the list is deleted (say so to the plan writer). |
| **R7** README drift | Wave 3. One line changed early; memory limit, upsert delay and retry wording still disagree with code. |

**The wave gate:** no wave's end-to-end proof has been run. The proof is one Generate on the
Accounts Payable project, all seven live types, read by Jeff, every quote found on the page it cites.
**Wave 2 (R3, the rest of R6) waits for it.**

**The plan's Status section is stale on R4:** it still lists the three R4 defects (two digests, no
F-R10 proof, `verify_citations`) as open. All three are fixed in `9a0c901`. Tell the plan writer;
the status section is theirs to update.

## 5. Corpus state (2026-10-06)

- 104 crawl runs. No index job pending or running.
- Runs indexed on the current code (collapsed, every point has `textDigest`): ramp.com `4563f7ec`
  (re-crawled 2026-10-05; 2,150 pages, 28,150 points, 7,339 repeats skipped), lightyear.cloud
  `84f4f34f` (1,727 / 593), bill.com `e17ef3c0` (11,720 / 2,956), and the five surviving runs of the
  six re-indexed on 2026-10-04 (airbase `c60dc645`, fnshiftsolutions `d880fb46`, highnote `324af3f2`,
  dost `dbd75d19`, invoiced `67ac7054`; lightyear `44ba341c` from that list is superseded).
  Runs crawled after `5e622b6` and indexed since are also collapsed; not enumerated here.
- The old Ramp run `f8a3aa8c`, on which R1 and R2 were measured, **no longer exists**. Any further
  Ramp measurement uses `4563f7ec`.
- Older runs keep their duplicate points until Jeff re-queues them.

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

- One run's job: `GET /v1/index/{runId}` (header `X-API-Key: $API_KEY` inside the container; there is
  no `curl` in the image — use `python -c` with `httpx`).
- A run's points: Qdrant `count` with `runId` = run, and again with `textDigest` present; equal and
  equal to `chunksUpserted` means it was rewritten whole on current code.
- `uv run pytest` (≈494 tests). Cross-repo CI: `.github/workflows/cross-repo-citation-contract.yml`.
- Do not print a range of the VPS compose: `MONGO_CRAWLER_URL` carries the Mongo password. Grep keys.
- `docker compose exec -T` reads stdin: two in one ssh line and the second gets nothing. One per call.
- A regex over `crawl_pages.Origin` scans the whole collection (minutes). Prefer `RunId`.

## 7. Where the code is

| Concern | File |
|---|---|
| Index a run: delete, page loop, repeat collapse, readiness gate | `src/geek_crawler_rag/indexer.py` (`_index_run`, `_admit_page_nodes`, `readiness_refusal`) |
| Chunks → nodes, `textDigest`, `sourceDigest` | `src/geek_crawler_rag/llama_nodes.py` (`text_digest`, `page_source_digest`) |
| Embedding (dedupe within a flush, cross-run cache), hybrid query | `src/geek_crawler_rag/llama_engine.py` |
| Query: hybrid candidates → BM25 re-rank → RRF → (Cohere off) → select | `src/geek_crawler_rag/query.py` |
| Verify route, page reads, shared refusal helper | `src/geek_crawler_rag/app.py` (`verify_quotes`, `_citable_page_text`) |
| Quote check and the F-R10 normalisation | `src/geek_crawler_rag/citation_verify.py` (`verify_quote`) |
| Block → text, the one projection | `src/geek_crawler_rag/block_text.py` |
| Job rows (Mongo `rag_index_jobs`) | `src/geek_crawler_rag/status_store.py` |
| Wire contracts, copied byte-identically into GeekBackend | `contracts/rag-index-status/webhook.v1.json`, `contracts/block-projection/v1.json` |

## 8. Known and open

- **Cross-repo CI is red on four GeekBackend tests** (`RagClientContractTests` retry-model, 404s).
  They failed before this work and are GeekBackend's.
- **`diagnostics.py:629`** checks a caller-declared `sourceDigest` over a supplied document's text — a
  second meaning of the name, for non-corpus documents. Left as is; flagged to the plan writer.
- **`lexicalScore`** on `ChunkHit` is always null; kept because it is on the wire to GeekAPI.
- **Curly vs straight quotes are not folded**, by instruction: both projections keep the page's
  characters. A model that straightens an apostrophe gets `found: false`.
- **`CLAUDE.md` §1a** still names `mongo.find_smallest_content_ready_run`; the code is
  `find_oldest_content_ready_run`.
- **`plans/partner-evidence-reaches-the-writer.md`** is untracked here and is not this session's;
  leave it to its author.
- **Cohere rerank is off in production** (no key), so ranking is RRF over hybrid + BM25 (F-R7).
- **Ramp's passages are mostly `/blog`** (21–27 of 32). Product pages were never fetched because of the
  crawler's sitemap filter (Geek-Crawler-v2 Stage 2); a retrieval change cannot fix that.

## 9. Pitfalls recorded this week

- **A claim read from code is not a fact about production.** Check Mongo/Qdrant/the VPS before saying
  what happens there.
- **Names hide behaviour.** `query.py` calls the hybrid query `dense_query` / `DenseRetriever`; the
  plan's audit missed that hybrid BM25 existed and decided to build a second keyword search.
- **`attempt` rises on every claim**, including a re-post of a completed run. That is why the old
  "skip delete on attempt > 1" path merged instead of replacing; it is gone, do not bring it back.
- **The job store reads fields back one by one** (`status_store._from_doc`); a new status field must
  be added there too. `test_the_job_store_reads_back_every_status_field` pins it.
- **zsh:** `echo ===` is command expansion and aborts the line.
