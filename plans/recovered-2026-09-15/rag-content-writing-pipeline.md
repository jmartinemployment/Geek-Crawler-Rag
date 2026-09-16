# RAG pipeline upgrades (Geek-Crawler-Rag scope)

Status: **Phase 0 + B + E + D implemented**; **Phase M backfill running against Hostinger Mongo** (Readability → markdownify); **Phase U unified content pipeline implementation in progress, cross-repository verification pending**.
Sibling plans: Geek-Crawler-v2 (markdown ingest), GeekBackend (`/api/rag/generate`), content-creator-v2 (consume generate).

## Governing principle

> Use every available signal and the strongest appropriate technology to create the highest-quality content possible, while preserving editorial control and verifiable evidence.

The RAG service is one component of a single content-creation product. Crawler, RAG, GeekBackend, the canonical brief, PLAN/WRITE/VALIDATE, Canvas, model policy, citations, exports, and publishing must work as one quality system.

This is a new unified implementation informed by the history and proven capabilities of the existing projects. Existing contracts remain compatible only where that does not preserve an accidental product boundary or silently reduce quality.

## This repo owns

- Indexing Mongo `crawl_pages` → Qdrant
- Chunking, embeddings, hybrid retrieval, rerank
- **LlamaIndex** as the RAG framework (ingestion, indices, query pipelines) **inside** this service
- `POST /v1/index`, `GET /v1/index/{runId}`, `POST /v1/query`
- Reading Mongo `entities` for domain → `entityName` / `sourceType` (collection owned by Backend; this service consumes it)

## Today (after Phase D)

- Prefer Mongo `Markdown`/`Title` when present; else BeautifulSoup HTML → text; EN-only `langdetect`
- Parent (~1000 tok) + child (~200 tok) chunks; both embedded as separate LlamaIndex `TextNode`s
- **LlamaIndex** `OpenAIEmbedding` + `QdrantVectorStore` drive ingest/dense query inside FastAPI
- OpenAI `text-embedding-3-small` → Qdrant `geek_crawler_chunks`
- Rich payload: run/host/url + `parentText`/`childText`/`chunkRole` + entity/category/quality fields
- Hybrid query: LlamaIndex dense + BM25/text RRF; optional Cohere rerank (`COHERE_API_KEY`)
- Query filters: `preferParent`/`preferChild`/`chunkRole`/`sourceTypes`/`entityNames`/`categories`/`minQuality`
- **Graph mode:** `retrievalMode: graph` → parent-biased hybrid + `themes[]` (entity/category/co-occurrence)
- **Ad templates:** `/v1/templates/index` + `/v1/templates/query` on collection `geek_ad_templates`
- FastAPI remains the HTTP/deploy shell (`v1/*` stable for GeekAPI)

## Locked decisions

- **Keep Qdrant** (not Pinecone/Chroma/Mongo vectors)
- **FastAPI + LlamaIndex are complementary** — FastAPI stays the HTTP/API shell; LlamaIndex owns ingestion, indices, and query pipelines (not “pick one”)
- **Adopt LlamaIndex in this repo** (Phase E) — current custom index/query is an interim shape, not the end state
- Hybrid retrieval (vector + BM25 + RRF) stays required; implement/keep it via LlamaIndex (or LlamaIndex + Qdrant) under the existing `v1/*` contracts
- Crawlee stays the crawler (Firecrawl not adopted)
- OpenAI writer for **citeable** drafts lives in this service (`POST /v1/generate`); GeekAPI orchestrates and proxies.
- Canonical PLAN/WRITE jobs **never silently fall back** to GeekAPI one-shot or a weaker model. Missing RAG, evidence, or model prerequisites produce an actionable job state.
- The complete versioned Creator brief—not `topic + writingIntent` alone—is the generation quality contract.
- Model selection is explicit and stage-aware: o1-pro and o3 are first-class policy choices, enforced independently by Backend and this service.
- Operator-controlled downgrade is permitted only through an audited UI action; citation and validation standards never downgrade.

## Phase 0 (shared gate — data first)

Before coding Phase B here:

1. Inventory Qdrant point count + payload field set + recent `rag_index_jobs`
2. Sample indexed `runId`s and a `v1/query` response
3. Confirm whether any pages already have `markdown` in Mongo (after Crawler-v2 Phase A)
4. Entity seed list available for domain match tests

Artifact: `plans/rag-baseline-inventory.md` (this repo or Crawler-v2).

## Phase B — This repo’s main work

### B1. Prefer markdown when present — done

- If page has `markdown` / `title` from ingest → chunk that
- Else fall back to current HTML extract

### B2. Parent-child chunking + dual embed — done

- **Child** ~200 tokens (feature/stat pinpoint)
- **Parent** = heading section or ~800–1200 tokens
- Embed both; store both texts in payload
- Long-form path: match on child → return **parent** text to caller
- Short-form path: return **child** text

### B3. Richer payload + filters — done

```text
runId, crawlType, host, url, pageId, title,
parentText, childText, sectionTitle, chunkIndex, language,
sourceType, entityName, entityId?,
category, contentIntent, tags[], qualityScore,
isEvergreen, lastCrawled
```

- Resolve entity via Mongo `entities.domains` match; else `crawlType` + host
- Category/intent heuristics from URL + text (backfill old HTML pages on reindex)

### B4. Hybrid retrieval (vector + BM25 + RRF) — done

- Dense Qdrant search + keyword/BM25 (exact product names, codes, stats)
- Reciprocal Rank Fusion merge
- Extends `POST /v1/query`

### B5. Rerank — done

- Initial pool k=20–30 → Cohere v3 (or configured reranker) → top 5
- Config via env (soft-disable if key missing)

### B6. Query API contract extensions — done

Request additions (camelCase):

- `preferParent` | `preferChild` (or `chunkRole`)
- `sourceTypes[]`, `entityNames[]`, `categories[]`, `minQuality`
- Keep existing `need`, `runId`, `crawlType`, `host`, `topK`

Response: include `url`, chosen text, `entityName`, `category`, scores.

## Phase E — LlamaIndex under FastAPI (this repo — done)

LlamaIndex and FastAPI are different layers: FastAPI = routes/deploy; LlamaIndex = ingest/indices/query pipelines.

1. [x] Add LlamaIndex dependency; keep existing `v1/index` / `v1/query` HTTP contracts stable for GeekAPI
2. [x] Move Mongo → document/node pipeline (markdown-preferring, parent/child) into LlamaIndex nodes (`llama_nodes.py`)
3. [x] Wire Qdrant as the vector store behind LlamaIndex (`llama_engine.py` + `QdrantVectorStore`)
4. [x] Route hybrid (dense via LlamaIndex + BM25/text RRF) + optional Cohere rerank under FastAPI query service
5. [x] Thin bespoke indexer/query to orchestrate LlamaIndex (filters, preferParent/Child, entity metadata preserved)

**Not in Phase E:** replacing FastAPI, changing GeekAPI writer, or Firecrawl.

## Phase M — One-time markdown backfill (this repo)

Draft-research cleaning for **existing** Mongo HTML (no re-crawl). New crawls already clean in Geek-Crawler-v2 via Readability → Turndown.

1. Script: `scripts/backfill_markdown.py` (deps: `readability-lxml`, `markdownify`)
2. For pages with `Html` and empty `Markdown`: extract → write `Title` / `Markdown` / `Excerpt` / `MarkdownBackfilledAt`
3. Locale filter ported from Crawler-v2 `locale-path.ts` (KEEP `/us/`, DROP foreign regions/non-English)
4. Dry-run first, then write; **delete** robots-denied / failures / locale / extract-empty (do not leave skip-marked junk)
5. After write: operators reindex affected runs via `POST /v1/index`

Prerequisite: GeekBackend page docs accept those fields on ingest (and script may write Mongo directly on Hostinger).

## Phase C — Unusable corpus cleanup (this repo)

Sweeper when junk leaks past Crawler-v2. See [`cleanup-unusable-pages.md`](./cleanup-unusable-pages.md).

1. Shared classify: `src/geek_crawler_rag/unusable.py`
2. Bulk: `scripts/cleanup_unusable_pages.py --write`
3. Index: delete locale / failure / empty / non-English pages during `POST /v1/index` + report delete counts
4. Cascade `crawl_links`; optional Qdrant `pageId` delete

## Phase D — GraphRAG + few-shot ad template index (this repo — done)

### D1. GraphRAG (slides / strategy themes) — done

- Theme overlay on hybrid hits: entity nodes, category themes, co-occurrence edges (`graph_retrieve.py`)
- `POST /v1/query` with `retrievalMode: "graph"` (defaults to parent sections; returns `themes[]` + chunks)
- Vector hybrid remains default for technical deep dives

### D2. Few-shot ad template index — done

- Separate Qdrant collection `geek_ad_templates` (configurable)
- `POST /v1/templates/index` — upsert templates (corpus owned by content-creator-v2)
- `POST /v1/templates/query` — retrieve exemplars by need + optional channel/framework/entityTags
- Metadata: channel, framework, tone, entityTags

## Phase U — Unified content-creation contract

Phase U connects the completed RAG infrastructure to the canonical `/creates/new` → persisted job → Canvas product. It replaces the standalone `/rag` workbench as a user-facing creation path without removing any RAG capability.

### U1. Canonical brief contract

`POST /v1/generate` accepts a structured `canonicalBrief` object. It remains backward-compatible with historical standalone requests, but canonical jobs always provide the full object.

Required contract:

```json
{
  "canonicalBrief": {
    "version": "gcc-v2-generation-brief.v1",
    "title": "...",
    "targetKeyword": "...",
    "contentType": "pillar",
    "primaryIntent": "...",
    "audience": {},
    "buyingStage": "...",
    "toneOfVoice": ["..."],
    "brandKit": {},
    "paaQuestions": ["..."],
    "requiredTopics": ["..."],
    "operatorInstructions": ["..."],
    "exclusions": ["..."],
    "hierarchy": {},
    "internalLinks": [],
    "outputRequirements": {},
    "channelRequirements": {},
    "ctaRequirements": {},
    "conversionObjective": "...",
    "publishingDestination": "..."
  }
}
```

The brief drives retrieval needs, evidence allocation, outline strategy, section drafting, repair, validation, repurposing, and final editorial synthesis. Unknown additive brief fields are preserved for forward compatibility.

Before PLAN, GeekBackend assembles an inspectable research/evidence manifest:

- first-party, partner, competitor, and external sources
- source authority/freshness and crawl/index readiness
- exact quote-level evidence and candidate claims
- evidence gaps and conflicting claims
- internal-link and product-proof opportunities

Missing required evidence is a visible quality gate, not permission to generate unsupported prose.

### U2. Exact cross-repository generate contract

Canonical request fields:

```text
writingIntent, topic, partnerRunId, competitorRunId,
targetEntities, adTemplates, graphEnabled,
generationStage, outline, sectionKey, sectionHeading,
sectionBrief, completedSectionSummaries,
canonicalBrief, modelPolicyPreset, modelPolicyVersion,
stageModelOverrides
```

Canonical response fields:

```text
intent, content, variations, battlecard, themes, outline,
sources, citations, warnings, evidenceWarnings,
modelUsed, retrieval, provenance
```

Additional structured requirements:

- `outline[].evidenceIds` allocates evidence during PLAN.
- `citations[]` contains `pageId`, URL, title, section title, exact quote, and crawl type.
- `provenance` contains generation stage, effective model, model-policy version, prompt version, retrieval strategy, and evidence IDs.
- GeekBackend forwards these fields without renaming or flattening away information.
- Contract versions and field names are identical in Python, C#, TypeScript, deterministic fixtures, and staging smoke tests.

Canonical model-policy version: `content-model-policy.v1`.

### U3. Explicit o1-pro/o3 model policy

Initial best-quality policy:

| Stage | Default model | Purpose |
|-------|---------------|---------|
| Research planning | o3 | Form retrieval needs, compare sources, find evidence gaps |
| Outline | o1-pro | Interpret the complete brief and design high-value long-form strategy |
| Section | o3 | Evidence allocation and citation-grounded drafting |
| Repair | o3 | Correct unsupported, repetitive, or weak sections |
| Validation | o3 | Structured evidence, contradiction, brand, SEO, and GEO review |
| Final synthesis | o1-pro | Whole-document editorial coherence without altering verified evidence |

Supported presets:

- `best-quality`: stage-aware o1-pro + o3 policy
- `o3-only`: explicit operator downgrade to o3 for reasoning stages
- `custom`: approved per-stage o1-pro/o3 overrides

Rules:

- No silent model substitution.
- This service rejects unknown policy versions, presets, stages, or models with actionable errors.
- Reasoning models use `max_completion_tokens`, omit unsupported temperature settings, and run in cancellable jobs with appropriate timeouts.
- Responses always identify `modelUsed`; Backend rejects a result whose effective model differs from the requested policy.
- Model/prompt/retrieval versions, evidence IDs, latency, tokens, warnings, and retry lineage are persisted.
- Cost and latency are observable, but do not silently control quality routing.

### U4. Operator-controlled model downgrade

Content Creator provides:

- Create-level **Model policy** selector: Best quality, o3 only, or approved custom per-stage models.
- Canvas **Change model / Retry with another model** action for a failed or delayed stage.
- Explicit quality/capability/cost/latency tradeoff disclosure and confirmation.

GeekBackend provides:

- `modelPolicyVersion` and `approvedStageModels` through RAG status.
- Authenticated and owner-authorized `POST /api/geek-content-creator-v2/jobs/{jobId}/retry-model`.
- Durable model override, reason, operator ID, timestamp, replaced attempt ID, and retry lineage.
- Affected-stage-only retry; approved work is not discarded without confirmation.

Model downgrade never disables evidence requirements, citation verification, validation gates, or editorial approval.

### U5. All 17 canonical content types

Creator content types remain the product taxonomy. RAG writing intents become internal strategies:

| Canonical content types | RAG strategy |
|-------------------------|--------------|
| pillar, blog, guide, tech-article, case-study, whitepaper, listicle | Long-form outline + section generation |
| comparison, alternatives | Partner/competitor battlecard retrieval + section generation |
| ads, social, email | Short-form evidence + approved ad-template exemplars |
| PDF (`linkedin-document` legacy ID) | Slide strategy + GraphRAG themes |
| tool, service, local | Long-form evidence with existing specialized product/page requirements |
| image-prompt | Evidence- and brand-grounded visual brief with specialized validation |

No canonical content type silently returns to a legacy writer. Specialized output formatting may remain in GeekBackend, but its claims and strategic inputs come from the canonical brief and citeable RAG evidence.

### U6. PLAN, WRITE, VALIDATE, and Canvas

- PLAN calls `generationStage=outline`; outline approval remains an editorial gate.
- WRITE calls `generationStage=section` with the full outline and completed-section summaries.
- REPAIR and Canvas rewrite/expand/re-tone re-run only the affected section through citeable RAG.
- VALIDATE gates citation integrity, unsupported claims, source conflicts, originality/overlap, brief alignment, brand voice, SEO, GEO, readability, usefulness, CTA quality, and content-type requirements.
- Section citations and provenance persist in stage/job JSON and are returned by Canvas APIs.
- Canvas shows evidence, exact quote citations, model/prompt/retrieval provenance, warnings, and retry lineage.
- Existing BrandKit, outline editing, section controls, validation, remix, carousel, CMS publish, and export capabilities remain.

### U7. One product surface

- `/creates/new` → persisted jobs → Canvas is the only content-creation experience.
- The complete existing brief is retained and expanded.
- Entities, templates, guided outlines, battlecards, variations, slides, strategy themes, and citations move into Create/Canvas based on canonical content type.
- Standalone `/rag` navigation is removed.
- `/rag` preserves authentication, migrates useful topic/intent/content-type query parameters, and redirects to `/creates/new`.
- `/api/rag/*` may remain as an internal BFF namespace; it is not a separate product workflow.

### U8. Contract and quality verification

Required deterministic checks:

1. Python contract tests for canonical brief propagation, stage-aware model enforcement, reasoning-model request constraints, and citation verification.
2. GeekBackend tests proving all content types map correctly, PLAN/WRITE use citeable RAG, model substitution is rejected, citations/provenance persist, and explicit retry is authorized/audited.
3. Creator Playwright tests covering all 17 type options, model downgrade confirmation, create → outline approval → WRITE → Canvas citations/provenance, legacy redirect, and RAG-unavailable quality gating.
4. A true cross-repository contract fixture using the same checked-in request/response examples in Python, C#, and TypeScript so mocks cannot drift.
5. Optional staging smoke covering deployed OAuth, crawl/index readiness, o1-pro/o3 routing, exact quote verification against Mongo Markdown, persistence, and Canvas display.

Quality evaluation corpus:

- representative briefs across all 17 content types
- strong, weak, missing, and conflicting evidence
- varied brand voices, intents, buying stages, and partner/competitor scenarios
- o1-pro versus o3 stage bakeoffs

Promotion gates:

- factual/citation accuracy
- evidence coverage and source quality
- brief/audience/intent alignment
- strategic depth and usefulness
- originality and non-repetition
- brand and editorial coherence
- SEO/GEO quality
- human editor preference and editing effort

No model, prompt, or retrieval-policy change is promoted solely because it is newer, faster, or cheaper.

## Ownership boundaries

- Geek-Crawler-Rag owns retrieval, full-page reads, generation prompts, model-policy enforcement, citation verification, and generate provenance.
- GeekBackend owns canonical brief assembly, content-type strategy mapping, job orchestration, policy authorization, durable overrides/retries, stage/job persistence, validation orchestration, and the thin RAG proxy.
- content-creator-v2 owns the brief/model controls, template product corpus and picker, Canvas editorial workflow, citations/provenance display, and explicit downgrade confirmation.
- Geek-Crawler-v2 owns crawl acquisition and clean Markdown ingest.

## Out of scope here

- Replacing Crawlee
- Running LlamaIndex from Geek-Crawler-v2 or content-creator-v2
- Replacing Qdrant or the required hybrid retrieval path
- Maintaining standalone `/rag` as a peer creation product
- Silently falling back to GeekAPI one-shot or a weaker model for canonical jobs
- Reducing the canonical brief to topic/intent

## Success criteria

- [x] Reindex run produces parent+child points with metadata
- [x] `v1/query` supports hybrid + rerank + preferParent/Child + entity filters
- [x] Backward compatible: old clients omitting new fields still get results
- [x] Phase 0 artifact: [`rag-baseline-inventory.md`](./rag-baseline-inventory.md)
- [x] Phase E: LlamaIndex drives ingest + query pipelines inside FastAPI; hybrid + filters parity; GeekAPI clients unchanged
- [x] Phase D1: GraphRAG retrieval usable for slide/strategy intents (`retrievalMode: graph` → `themes`)
- [x] Phase D2: Ad template index query returns few-shot exemplars for short-form (`/v1/templates/*`)
- [ ] Phase M: one-time Readability markdown backfill over existing HTML; dry-run then write
- [x] Phase C: unusable pages deleted via cleanup script + index/backfill sweepers ([`cleanup-unusable-pages.md`](./cleanup-unusable-pages.md))
- [x] Citeable: `pageId` on hits, `GET /v1/pages/*`, `POST /v1/generate` workflow ([`citeable-rag-output.md`](./citeable-rag-output.md))
- [ ] Phase U1-U2: canonical brief and exact cross-repository contract verified end to end
- [ ] Phase U3-U4: o1-pro/o3 policy and audited downgrade/retry verified
- [ ] Phase U5-U7: all 17 content types use the unified Create/Canvas product with no legacy writer
- [ ] Phase U8: shared contract fixtures, canonical E2E, staging smoke, and quality evaluation gates pass

**Ops note:** Deploy new image, then markdown backfill + `POST /v1/index` for runs that should get Markdown-backed chunks. See [`citeable-rag-output.md`](./citeable-rag-output.md) and `scripts/markdown_coverage_report.py`.
