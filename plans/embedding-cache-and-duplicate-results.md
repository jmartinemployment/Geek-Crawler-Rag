# Cache identical embeddings and drop duplicate search results

Status: **Not started**
Measured: **2026-09-11** against live Qdrant (run `99c6b00b`, page `699f3333…`)
Prerequisite shipped: resume-from-Qdrant (`731b30d`), throttle 1,000,000 → 400,000

## Objective

Stop paying OpenAI twice for the same vector during indexing, and stop returning
the same text twice from `/v1/query`. Both symptoms share one cause. Neither fix
deletes data, changes chunk sizes, or re-enables retries.

## Cause

`parent_child_units` (`chunk.py:101`) windows children at 200 tokens
(`child_chunk_size_tokens`). A heading section **shorter than 200 tokens cannot be
sliced**, so the child text comes back byte-identical to its parent. The corpus is
partner and competitor marketing pages — mostly short headed blocks — so this is
the common case, not an edge case.

Measured on one real page (`https://clickup.com/blog/workflow-automation/`):

```
points stored:     139   (82 child + 57 parent)
distinct strings:   98
redundant calls:    41   → 29% of embedding calls for that page
```

## Symptom 1 — wasted indexing

Each identical pair is sent to OpenAI twice and returns the same vector twice.
29% of embedding time, spend, and rate-limit budget buys nothing. On the clickup
run that took 2h29m, that is roughly 43 minutes.

## Symptom 2 — duplicate search results

Confirmed live. Duplicate density rises with depth:

```
topK=8    8 returned,  5 distinct,  3 redundant   (37%)
topK=20  20 returned, 11 distinct,  9 redundant   (45%)
topK=40  40 returned, 21 distinct, 19 redundant   (48%)
```

Identical vectors score identically, so the duplicate always lands in the
immediately adjacent slot. `prefer_parent` defaults to `None` (`models.py:87`) and
`_should_collapse_parents` (`query.py:298`) requires it truthy, so a caller that
omits the flags receives ~37% less distinct context than it asked for.

`GccV2GeekCrawlerResearchResolver.cs:501` passes the safe combination, but
`HttpGeekCrawlerRagClient` takes both as `bool?` defaulting to null, so any new
caller lands on the broken path. Content Creator is not production yet — this is
the cheap moment.

## Change 1 — cache identical embeddings (`llama_engine.py`)

In `embed_and_upsert`, after the empty-text filter and the resume-from-Qdrant skip,
collapse to unique texts before calling `embed_texts`, then fan vectors back out:

```python
uniq = list(dict.fromkeys(texts))          # order-preserving
uniq_meta = [first metadata seen per unique text]
vecs = await self.embed_texts(uniq, metadata_list=uniq_meta)
by_text = dict(zip(uniq, vecs, strict=True))
embeddings = [by_text[t] for t in texts]
```

**Sanitization is safe — verified, not assumed.** `sanitize_embedding_texts` is a
loop over `sanitize_embedding_text` (identical per-item), and it is idempotent
across 94 real samples with 0 mismatches. `embed_and_upsert` already sanitizes each
node before building `texts`, so the second pass inside `embed_texts` is a no-op
and dedup keys are stable.

**Guard anyway.** `embed_texts` (`llama_engine.py:186`) may drop empty strings,
returning a list aligned to its *filtered* input. If `len(vecs) != len(uniq)`, fall
back to the un-cached path rather than mis-mapping vectors to nodes. Log the
saved-call count beside the existing `resume_skipped_existing_points` line.

Both point IDs are still written. Qdrant ends up byte-identical to today.

## Change 2 — always drop exact-duplicate result text (`query.py`)

`_select_ranked_candidates` (`query.py:324`) applies the `target_top_k` cap, so the
dedup must happen **there**, not in the assembly loop at line 190 — otherwise the
cap is spent on duplicates and callers get short results.

Pass `request` into `_select_ranked_candidates` and skip any candidate whose
`_return_text(payload, request)` was already emitted. This runs **unconditionally**,
independent of `collapse_parents`.

**Over-fetch is required, or dedup starves large topK.** Measured duplicate density
is ~48% at depth, and the pool is `min(len(fused), max(top_k, rerank_pool_size=40))`.
At `topK=8` a 40-pool yields ~21 distinct — fine. At `topK=40` the pool is also 40,
so dedup would return ~21 instead of 40. Starvation begins around `topK > 20`.
Widen the pool to roughly `max(top_k * 2, rerank_pool_size)` and raise
`hybrid_dense_limit` / `hybrid_lexical_limit` (both 30) to match, so there are
always enough distinct candidates to fill the requested slots. Cap to `top_k`
*after* dedup.

Do **not** default `_should_collapse_parents` on. Its key (`_parent_lineage_key`)
collapses all siblings sharing a parent, not just identical text, so defaulting it
would drop legitimately distinct children of large sections. Sibling collapse
stays opt-in; exact-text dedup is the surgical fix.

## Change 3 — pin the throttle

The box was hand-edited to `400000`; `deploy/hostinger-compose.yml` still carries
the old value. Commit `400000` so a deploy cannot drift back to the 1,000,000
ceiling, and align `README.md`.

## Verification

1. **Unit** — a batch containing the same string twice sends one entry to
   `embed_texts`; both nodes receive the vector; `async_add` still receives both.
   A mismatched-length return falls back instead of mis-mapping.
2. **Unit** — `_select_ranked_candidates` given duplicate return text emits one and
   fills the freed slot with the next distinct candidate, so `len(chunks) == top_k`
   when the widened pool allows.
3. **Starvation regression** — assert `topK=40` returns 40 distinct chunks, not ~21.
   This is the case that fails without over-fetch.
4. **Regression** — full suite (244 tests) green, especially
   `tests/test_embedding_circuit.py`; the quarantine path is untouched.
5. **Retrieval end-to-end** — repeat the query that exposed this
   (`need="Zapier no-code automation platform connects apps"`, run `99c6b00b…`)
   with flags omitted at topK 8, 20, 40. Expect distinct == returned at each.
6. **Indexing end-to-end** — re-index klaviyo `5e546c8f` (2,248 pages); confirm
   cached-call count is ~25-30% of batch size, `chunksUpserted` matches a
   non-cached run, and wall time drops proportionally.

## Out of scope

- Deleting duplicate parent points (~2.2 GB). No consumer filters `chunkRole`, so
  it is safe in principle, but it changes stored data and is not needed for speed.
- Changing `child_chunk_size_tokens` from 200. Would reduce collapse at the source
  but alters retrieval behavior and forces a full re-embed.
- Retries. `openai_embedding_max_retries` stays 0 by design.

## Current box state

Scheduler disabled. 76 complete / 16 skipped / 1 pending — that pending row
appeared after the 2026-09-10 lockdown and should be identified before
re-enabling. Throttle 400,000. clickup `99c6b00b` complete at 355,331 chunks.
Quarantine dumps still 4; no failures since the throttle change.
