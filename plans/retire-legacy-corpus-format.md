# Retire the legacy corpus format from the RAG Library — LANDED

Completed 2026-09-18. Kept as a record of what changed and why, plus the two
operational items that need the Hostinger box.

## What the corpus is

The crawler emits clean semantic `contentHtml` plus typed `blocks`
(`Geek-Crawler-v2/plans/corpus-rebuild.md`). That is the only representation —
for storage, for verification, and on the wire between services. No hop converts
it to anything else, and nothing here may add a conversion step.

| Concern | The method |
|---|---|
| Corpus body | typed **`blocks`** — `heading`+`level`, `paragraph`, `listItem`, `quote`, `code`, `row`+`cells`, `term`, `definition`; each with `text`/`cells`, `html`, `anchors` |
| Display / audit | **`contentHtml`** |
| Page as a string | **one** projection — `block_text.derive_plaintext_from_blocks` |
| Quote verification | `citation_verify.quote_in_text(quote, plain_text, blocks)` against that same string |
| Page read API | `GET /v1/pages…` → `PageTextResponse.text` |
| Run readiness | **`ContentReadyAt`** |

Chunk text and verification text come from the same projection deliberately: a
quote is taken from a retrieved chunk and then matched against the page, so two
implementations of "join the blocks" make correct citations fail.

## Why it was urgent

The crawler changed corpus format and this service did not. Every page then
classified as having no usable body, and `_delete_unusable` removed it along with
its Qdrant points — eight runs, 5,274 pages, 0 chunks upserted, corpus destroyed
on 2026-09-18. A second representation of one contract is not a convenience; it
is the seam the two halves drift apart along.

## What changed

- **Text derivation** — `block_text.py` is the single projection, shared by the
  chunker and by citation verification so they cannot diverge. `extract.py`
  projects a page through it; `citation_verify.quote_in_text` matches against it,
  with a whole-cell fallthrough for table values under the 12-character floor
  (`$15/month` is 9, `99.9%` is 5).
- **Public contract** — `GET /v1/pages…` returns `PageTextResponse.text`.
- **Schema reads** — `CrawlPage` carries `content_html` and `blocks`; the page
  projections hedge both Mongo casings.
- **Readiness** — the run filter and its covering index are `ContentReadyAt` and
  `ix_crawl_runs_content_ready` (`78c143b`).
- **Selective pruning retired** — `unusable.py` reports `no_content` and
  `indexer._skip_unusable` counts without deleting. The crawler owns the reject
  taxonomy; a consumer re-adjudicating it is what cost the corpus.
- **Chunking** — `parent_child_units` takes the page's blocks, cuts sections at
  heading blocks, and carries each section's own anchors onto its chunks
  (`b9fadcc`). Before that, the splitter keyed on a plaintext heading convention
  the projection never emits, so every page was one nameless section and
  `sectionTitle` was empty on every chunk in the index.
- **Provenance** — `parserId` is `crawler-blocks`.
- **Ops scripts** — the two that existed only to serve the retired format were
  deleted, along with three orphaned fixtures no test referenced.

### Deliberate deviation — the index and hint swap

This work originally prescribed four deploys for the readiness swap, because
`mongo.py` passes the index name as a `hint=` and Mongo **errors on a hint naming
an index that does not exist**, so a rename split across deploys breaks every scan
in between. It was collapsed into a **single edit** instead: filter, index name
and hint moved together. Correct here only because no container instance was
serving scheduler scans, so no runtime query could hit the missing-index window.
On a live scheduler the four-deploy sequence would still be the right shape.

The index scheduler is deprecated in any case — indexing is triggered by
`POST /v1/index`, and `INDEX_SCHEDULER_ENABLED` is `false`.

## Outstanding — needs the Hostinger box

- [ ] **Confirm the Mongo key casing** for `ContentReadyAt`. Run
      `scripts/verify_ingest_fields.py`, which reports presence *and* casing per
      field. The page projections hedge both casings; the run filter and the index
      cannot. The expected spelling is inferred from the writer's serializer
      config, never observed against the data.
- [ ] **Drop the superseded readiness index.** `db.crawl_runs.getIndexes()` lists
      them; the live one is `ix_crawl_runs_content_ready` and the other readiness
      index is the one to drop. Safe once the image carrying `78c143b` is running —
      not before, or an older instance still hinting the old name breaks.

## Verification

`uv run pytest -q` → 162 passed. Locked by
`test_section_text_is_the_page_projection_restricted_to_those_blocks`: every
section's text must be a substring of the whole-page projection, which is what
keeps a quote pulled from a chunk verifiable against its page.
