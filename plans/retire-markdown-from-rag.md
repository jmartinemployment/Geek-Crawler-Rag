# Blunt Greenfield: retire Markdown from the RAG Library

## Context

The crawler completed its Markdown→HTML migration (`plans/corpus-rebuild.md`,
`extract-content.ts:17`). It emits `contentHtml` + typed `blocks`. The RAG
Library never followed, and still requires Markdown — by explicit design, not by
accident:

- `extract.py:55-62` — *"Require crawler Markdown. Do not synthesize from HTML."*
  plus `del html  # never fall back to HTML extract`
- `unusable.py:107` — *"Pages without crawler Markdown are deleted."*

`classify_unusable_page(markdown=page.markdown)` therefore returns `no_markdown`
for every page, `indexer.py:467` maps it to `extract_empty`, and
`_delete_unusable` deletes the page **and its Qdrant points**.

Measured today across 8 runs: **5,274 pages seen, 5,274 deleted, 0 chunks
upserted**; `page-urls` returns `[]` for every run. The corpus survives only in
`DATA_DIR/extract-cache` (208 MB).

`no_markdown` is a legacy artifact of the stringification era, and so is the
reason-ranked pruning order around it. Since the corpus is being wiped anyway,
both are retired rather than repaired.

## Strategy

Public contracts move to **naked plain text derived from `blocks`**, not to a
new stored field. One source of truth: the same array feeds chunking and
verification, so the two cannot drift.

Markdown was doing double duty as storage and as a plaintext-ish match target,
and its syntax tokens (`#`, `*`, `**`) and significant line breaks are exactly
what break substring quote matching. Plain text has none.

## Already done (no work required)

Verified in today's commits — `de75b62`:

- **Inline element separator** — `<br>` now emits whitespace, fixing fused
  headlines (`BusinessEfficiency` → `Business Efficiency`).
- **`blocks` on the Page Input DTO** — `IngestPageInput` carries them and
  `persist.ts:533` ships them.

So steps 1–2 of the sequence are complete. **The crawler needs no changes.**

## Changes (all in `Geek-Crawler-Rag`)

### 1. Plaintext derivation — the one new primitive

New module `src/geek_crawler_rag/block_text.py`. **One function, imported by
both the chunker and `citation_verify`** — not two identical loops. Duplicated
loops drift, and that drift is this entire bug: one repo migrated, the other did
not.

```python
ROW_CELL_SEPARATOR = " | "
BLOCK_SEPARATOR = "\n\n"


def render_block_text(block: dict) -> str:
    """The single, authoritative text projection for one Block.

    Shared by both chunking and verification engines to eliminate drift.
    """
    if not block:
        return ""

    kind = block.get("kind")
    if kind == "row":
        # `|` survives normalization and acts as an explicit structural anchor.
        return ROW_CELL_SEPARATOR.join(
            c.strip() for c in block.get("cells", []) if c is not None
        )

    return (block.get("text") or "").strip()


def derive_plaintext_from_blocks(blocks) -> str:
    """Whole-page projection. The join is shared for the same reason the
    per-block projection is: two consumers that agree on rendering but differ on
    the join still produce different strings, and the invariant is equality of
    the final text, not of its parts."""
    if not blocks:
        return ""
    rendered = (render_block_text(b) for b in blocks)
    return BLOCK_SEPARATOR.join(line for line in rendered if line.strip())
```

**`row` is the only Block kind without `.text`** (`extract-content.ts:187` — it
carries `cells: string[]`). A naive `b.text` map emits nothing for every table
row, dropping tables out of the verification text while the chunker still
indexes them: a cited table fact becomes unverifiable, silently.

**Separator semantics, precisely:**

- `citation_verify._normalize_ws` (`:13-16`) applies `\s+ → " "` and lowercases
  *before* comparison. So `\n` versus `\n\n` has **no effect on quote matching**.
  Per-kind newline rules are a chunk-boundary and token-counting concern, not a
  verification one — uniform `\n\n` is correct here, and refining it for
  `listItem`/`row` should wait until the chunker needs it.
- `|` is **not** whitespace, so it survives normalization and is load-bearing.
  It appears in both the embedded chunk and the verification target, which is
  why it must come from one shared function rather than two implementations.

### 2. Public contract — `page-text`

- `citation_verify.py:19` — `quote_in_markdown(quote, markdown)` →
  `quote_in_text(quote, plain_text)`; `:40`/`:53`/`:63` read the derived text.
  Keep `_normalize_ws` unchanged.
- **Short-cell contextual exact match.** `citation_verify.py:22` rejects any
  quote under 12 normalized characters. Tables now enter the verification text
  and cell facts are routinely shorter — `"$15/month"` is 9, `"99.9%"` is 5 — so
  the floor turns correct citations into false-negative verification failures.

  Under the floor, fall through to an exact cell match instead of returning
  `False`: the quote passes when it equals a whole cell, i.e. it appears in the
  plaintext bounded by `ROW_CELL_SEPARATOR` or by a line edge. Whole-cell
  bounding is what makes this safe — it is an exact field match, not a substring
  coincidence, and the bounds come from the same shared constant that produced
  the text.

  One guard worth keeping: require at least one alphanumeric character, so
  punctuation-only fragments cannot verify. Residual and accepted: a common
  short cell (`"Yes"`) verifies against any page whose tables contain it. That
  is a weak citation rather than a false one — the match is still whole-cell and
  still scoped to the cited page's own text.
- `app.py:361,389` — `/page-markdown` → `/page-text`, `PageMarkdownResponse` →
  `PageTextResponse` with a `text` field, built via `blocks_to_text`.
  **Breaking for ContentCreatorV2**, which grounds on this endpoint — it ships in
  the same release.

### 3. Read the new schema

- `mongo.py`
  - `CrawlPage` (`:65`) — replace `markdown` with `content_html` and `blocks`.
  - Projections at `:212`, `:299`, `:325` — request
    `ContentHtml`/`contentHtml` and `Blocks`/`blocks`, keeping the existing
    both-casings hedge; drop `Markdown`/`markdown`.
  - `_page_from_doc` (`:359-389`) — map the new fields.
- `unusable.py` — drop the `markdown` parameter and the `no_markdown` reason;
  gate on `blocks` being non-empty. Locale, failure and robots branches unchanged.
- `extract.py:49` — `page_text_and_title` takes `blocks` and `title`; delete the
  `del html` line and the `used_markdown` flag.
- `llama_nodes.py:38-61` — chunk from `blocks`: heading blocks give real parent
  boundaries for `parent_child_units` instead of token-count guesses, and each
  block's `anchors` carry onto node metadata. Replace
  `quality_score(has_markdown=…)` with a block-derived signal.

### 4. Scheduler state gate

- `mongo.py:146` — `find_smallest_markdown_ready_run` →
  `find_smallest_content_ready_run`, gating on `ContentReadyAt` (what the
  crawler sends, `persist.ts:262`).
- `mongo.py:99` — index `ix_crawl_runs_markdown_ready` →
  `ix_crawl_runs_content_ready` over `(Status, ContentReadyAt, Id)`.

**This is not a text substitution.** `mongo.py:171` passes
`hint="ix_crawl_runs_markdown_ready"`, and Mongo **errors on a hint naming an
index that does not exist** — a rename in one edit breaks every scheduler scan.
Four deploys, in this order:

1. Deploy code that creates `ix_crawl_runs_content_ready` natively, leaving the
   legacy index alive and the query untouched.
2. Verify key casing against the incoming GeekAPI payload structure
   (`ContentReadyAt` vs `contentReadyAt`).
3. Swap the query filter and the `hint=` parameter together in a single commit.
4. Drop `ix_crawl_runs_markdown_ready` once scan metrics are stable.

The page projections hedge both casings; the run filter and the index cannot,
which is why step 2 gates step 3 — see the external dependency below.

### 5. Retire selective pruning

- `indexer.py` — delete `_delete_unusable` and `_delete_page_points`; the three
  call sites (`:467`, `:484`, `:487`) increment a counter and `continue`.
  Indexing becomes read-only over Mongo; the crawler already owns rejection
  (`reject.ts` taxonomy, prose floor, locale exclusion).
- `models.py:38-41` — remove `pages_deleted_locale`, `pages_deleted_failure`,
  `pages_deleted_empty`, `pages_deleted_non_english`; keep
  `pages_skipped_lang` / `pages_skipped_empty`.
- Leave `delete_by_run_id` (`indexer.py:335`) alone — whole-run purge on
  re-index is correct and is not selective pruning.

## External dependency — GeekAPI (.NET) — **CONFIRMED NEGATIVE**

Read from source 2026-09-18, `GeekBackend/GeekAPI/Controllers/GeekCrawler/GeekCrawlerIngestController.cs`.
No crawl was needed; the DTO settles it.

```csharp
public record IngestPageItem(
    string? Origin, string? Url, string? FinalUrl,
    int StatusCode, bool RobotsAllowed,
    string? Html,
    string? FailureReason = null,
    string? Title = null,
    string? Markdown = null,
    string? Excerpt = null);          // :860 — no ContentHtml, no Blocks, no Text

public record IngestPatchRunRequest(
    ..., DateTimeOffset? MarkdownReadyAt = null,
    bool ClearMarkdownReadyAt = false, ...);   // :847 — no ContentReadyAt
```

ASP.NET ignores unknown JSON properties, so `contentHtml`, `blocks` and
`contentReadyAt` are discarded on arrival. `:529-538` maps only
`Html, FailureReason, Title, Markdown, Excerpt` into
`CreateGeekCrawlerPageItemCommand`.

**Why §2e's "ingest fails closed" prediction did not hold.** Acceptance at
`:521-524` requires `Html` **or** `Markdown`, and the crawler still sends
`html`. So the payload passed validation while the content that mattered was
thrown away — it failed open and silent, which is how 5,274 pages were stored
and then deleted rather than refused at the door.

This is the root cause of the whole failure chain: pages arrive without
`Markdown`, the Library's `classify_unusable_page` returns `no_markdown`,
`_delete_unusable` removes page and vectors, and `chunksUpserted` is 0.
`MarkdownReadyAt` stays null for the same reason, starving the scheduler.

**So this is no longer a confirmation step — it is work item #1**, and nothing
in Geek-Crawler-Rag can be built before it:

1. `IngestPageItem` — add `ContentHtml`, `Blocks`, and `Text` if wanted
2. `IngestPatchRunRequest` — add `ContentReadyAt` / `ClearContentReadyAt`
3. `CreateGeekCrawlerPageItemCommand` + the Mongo write — persist them
4. Acceptance at `:521-524` — require real content, so a page with markup but
   no extract is refused rather than stored and later deleted

## Sequence

1. Confirm the GeekAPI/Mongo fields exist (above).
2. Sections 1–3 — plaintext primitive, public contracts, schema reads.
3. Section 4 + drop the legacy corpus and indexes.
4. Section 5.
5. Pristine re-crawl. The extract cache holds today's 5,274 pages if replay is
   preferred to re-fetching.

## Verification

- **Unit** — `render_block_text` emits `|`-joined cells for a `row` and plain
  text for every other kind (the regression above); the chunker and
  `citation_verify` produce byte-identical strings for the same blocks, asserted
  by both calling `derive_plaintext_from_blocks` rather than by comparing two
  implementations; `classify_unusable_page` returns `None` for a page with
  blocks and no markdown; `quote_in_text` matches a quote spanning two blocks,
  a quote drawn from a table row, and a sub-12-character cell quote
  (`"$15/month"`) via contextual exact match — while still rejecting a
  sub-12-character substring that is not whole-cell bounded.
- **Non-destruction** — index a run containing an unusable page; assert
  `mongo.delete_page` is never called and the page survives. This is the
  regression that cost 5,274 pages.
- **End to end** — crawl one small seed, confirm
  `GET /crawls/{runId}/rag-index` reports `chunksUpserted > 0`, then confirm
  `page-urls` is still non-empty *after* indexing.
- **Chunk quality** — spot-check a chunk's heading path against the extract
  cache (`listCachedPages` / `readCachedPages`, `src/storage/extract-cache.ts`)
  as the reference extraction.
