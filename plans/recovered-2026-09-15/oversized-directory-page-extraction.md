# Filter oversized link-directory pages before they inflate the corpus

Status: **Selector identified, extraction fix not yet implemented**
Found: **2026-09-11** while auditing why n8n `/integrations/*` pages hit the
500,000-char truncation cap (see `plans/crawl-scope-and-duplicate-pages.md`,
already shipped in `36f8785` / `34a0aad`)

## Objective

n8n's `/integrations/if/` and `/integrations/set/` pages truncate at the
500,000-char markdown cap. The initial theory — an embedded Nuxt/Next
client-hydration JSON blob leaking into extracted text — was **tested and is
wrong**. The real cause and a concrete, testable fix are below.

## What is actually happening (verified, not inferred)

Fetched `https://n8n.io/integrations/if/` with a real browser UA (Cloudflare
blocks the default Node fetch UA — confirm any earlier "content" observed with
plain `curl`/`node https.get` wasn't a Cloudflare interstitial before trusting
it) and ran it through the exact extraction path in `extract-content.ts`:

```
raw HTML:                          7,124,342 bytes
<script id="__NUXT_DATA__">        4,032,112 bytes   ← REMOVED by Readability
remaining visible body text:         420,628 chars   ← this is what gets kept
<a class="card card--default">         5,857 real workflow-gallery cards
gallery grid starts at byte 91,507 of 7,124,080 (1.3% into the page)
```

Mozilla Readability's `_prepDocument` strips `<script>` tags — including
`type="application/json"` ones — before scoring content. Confirmed by removing
the `__NUXT_DATA__` node manually and diffing `body.textContent` length: it
barely changes (420,628 vs the ~414,076 Readability itself extracts). The 500k
cap is being hit by **5,857 legitimate, visible workflow-gallery cards** (e.g.
"Automate multi-platform social media content creation with AI"), not by JSON
application state. The gallery starts 1.3% into the document — the actual node
description prose is a small fraction of the page; almost everything after it
is gallery cards.

**Why this looked like a hydration-blob problem:** the page fingerprints as
Nuxt (`__NUXT__`/`__NUXT_DATA__` present), and the text-to-raw-HTML ratio is
unusually low (6.49% vs clickup's 20.90% on a normal blog post). Both are real
signals — they just don't mean "large inert payload instead of the content"
here. They mean "large inert payload *alongside* legitimate content," which is
a different fix.

## Root cause and fix: a stable, identifiable DOM boundary exists

The gallery is a consistent, framework-rendered component, not scattered
inline links:

```
container:  <section class="w-full px-section-gap-x ..."> 
              <div class="mx-auto w-full max-w-section-default">
                <div class="grid grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
                  <a class="card card--default rounded-small p-6 flex min-h-40 ..." href="/workflows/...">
```

`a.card--default` matched **5,857** times — within 2 of the 5,855 counted via
`href^="/workflows/"` earlier (the small delta is likely dedup during parse).
This is a selector, not a heuristic: strip the enclosing `.grid` (or the
`<section>` wrapping it) before handing the document to Readability, and the
node's actual description prose survives untouched because it lives before
this container in document order.

**Implementation, in `extract-content.ts`:** before constructing the
`Readability` reader, run a small Cheerio/DOM pass that removes elements
matching a short allowlist of known gallery-container selectors (start with
`.grid:has(a.card--default)` or the closest stable ancestor found per-site).
This is framework-specific — n8n's Nuxt build uses these exact class names —
so the selector list is expected to need an entry per site family encountered,
not one universal rule.

## Options considered

1. **Do nothing new.** Free; the truncation-visibility fix (`34a0aad`) means
   these pages are no longer silently cut, just still capped mid-gallery.
   Leaves ~5,700 cards' worth of markdown noise in every affected page.
2. **Generic link-density heuristic** ("region with >K consecutive `<a>` tags
   and <M avg text/link, drop it"). Works without inspecting each site, but
   carries real maintenance risk: could clip legitimate long-form lists,
   indices, or glossaries on other sites. Not chosen — a concrete selector
   exists for this case, so there's no need to guess.
3. **Skip `/integrations/*` entirely at the crawl-scope layer**
   (`filterEnqueueUrls`/`sitemap.ts`). Cheapest at runtime, but throws away the
   real node-description prose along with the gallery — confirmed above that
   the prose is real and comes *before* the gallery in the DOM, so this is
   strictly worse than the selector strip for no runtime savings that matter.
4. **Raise `MAX_MARKDOWN_CHARS`.** Rejected — does not address the cause, and
   works against the embedding-cost reductions shipped in `a99e386`.

**Chosen: selector-based container removal (the approach in "Root cause and
fix" above).** It is Option 2's intent without Option 2's fragility, because a
concrete, verified selector replaces the guessed heuristic.

## Decision log (kept for the next time this comes up)

- Nuxt hydration JSON theory: tested, false. Do not re-propose without
  re-testing — see the byte-count reproduction above.
- Playwright routing: not the fix. The gallery is already server-rendered and
  present in raw HTML; Playwright would render the identical DOM Cheerio
  already fetches. The promotion-to-Playwright path (`cheerio-runner.ts`)
  triggers on `viability.reason` (pages that look empty/broken), which these
  pages are not.
- Generic density heuristics: available as a fallback if selector-based
  removal proves too site-specific to maintain at scale, but not the first
  approach given a concrete selector already exists for the case in hand.

## Verification

1. Re-crawl n8n from a clean slate (all four prior n8n runs and their Qdrant
   points were deleted 2026-09-11) to get a baseline free of the pre-fix scope
   and duplicate-row issues.
2. On `/integrations/if/` and `/integrations/set/` specifically: confirm the
   extracted markdown drops from 500,000 (truncated) to a size consistent with
   prose-only content, and that the `truncated` flag no longer fires for them.
3. **Guardrail — do not clip low-gallery pages.** Sample a handful of
   `/integrations/*` pages across `sitemap-integrations.xml` (2,171 total) for
   nodes used in only a few workflows (small or absent gallery). Confirm the
   selector removal is a no-op on those pages and their (already-short)
   content is untouched.
4. Confirm `duplicatePagesSkipped` in the finished run stub — never observed
   to completion yet; the last live run reached 873 pages before this
   investigation paused it.
5. Spot check that the retained content on a gallery page is genuinely the
   node's description prose and not incidentally empty (i.e. the selector
   removal didn't also eat the prose if the DOM structure differs slightly
   between node pages).

## Out of scope here

- The scope-drift and duplicate-row fixes — already shipped, this file is
  purely about the truncation/gallery finding.
- A generic cross-site density heuristic — deferred unless selector-based
  removal turns out not to generalize past n8n's specific markup.
- Deciding whether workflow-gallery *links* themselves have retrieval value
  (e.g. as a separate lightweight index) — this plan only removes them from
  the primary content extraction; it does not propose capturing them
  elsewhere.
