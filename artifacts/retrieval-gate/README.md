# Retrieval gate artifacts

`scripts/retrieval_baseline.py` writes these. They exist so the embedding-stack rebuild in
`plans/go-local-embeddings.md` is judged against a measurement rather than an impression.

## Files

- **`queries.json`** — the fixed query set. **Do not regenerate it for the "after" run.** Sampling
  fresh queries would compare two different questions and report the difference as a result.
- **`before-openai-splade.json`** — captured 2026-09-30 against the live stack
  (`text-embedding-3-small` 1536-d + `prithivida/Splade_PP_en_v1`), before any change.

## The bar to beat

```
queries               80   (40 literal / 40 wide_parent)
literal_found         40/40   100%
literal_rank1         35/40   87.5%
wide_parent_returned  40/40
errors                 0
```

`literal_found` is the gate. It asks the same question `citation_verify.quote_in_text` asks — is the
exact string present in a retrieved passage — and it is the job the sparse channel exists for. A drop
there is a measured regression and the plan says stop.

## Two strata, because a uniform sample cannot see the risk

71.4% of this corpus has a parent byte-identical to its child, so most queries land where the
parent-tier change provably costs nothing.

- **`literal`** — exact tokens scraped from the corpus. Tests term binding.
- **`wide_parent`** — section titles from units whose parent is ≥2× its child. Tests the synthesis
  window no 200-token child contains; these are the queries most at risk when the parent tier stops
  being emitted for duplicates.

## Caveat on the literal set, stated rather than discovered later

About half are genuine product and vendor names — `QuickBooks`, `NetSuite`, `bill.com`,
`AvidXchange`, `SquareWorks`. The rest are CamelCase **extraction artifacts**: `TypeSales`,
`ManagementArticle`, `NewsApril`, `ReceiptNumber` — boilerplate text concatenating without spaces.

They are still valid exact-token probes: the question is whether the index returns the passage holding
that precise string, and that is answerable regardless of whether the string is meaningful. But they
are weak proxies for the names that matter commercially, and **curating `queries.json` by hand is
worth more than any regex I can write** — you know which vendor and product names a draft must be
able to ground on. Editing the file is supported; it is read verbatim.

Their presence is also a signal about the corpus rather than the gate: boilerplate is reaching chunk
text without word boundaries, which is adjacent to the 72–76%-editorial composition problem tracked
separately.

## Usage

```bash
# after the rebuild, reusing the SAME query set
python scripts/retrieval_baseline.py --queries artifacts/retrieval-gate/queries.json \
                                     --out /tmp/after.json
python scripts/retrieval_baseline.py --compare \
    artifacts/retrieval-gate/before-openai-splade.json /tmp/after.json
```

`--compare` exits non-zero on a regression and names every literal that stopped being found.
