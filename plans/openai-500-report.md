# OpenAI text-embedding-3-small — HTTP 500 report

Evidence for a vendor support ticket. Everything here is measured, not inferred.

## 1. Character / word count

| | Batch A | Batch B |
|---|---|---|
| inputs in the array | 37 | 46 |
| total characters | 11,197 | 43,361 |
| approx words | ~1,800 | ~7,000 |
| tokens (our count) | 2,326 | 8,715 |
| per-input chars: min | 4 | 43 |
| per-input chars: median | 188 | 920 |
| per-input chars: max | 2,058 | 3,707 |

Largest single input is roughly 900 tokens, against an 8,191-token per-input ceiling.

## 2. Array or single string

**Array of multiple strings** — 37 and 46 elements respectively. Configured maximum is 56 per
request. Never a single string.

## 3. Special characters, code snippets, non-English scripts

- **Control characters: zero** in both batches, verified character by character.
- **Non-ASCII: 31 and 134 characters**, all typographic — `EN DASH`, `EM DASH`,
  `HORIZONTAL ELLIPSIS`, `LEFT SINGLE QUOTATION MARK`, `RIGHT SINGLE QUOTATION MARK`. Smart quotes
  and dashes from marketing copy.
- **No non-English scripts** — no CJK, no RTL, no Cyrillic, no emoji.
- **No code snippets.**
- **No LLM special tokens.** Scanned all 11,752 source pages for `<|endoftext|>`, `<|im_start|>`,
  `<|im_end|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>` — **zero
  occurrences**. One page contains a bare `<|` sequence with no token form.

Content is English marketing and product web copy.

## 4. Configuration

```
model            text-embedding-3-small
dimensions       1536
batch size       56 max (failures were 37 and 46)
client           OpenAI Python SDK via llama-index
max_retries      0
httpx timeout    60s read, 10s connect
concurrency      serialised — one embedding call in flight at a time
```

**Not rate limiting.** Account ceiling is 1,000,000 TPM (`x-ratelimit-limit-tokens`, confirmed by a
live probe); the client throttles to 400,000. At the time of the first failure the rolling window was
empty — it was the job's first batch — and our throttle counters recorded 0 waits and 0 rate-limit
deferrals.

## 5. The two failures

Both HTTP 500, `type: server_error`, message *"The server had an error while processing your request.
Sorry about that!"*

```
req_2f6f8d21426f4065a5248ca39811c1cd    2026-09-29 17:34 UTC
req_fede7d5ceb3f4a60b07cb3a705fbc175    2026-09-29 17:57 UTC
```

## 6. The caveat that matters

**Both batches succeeded on the next attempt with identical content.** The same chunks, byte for
byte, embedded without error — one run went from 3,066 chunks to 83,234 on its second attempt. That
rules out the payload as the cause and points at a transient server-side condition in that window.

## 7. What has been ruled out, and how

| hypothesis | verdict | method |
|---|---|---|
| malformed or poisoned content | ruled out | identical bytes succeeded on retry |
| LLM special tokens in input | ruled out | scanned 11,752 pages, zero occurrences |
| oversized input | ruled out | max input ~900 tokens vs 8,191 ceiling |
| empty or null input | ruled out | zero-length item count is 0 in both batches |
| rate limiting | ruled out | 40% of ceiling; limiter never engaged; failed on an empty window |
| client-side memory pressure | ruled out for this window | OOM events on this host were 2026-09-25, not 09-29 |
| concurrency on the HTTP client | ruled out | calls are serialised behind a lock |

No cause has been found on the client side. The request IDs above are the remaining route.
