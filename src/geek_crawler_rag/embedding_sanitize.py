"""Sanitize text before OpenAI embedding batches.

OpenAI text-embedding-3-small can return undocumented HTTP 500s when a batch
contains null bytes, corrupt encoding, or certain control characters. One bad
item rejects the entire LlamaIndex batch POST.

What "clean" means (encoding hygiene only)
-----------------------------------------
REMOVED:
  - Null bytes (\\x00) and other C0 controls except tab / LF / CR
  - Lone UTF-16 surrogates (round-trip UTF-8 with replace)
  - Zero-width / BOM / soft-hyphen (U+200B–U+200D, U+FEFF, U+00AD)

NOT done here (by design):
  - HTML/Markdown stripping
  - Semantic rewriting
  - Token truncation (handled by partition_embedding_batches)

Sanitization must not remove meaningful English prose beyond the junk above.
"""

from __future__ import annotations

import re

# Keep tab / LF / CR; strip other C0 controls and DEL.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff\u00ad]")


def sanitize_embedding_text(text: str | None) -> str:
    """Return embedding-safe UTF-8 text; empty input becomes empty string."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""

    # Normalize corrupt / lone-surrogate sequences without raising.
    cleaned = text.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="replace"
    )
    cleaned = cleaned.replace("\x00", "")
    cleaned = _CONTROL_RE.sub("", cleaned)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    return cleaned


def sanitize_embedding_texts(texts: list[str]) -> tuple[list[str], int]:
    """Sanitize a batch; return cleaned texts and count of mutated items."""
    cleaned: list[str] = []
    mutated = 0
    for text in texts:
        safe = sanitize_embedding_text(text)
        if safe != text:
            mutated += 1
        cleaned.append(safe)
    return cleaned, mutated
