"""Lexical BM25 scoring helpers for hybrid retrieval."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t]


def bm25_rank_indices(query: str, documents: list[str]) -> list[int]:
    """Return document indices sorted by BM25 score (best first)."""
    if not documents:
        return []
    corpus = [tokenize(doc) for doc in documents]
    q = tokenize(query)
    if not q or not any(corpus):
        return list(range(len(documents)))
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q)
    return sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
