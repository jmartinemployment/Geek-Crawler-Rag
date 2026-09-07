"""Reciprocal Rank Fusion for hybrid retrieval."""

from __future__ import annotations

from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    ranked_lists: list[list[T]],
    *,
    k: int = 60,
) -> list[tuple[T, float]]:
    """Merge multiple ranked id lists via RRF. Higher score is better."""
    scores: dict[T, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
