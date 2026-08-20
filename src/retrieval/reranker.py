"""Reranking stage, applied after hybrid retrieval and before the confidence
gate / generation.

Production (Azure): Azure AI Search's managed semantic ranker already
reranked the fused hybrid candidates server-side (see vectorstore/azure_search.py),
so this module is a no-op there — `hits` already carry `@search.reranker_score`.

Local: a lightweight lexical-overlap reranker (no torch/cross-encoder
dependency — see src/embeddings/local.py for why). It re-scores the RRF
shortlist by exact content-word overlap between the query and each chunk's
text+section, which corrects a common hybrid-search miss: a chunk that ranks
high on embedding similarity alone but doesn't actually contain the specific
term the user asked about (e.g. a neighboring section that *talks about* the
same policy without stating the number being asked for).
"""
from __future__ import annotations

import re

from ..vectorstore.base import SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "and", "or", "for", "what", "how"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def rerank_local(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    query_tokens = _tokens(query)
    if not query_tokens or not hits:
        for h in hits:
            h.rerank_score = h.score
        return hits

    max_fused = max((h.score for h in hits), default=1.0) or 1.0
    scored = []
    for h in hits:
        chunk_tokens = _tokens(h.record["text"]) | _tokens(h.record.get("section", ""))
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        fused_norm = h.score / max_fused
        h.rerank_score = 0.6 * overlap + 0.4 * fused_norm
        scored.append(h)

    scored.sort(key=lambda h: h.rerank_score, reverse=True)
    return scored
