"""Confidence scoring for Scenario 4 (hallucination / missing information).

The failure mode: a naive RAG pipeline always retrieves *something* (top-k is
fixed) and always asks the LLM to answer from it, so a question with no real
answer in the KB still gets a confident-sounding, made-up response.

The fix is a hard gate *before* generation: estimate how well the retrieved
context actually supports the query, and if it's below threshold, skip the
LLM call and return "insufficient evidence" instead of ever giving the model
a chance to fill the gap with invented content.

- Azure AI Search path: use `@search.reranker_score` from the managed semantic
  ranker (a calibrated 0-4 relevance score from a cross-encoder over the
  fused hybrid candidates) — Microsoft's own guidance treats ~<2.0 as low
  confidence.
- Local path: no calibrated cross-encoder is available, so this is a
  transparent heuristic combining (a) embedding cosine similarity and (b) the
  fraction of the query's content words that literally appear in the top
  retrieved chunks. It is intentionally conservative — documented as a
  heuristic, not a calibrated probability. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import re

from ..vectorstore.base import SearchHit

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "do", "does", "did", "for", "of", "to", "in", "on", "and", "or",
    "i", "we", "you", "it", "this", "that", "can", "my", "our", "there", "any",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def azure_confidence(hits: list[SearchHit]) -> float:
    if not hits:
        return 0.0
    return max(0.0, min(1.0, hits[0].score / 4.0))


def local_confidence(query: str, hits: list[SearchHit], top_n: int = 3) -> float:
    if not hits:
        return 0.0

    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0

    context_tokens: set[str] = set()
    vector_scores = []
    for h in hits[:top_n]:
        context_tokens |= _content_tokens(h.record["text"]) | _content_tokens(h.record.get("section", ""))
        vector_scores.append(h.vector_score)

    lexical_overlap = len(query_tokens & context_tokens) / len(query_tokens)
    vector_component = sum(max(0.0, v) for v in vector_scores) / len(vector_scores) if vector_scores else 0.0

    return 0.5 * lexical_overlap + 0.5 * min(1.0, vector_component * 2.0)


def compute_confidence(query: str, hits: list[SearchHit], use_azure: bool) -> float:
    return azure_confidence(hits) if use_azure else local_confidence(query, hits)
