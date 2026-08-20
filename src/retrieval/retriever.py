"""Single-shot retrieval: query text -> ranked, deduped, version-resolved,
department-filtered list of chunks ready for the confidence gate / generator.

Handles three of the six assignment failure scenarios on its own:
  - Scenario 1 (wrong chunk) is mostly fixed upstream in chunking, but hybrid
    search + reranking here catch cases embeddings alone miss.
  - Scenario 2 (multi-document questions): `_expand_queries` splits an
    explicit comparison ("X vs Y", "compare A and B") into sub-queries so both
    sides actually get retrieved, instead of one side dominating top-k.
  - Scenario 3 (conflicting/superseded documents): `_resolve_versions` drops
    superseded family members unless the query explicitly asks for a past
    version/year.
"""
from __future__ import annotations

import re

from ..config import PipelineProfile, Settings
from ..embeddings.base import EmbeddingClient
from ..ingestion.catalog import latest_version_by_family, load_catalog
from ..vectorstore.base import SearchHit, VectorStore
from .reranker import rerank_local

_COMPARISON_RE = re.compile(r"\bvs\.?\b|\bversus\b|\bcompare\b|\bcompared to\b|\bdifference between\b", re.I)
_SPLIT_RE = re.compile(r"\band\b|,|/", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingClient, profile: PipelineProfile, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.profile = profile
        self.settings = settings
        self._catalog = load_catalog()
        self._latest_by_family = latest_version_by_family(self._catalog)

    def retrieve(self, query: str, allowed_departments: list[str] | None) -> list[SearchHit]:
        queries = self._expand_queries(query) if self.profile.use_multi_query_expansion else [query]

        merged: dict[str, SearchHit] = {}
        for q in queries:
            vector = self.embedder.embed([q])[0]
            hits = self.store.search(
                query_text=q,
                query_vector=vector,
                top_k=self.profile.top_k_retrieve,
                department_filter=allowed_departments if self.profile.use_department_acl else None,
                use_hybrid=self.profile.use_hybrid_search,
            )
            for h in hits:
                key = h.record["id"]
                if key not in merged or h.score > merged[key].score:
                    merged[key] = h

        hits = list(merged.values())
        hits.sort(key=lambda h: h.score, reverse=True)

        if self.profile.use_version_resolution:
            hits = self._resolve_versions(query, hits)

        if self.profile.use_reranking and not self.settings.use_azure():
            hits = rerank_local(query, hits)
        elif self.profile.use_reranking:
            hits.sort(key=lambda h: h.score, reverse=True)  # already reranked server-side

        return hits[: self.profile.top_k_final]

    def _expand_queries(self, query: str) -> list[str]:
        if not _COMPARISON_RE.search(query):
            return [query]
        # Pull out the clause(s) around the comparison keyword and split on
        # conjunctions so each side of the comparison gets its own retrieval pass.
        parts = [p.strip() for p in _SPLIT_RE.split(query) if len(p.strip()) > 2]
        if len(parts) < 2:
            return [query]
        return [query] + parts

    def _resolve_versions(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        mentioned_years = set(_YEAR_RE.findall(query))
        resolved = []
        for h in hits:
            family = h.record.get("doc_family")
            doc_id = h.record.get("doc_id")
            latest = self._latest_by_family.get(family)
            if latest is None or doc_id == latest:
                resolved.append(h)
                continue
            # Superseded document: keep it only if the query explicitly names
            # its year/version (e.g. "what was pricing in 2025") — otherwise
            # a superseded chunk would silently outrank the current policy.
            if mentioned_years and any(y in (h.record.get("source_path") or "") for y in mentioned_years):
                resolved.append(h)
                continue
            if mentioned_years and any(y in (h.record.get("effective_date") or "") for y in mentioned_years):
                resolved.append(h)
                continue
            # Drop silently (superseded + not explicitly requested).
        return resolved
