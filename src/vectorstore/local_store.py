"""Local vector store: numpy cosine similarity + BM25, fused with Reciprocal
Rank Fusion (RRF) — the same fusion technique Azure AI Search's hybrid search
uses under the hood. This lets the local backend be a faithful stand-in for
Azure AI Search when developing/evaluating without cloud credentials: the
retrieval *behavior* (hybrid ranking, department filtering) matches; only the
embedding quality and the managed semantic reranker differ.

Persisted under data/index/local_<profile>/ as three files: records.jsonl
(one JSON object per chunk), embeddings.npy (float32 matrix, same row order),
and a re-tokenized BM25 index rebuilt at load time (cheap, keeps this file
self-contained rather than pickling a third-party class across versions).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .base import SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LocalVectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.bm25: BM25Okapi | None = None

    @property
    def records_path(self) -> Path:
        return self.index_dir / "records.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    def build(self, records: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        with open(self.records_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        np.save(self.embeddings_path, embeddings)
        self.records = records
        self.embeddings = embeddings
        self._build_bm25()

    def load(self) -> None:
        if not self.records_path.exists():
            raise FileNotFoundError(
                f"No local index found at {self.index_dir}. Run scripts/ingest.py first."
            )
        with open(self.records_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        self.embeddings = np.load(self.embeddings_path)
        self._build_bm25()

    def _build_bm25(self) -> None:
        corpus = [_tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int,
        department_filter: list[str] | None = None,
        use_hybrid: bool = True,
    ) -> list[SearchHit]:
        if not self.records:
            return []

        allowed_idx = (
            list(range(len(self.records)))
            if department_filter is None
            else [i for i, r in enumerate(self.records) if r["department"] in department_filter]
        )
        if not allowed_idx:
            return []

        vector_scores = self.embeddings[allowed_idx] @ query_vector
        vec_rank = np.argsort(-vector_scores)

        if use_hybrid and self.bm25 is not None:
            all_bm25 = self.bm25.get_scores(_tokenize(query_text))
            keyword_scores = np.array([all_bm25[i] for i in allowed_idx])
            kw_rank = np.argsort(-keyword_scores)
        else:
            keyword_scores = np.zeros(len(allowed_idx))
            kw_rank = vec_rank  # degrades to vector-only ranking

        rrf_scores = _reciprocal_rank_fusion(vec_rank, kw_rank, len(allowed_idx))

        order = np.argsort(-rrf_scores)[:top_k]
        hits = []
        for local_i in order:
            global_i = allowed_idx[local_i]
            hits.append(
                SearchHit(
                    record=self.records[global_i],
                    score=float(rrf_scores[local_i]),
                    vector_score=float(vector_scores[local_i]),
                    keyword_score=float(keyword_scores[local_i]),
                )
            )
        return hits


def _reciprocal_rank_fusion(rank_a: np.ndarray, rank_b: np.ndarray, n: int, k: int = 60) -> np.ndarray:
    """rank_a/rank_b: arrays of indices [0..n) sorted best-first by each ranker.
    Returns an RRF score per original index (higher is better)."""
    scores = np.zeros(n)
    for rank_list in (rank_a, rank_b):
        for position, idx in enumerate(rank_list):
            scores[idx] += 1.0 / (k + position + 1)
    return scores
