"""Top-level orchestrator wiring every stage together:

  conversation history + new message
    -> query rewrite               (Scenario 6)
    -> ambiguity check             (Scenario 5)
    -> hybrid retrieval + ACL       (Scenario 1, Q4)
    -> version resolution          (Scenario 3)
    -> reranking                   (Scenario 1)
    -> confidence gate             (Scenario 4)
    -> generation + citations      (Scenario 2, all)
    -> citation guardrail          (Step 5 "production failure")

Instantiate once per (backend, profile) — `RAGPipeline(settings)` — and reuse
across turns; it loads the index once. Every stage's wall-clock time is
recorded on the returned RAGResult so latency can be broken down per stage
(Step 5, Q2) without re-running anything under a profiler.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .embeddings.factory import get_embedding_client
from .generation.generator import Answer, generate_answer
from .generation.llm_client import get_llm_client
from .retrieval.ambiguity import detect_ambiguity
from .retrieval.confidence import compute_confidence
from .retrieval.query_rewrite import rewrite_query
from .retrieval.retriever import Retriever
from .vectorstore.base import SearchHit
from .vectorstore.factory import get_vector_store


@dataclass
class RAGResult:
    original_query: str
    rewritten_query: str
    hits: list[SearchHit]
    answer: Answer | None
    ambiguous: bool
    clarification_options: list[str]
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.profile = settings.profile
        self.embedder = get_embedding_client(settings)
        self.store = get_vector_store(settings, embedding_dim=self.embedder.dim)
        if not settings.use_azure():
            self.store.load()
        self.llm = get_llm_client(settings)
        self.retriever = Retriever(self.store, self.embedder, self.profile, settings)

    def ask(self, query: str, allowed_departments: list[str] | None, history: list[dict[str, str]] | None = None) -> RAGResult:
        history = history or []
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        rewritten = (
            rewrite_query(query, history, self.llm, self.profile.conversation_window)
            if self.profile.use_query_rewrite
            else query
        )
        timings["rewrite_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = self.retriever.retrieve(rewritten, allowed_departments)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        if self.profile.ambiguity_detection:
            ambiguity = detect_ambiguity(rewritten, hits)
            if ambiguity.is_ambiguous:
                return RAGResult(
                    original_query=query,
                    rewritten_query=rewritten,
                    hits=hits,
                    answer=None,
                    ambiguous=True,
                    clarification_options=ambiguity.candidate_topics,
                    timings_ms=timings,
                )

        if not hits:
            timings["generate_ms"] = 0.0
            return RAGResult(
                original_query=query,
                rewritten_query=rewritten,
                hits=[],
                answer=Answer(
                    text=(
                        "I don't have enough information in the knowledge base to "
                        "answer that confidently."
                    ),
                    citations=[],
                    confidence=0.0,
                    insufficient_evidence=True,
                ),
                ambiguous=False,
                clarification_options=[],
                timings_ms=timings,
            )

        confidence = compute_confidence(rewritten, hits, self.settings.use_azure())

        if confidence < self.profile.min_confidence:
            timings["generate_ms"] = 0.0
            return RAGResult(
                original_query=query,
                rewritten_query=rewritten,
                hits=hits,
                answer=Answer(
                    text=(
                        "I don't have enough information in the knowledge base to "
                        "answer that confidently."
                    ),
                    citations=[],
                    confidence=confidence,
                    insufficient_evidence=True,
                ),
                ambiguous=False,
                clarification_options=[],
                timings_ms=timings,
            )

        t0 = time.perf_counter()
        answer = generate_answer(rewritten, hits, confidence, self.llm)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = sum(timings.values())

        return RAGResult(
            original_query=query,
            rewritten_query=rewritten,
            hits=hits,
            answer=answer,
            ambiguous=False,
            clarification_options=[],
            timings_ms=timings,
        )
