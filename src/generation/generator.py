from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..vectorstore.base import SearchHit
from .guardrails import GuardrailResult, check_answer
from .llm_client import LLMClient, TokenUsage
from .prompt import SYSTEM_PROMPT, build_user_prompt


@dataclass
class Citation:
    index: int
    title: str
    section: str
    source_path: str
    effective_date: str | None
    version: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    confidence: float
    insufficient_evidence: bool
    ambiguous: bool = False
    clarification_options: list[str] = field(default_factory=list)
    guardrail: GuardrailResult | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    used_llm: bool = False


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that "
    "confidently. You may want to check with the relevant department directly, "
    "or rephrase the question if you think it should be covered."
)


def _citations_from_hits(hits: list[SearchHit]) -> list[Citation]:
    return [
        Citation(
            index=i,
            title=h.record["title"],
            section=h.record["section"],
            source_path=h.record["source_path"],
            effective_date=h.record.get("effective_date"),
            version=h.record["version"],
        )
        for i, h in enumerate(hits, start=1)
    ]


def generate_answer(query: str, hits: list[SearchHit], confidence: float, llm: LLMClient) -> Answer:
    citations = _citations_from_hits(hits)

    if llm.is_available():
        system = SYSTEM_PROMPT
        user = build_user_prompt(query, hits)
        text = llm.complete(system=system, user=user, max_tokens=500)
        guardrail = check_answer(text, num_context_excerpts=len(hits))
        return Answer(
            text=text,
            citations=citations,
            confidence=confidence,
            insufficient_evidence=False,
            guardrail=guardrail,
            token_usage=llm.last_usage(),
            used_llm=True,
        )

    return _extractive_answer(hits, citations, confidence)


def _extractive_answer(hits: list[SearchHit], citations: list[Citation], confidence: float) -> Answer:
    """No LLM configured: return the retrieved excerpts verbatim rather than
    attempting to paraphrase (which would require the very generation
    capability that's unavailable). Explicitly labeled as extractive so it's
    never mistaken for a synthesized answer."""
    lines = ["(Extractive mode — no LLM configured, showing top matching excerpts verbatim)\n"]
    for c in citations:
        hit = hits[c.index - 1]
        snippet = hit.record["text"].strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "…"
        lines.append(f"[{c.index}] {snippet}")
    text = "\n\n".join(lines)
    guardrail = check_answer(text, num_context_excerpts=len(hits))
    return Answer(
        text=text,
        citations=citations,
        confidence=confidence,
        insufficient_evidence=False,
        guardrail=guardrail,
        used_llm=False,
    )
