"""Post-generation guardrails.

Directly targets the Step 5 "Production Failure" scenario: "the chatbot gives
correct answers most of the time, but occasionally gives a completely wrong
answer with a valid-looking citation." A citation marker like [2] *looks*
trustworthy regardless of whether the model actually used excerpt 2 correctly
or made the number up / mis-attributed a claim to the wrong excerpt — so
citation markers must be verified mechanically, not trusted at face value:

  1. Every citation index must reference an excerpt that was actually in the
     retrieved context passed to the model (catches out-of-range/invented
     citation numbers).
  2. Every sentence carrying a factual claim should carry at least one
     citation; a substantial answer with zero citations is flagged.
  3. `groundedness_ratio` (cited sentences / total sentences) is surfaced to
     both the UI and the evaluation harness as a per-answer signal, not just
     a pass/fail gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REFUSAL_MARKERS = ("don't have enough information", "insufficient", "cannot find", "no information")


@dataclass
class GuardrailResult:
    cited_indices: set[int]
    invalid_citations: set[int]
    groundedness_ratio: float
    has_uncited_claims: bool
    flagged: bool
    flag_reason: str | None


def check_answer(answer_text: str, num_context_excerpts: int) -> GuardrailResult:
    cited = {int(m) for m in _CITATION_RE.findall(answer_text)}
    invalid = {i for i in cited if i < 1 or i > num_context_excerpts}

    is_refusal = any(marker in answer_text.lower() for marker in _REFUSAL_MARKERS)

    sentences = [s for s in _SENTENCE_SPLIT.split(answer_text) if s.strip()]
    if not sentences or is_refusal:
        groundedness_ratio = 1.0 if is_refusal else 0.0
        has_uncited = False
    else:
        cited_sentences = sum(1 for s in sentences if _CITATION_RE.search(s))
        groundedness_ratio = cited_sentences / len(sentences)
        has_uncited = cited_sentences < len(sentences)

    flagged = bool(invalid) or (not is_refusal and groundedness_ratio == 0.0 and len(sentences) > 0)
    reason = None
    if invalid:
        reason = f"Answer cites excerpt(s) {sorted(invalid)} which were not in the retrieved context."
    elif flagged:
        reason = "Answer makes claims with no citations at all."

    return GuardrailResult(
        cited_indices=cited,
        invalid_citations=invalid,
        groundedness_ratio=round(groundedness_ratio, 3),
        has_uncited_claims=has_uncited,
        flagged=flagged,
        flag_reason=reason,
    )
