"""Scenario 5 (ambiguous query): "What is the limit?" has no single answer —
the KB has expense limits, PTO caps, password length minimums, discount caps,
API call limits, etc. Answering with any one of them is a coin-flip that's
usually wrong; retrieving all of them and stuffing them in one prompt just
confuses the model. The fix is to *detect* the ambiguity before generation
and ask a clarifying question instead of guessing — matching how a human
analyst would respond to the same one-line question from a stranger.

Detection is two-signal, both must roughly agree:
  1. The query itself is short and has few content words (little to
     disambiguate on).
  2. The top retrieved hits, even after query rewriting, spread across
     several unrelated sections/documents with no single hit dominating —
     i.e. retrieval genuinely can't tell which "limit" is meant.
If conversation context already pinned down a topic (query rewriting folded
in a specific department/policy noun), rewritten queries stop looking short
and generic, and this naturally stops firing — ambiguity detection runs
*after* rewriting, not before.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..vectorstore.base import SearchHit

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "do", "does", "did", "for", "of", "to", "in", "on", "and", "or",
    "i", "we", "you", "it", "this", "that", "can", "my", "our", "there", "any",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class AmbiguityResult:
    is_ambiguous: bool
    candidate_topics: list[str]


def _content_words(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def detect_ambiguity(query: str, hits: list[SearchHit], max_content_words: int = 2, top_n: int = 5) -> AmbiguityResult:
    content_words = _content_words(query)
    if len(content_words) > max_content_words:
        return AmbiguityResult(is_ambiguous=False, candidate_topics=[])

    top_hits = hits[:top_n]
    distinct_topics: dict[tuple[str, str], SearchHit] = {}
    for h in top_hits:
        key = (h.record["doc_id"], h.record["section"])
        distinct_topics.setdefault(key, h)

    if len(distinct_topics) < 3:
        return AmbiguityResult(is_ambiguous=False, candidate_topics=[])

    labels = [f"{h.record['title']} — {h.record['section']}" for h in distinct_topics.values()]
    return AmbiguityResult(is_ambiguous=True, candidate_topics=labels[:top_n])
