"""Conversation-aware query rewriting (Scenario 6: conversational context).

The failure mode being fixed: if you embed raw chat history + the new turn,
a short follow-up like "What about Standard?" retrieves on the words
"What about Standard" alone (or, worse, on the concatenated noisy history),
which is a poor query. If you *don't* use history at all, "What about
Standard?" retrieves nothing useful either since it has almost no content
words. The fix is to condense history + new turn into one standalone query
*before* retrieval, and retrieve on that alone — never on raw history.
"""
from __future__ import annotations

import re

from ..generation.llm_client import LLMClient

_FOLLOWUP_STARTERS = (
    "what about", "and ", "what if", "is there", "any exception", "how about",
    "and what", "what's the", "whats the", "also", "same for",
)

_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's latest message as a single, standalone question that "
    "makes sense with no prior context. Preserve their intent exactly. Do not "
    "answer the question. Output only the rewritten question, nothing else."
)


def _looks_like_followup(message: str) -> bool:
    msg = message.strip().lower()
    if len(msg.split()) <= 5:
        return True
    return any(msg.startswith(s) for s in _FOLLOWUP_STARTERS)


def rewrite_query(
    message: str,
    history: list[dict[str, str]],
    llm: LLMClient | None,
    window: int,
) -> str:
    """history: list of {"role": "user"|"assistant", "content": str}, oldest first."""
    if window <= 0 or not history:
        return message

    if not _looks_like_followup(message):
        return message

    recent = history[-window:]

    if llm is not None and llm.is_available():
        convo = "\n".join(f"{h['role']}: {h['content']}" for h in recent)
        prompt = f"Conversation so far:\n{convo}\n\nLatest message: {message}"
        rewritten = llm.complete(system=_REWRITE_SYSTEM_PROMPT, user=prompt, max_tokens=80)
        rewritten = rewritten.strip().strip('"')
        return rewritten or message

    # No LLM available: cheap fallback — fold the previous user question's
    # content words into the new message so bag-of-words/embedding retrieval
    # still has something to match against.
    prior_user_turns = [h["content"] for h in recent if h["role"] == "user"]
    if not prior_user_turns:
        return message
    last_user = prior_user_turns[-1]
    combined = f"{last_user} {message}"
    return re.sub(r"\s+", " ", combined).strip()
