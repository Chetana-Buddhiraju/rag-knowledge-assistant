from __future__ import annotations

from ..vectorstore.base import SearchHit

SYSTEM_PROMPT = """You are the Northwind Traders internal Knowledge Assistant. \
Answer the user's question using ONLY the numbered context excerpts provided. \
Every factual claim must be followed by a citation like [1] or [2] referencing \
the excerpt(s) it came from. If the excerpts do not contain enough information \
to answer confidently, say so explicitly instead of guessing — do not use \
outside knowledge and do not speculate. If two excerpts conflict, prefer the \
one with the later effective date and mention that a newer version exists. \
Keep answers concise and directly responsive to the question."""


def format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        r = h.record
        header = f"[{i}] {r['title']} — {r['section']}"
        if r.get("effective_date"):
            header += f" (effective {r['effective_date']}, v{r['version']})"
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, hits: list[SearchHit]) -> str:
    context = format_context(hits)
    return f"Context excerpts:\n\n{context}\n\nQuestion: {query}"
