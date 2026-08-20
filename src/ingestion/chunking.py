"""Two chunking strategies, deliberately kept side by side so the eval harness
can demonstrate *why* the baseline fails Scenario 1 ("Correct Document, Wrong Chunk").

Baseline: flattens every block to one string per document and slides a fixed-size
window across it with no overlap and no respect for section/table boundaries.
This is what most "hello world" RAG tutorials do, and it reliably shreds tables:
a limit value ends up in a different chunk than the category name it belongs to,
so the embedding for that chunk no longer looks like the question being asked.

Improved: chunks per-section (so a heading and its content travel together),
keeps tables atomic (never splits a table mid-row), and only falls back to a
sliding window with overlap for long prose sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .parsers import Block

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_baseline(blocks: list[Block], doc_id: str, chunk_size: int = 300) -> list[Chunk]:
    """Naive fixed-size, no-overlap, section-blind chunking."""
    full_text = "\n".join(b.text for b in blocks)
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(full_text), chunk_size)):
        piece = full_text[start : start + chunk_size].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::b{i}",
                text=piece,
                section="(unstructured)",
                chunk_index=i,
            )
        )
    return chunks


def chunk_improved(
    blocks: list[Block],
    doc_id: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Section-aware chunking: group blocks by heading, keep tables atomic,
    split long prose sections on sentence boundaries with overlap."""
    sections: list[tuple[str, list[Block]]] = []
    for b in blocks:
        if sections and sections[-1][0] == b.section:
            sections[-1][1].append(b)
        else:
            sections.append((b.section, [b]))

    chunks: list[Chunk] = []
    idx = 0
    for section, section_blocks in sections:
        is_table = section.endswith("(table)") or section.startswith("Sheet:")
        section_text = "\n".join(b.text for b in section_blocks).strip()
        if not section_text:
            continue

        if is_table or len(section_text) <= chunk_size:
            pieces = [section_text]
        else:
            pieces = _split_with_overlap(section_text, chunk_size, overlap)

        for piece in pieces:
            # Repeat the heading in every piece so the embedding always carries
            # the section topic, even for the 2nd/3rd overlapping slice.
            enriched = f"{section}\n{piece}" if not piece.startswith(section) else piece
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::c{idx}",
                    text=enriched,
                    section=section,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            pieces.append(" ".join(current))
            # carry the tail of the previous piece forward as overlap
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            current = tail
            current_len = tail_len
        current.append(sentence)
        current_len += len(sentence)

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(blocks: list[Block], doc_id: str, strategy: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if strategy == "baseline":
        return chunk_baseline(blocks, doc_id, chunk_size=chunk_size)
    return chunk_improved(blocks, doc_id, chunk_size=chunk_size, overlap=overlap)
