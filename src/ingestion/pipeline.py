"""End-to-end ingestion: raw files -> parsed blocks -> chunks -> index-ready records.

A "record" is the unit stored in the vector store (Azure AI Search document or
local store row): chunk text plus every metadata field retrieval/ACL/versioning
needs, flattened so it can be used as Azure AI Search filterable fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import RAW_DIR, PipelineProfile
from .catalog import load_catalog
from .chunking import chunk_document
from .parsers import parse_document


@dataclass
class Record:
    id: str
    doc_id: str
    doc_family: str
    title: str
    department: str
    doc_type: str
    section: str
    chunk_index: int
    text: str
    effective_date: str | None
    effective_end: str | None
    version: str
    supersedes: str | None
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_records(profile: PipelineProfile) -> list[Record]:
    catalog = load_catalog()
    records: list[Record] = []

    for rel_path, meta in catalog.items():
        full_path = RAW_DIR / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Catalog references missing file: {full_path}")

        blocks = parse_document(full_path)
        chunks = chunk_document(
            blocks,
            doc_id=meta["doc_id"],
            strategy=profile.chunk_strategy,
            chunk_size=profile.chunk_size,
            overlap=profile.chunk_overlap,
        )

        for chunk in chunks:
            records.append(
                Record(
                    id=chunk.chunk_id.replace("::", "__"),  # Azure Search keys must be URL-safe
                    doc_id=meta["doc_id"],
                    doc_family=meta["doc_family"],
                    title=meta["title"],
                    department=meta["department"],
                    doc_type=meta["doc_type"],
                    section=chunk.section,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    effective_date=meta.get("effective_date"),
                    effective_end=meta.get("effective_end"),
                    version=meta["version"],
                    supersedes=meta.get("supersedes"),
                    source_path=rel_path,
                )
            )

    return records
