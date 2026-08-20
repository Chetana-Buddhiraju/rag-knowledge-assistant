"""Loads the curated document metadata catalog (data/catalog.json).

In a real production ingestion pipeline this metadata (department, effective date,
version, supersedes-chain) would come from the document management system /
SharePoint columns / a front-matter convention, not be hand-maintained. For this
assignment's fixed 11-document knowledge base, a small curated JSON file is the
simplest thing that correctly captures department ACLs and version chains —
see docs/ARCHITECTURE.md for how this generalizes to real ingestion at scale.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import CATALOG_PATH, RAW_DIR


def load_catalog() -> dict[str, dict]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_source_files() -> list[Path]:
    catalog = load_catalog()
    return [RAW_DIR / rel for rel in catalog.keys()]


def metadata_for(rel_path: str) -> dict:
    catalog = load_catalog()
    if rel_path not in catalog:
        raise KeyError(f"{rel_path} is not registered in data/catalog.json")
    return catalog[rel_path]


def latest_version_by_family(catalog: dict[str, dict] | None = None) -> dict[str, str]:
    """doc_family -> doc_id of the highest-effective_date member of that family.

    This is the lookup that fixes Scenario 3 (conflicting/superseded documents):
    at query time we know which doc_id is "current" for a family like
    'orbitsuite-pricing' without needing the LLM to guess from filenames.
    """
    catalog = catalog or load_catalog()
    families: dict[str, list[dict]] = {}
    for meta in catalog.values():
        families.setdefault(meta["doc_family"], []).append(meta)

    latest: dict[str, str] = {}
    for family, members in families.items():
        dated = [m for m in members if m.get("effective_date")]
        pool = dated or members
        best = max(pool, key=lambda m: m.get("effective_date") or "")
        latest[family] = best["doc_id"]
    return latest
