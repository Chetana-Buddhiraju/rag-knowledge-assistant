"""Build the retrieval index for a given (backend, profile) combination.

Usage:
    python scripts/ingest.py --profile improved   # BACKEND from .env / env var
    python scripts/ingest.py --profile baseline
    python scripts/ingest.py --profile improved --backend github   # free, no card
    BACKEND=azure python scripts/ingest.py --profile improved
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROFILES, Settings  # noqa: E402
from src.embeddings.factory import get_embedding_client  # noqa: E402
from src.ingestion.pipeline import build_records  # noqa: E402
from src.vectorstore.factory import get_vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="improved")
    parser.add_argument("--backend", choices=["local", "github", "azure"], default=None)
    args = parser.parse_args()

    settings = Settings()
    settings.profile_name = args.profile
    if args.backend:
        settings.backend = args.backend

    profile = settings.profile
    print(f"Ingesting with backend={settings.backend!r} profile={profile.name!r} "
          f"(chunk_size={profile.chunk_size}, overlap={profile.chunk_overlap})")

    t0 = time.time()
    records = build_records(profile)
    print(f"Parsed + chunked {len(records)} chunks in {time.time() - t0:.2f}s")

    embedder = get_embedding_client(settings)
    t0 = time.time()
    texts = [r.text for r in records]
    embeddings = embedder.embed(texts)
    print(f"Embedded {len(texts)} chunks in {time.time() - t0:.2f}s (dim={embeddings.shape[1] if len(embeddings) else 0})")

    store = get_vector_store(settings, embedding_dim=embeddings.shape[1])
    t0 = time.time()
    store.build([r.to_dict() for r in records], embeddings)
    print(f"Indexed into {settings.backend} store in {time.time() - t0:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
