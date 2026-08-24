from __future__ import annotations

from ..config import INDEX_DIR, Settings
from .azure_search import AzureAISearchStore
from .local_store import LocalVectorStore


def get_vector_store(settings: Settings, embedding_dim: int):
    if settings.backend == "azure":
        return AzureAISearchStore(settings.azure, embedding_dim)
    # "local" and "github" both use the local BM25+cosine hybrid store — they
    # only differ in which embedding/chat model produced the vectors, so each
    # gets its own index directory (different embedding dim/content, must not mix).
    index_dir = INDEX_DIR / f"{settings.backend}_{settings.profile_name}"
    return LocalVectorStore(index_dir)
