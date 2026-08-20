from __future__ import annotations

from ..config import INDEX_DIR, Settings
from .azure_search import AzureAISearchStore
from .local_store import LocalVectorStore


def get_vector_store(settings: Settings, embedding_dim: int):
    if settings.use_azure():
        return AzureAISearchStore(settings.azure, embedding_dim)
    index_dir = INDEX_DIR / f"local_{settings.profile_name}"
    return LocalVectorStore(index_dir)
