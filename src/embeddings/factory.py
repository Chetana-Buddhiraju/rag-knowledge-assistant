from __future__ import annotations

from ..config import Settings
from .azure_openai import AzureOpenAIEmbeddings
from .local import LocalHashingEmbeddings


def get_embedding_client(settings: Settings):
    if settings.use_azure():
        return AzureOpenAIEmbeddings(settings.azure)
    return LocalHashingEmbeddings()
