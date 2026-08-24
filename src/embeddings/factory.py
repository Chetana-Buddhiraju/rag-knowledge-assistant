from __future__ import annotations

from ..config import Settings
from .azure_openai import AzureOpenAIEmbeddings
from .github_models import GitHubModelsEmbeddings
from .local import LocalHashingEmbeddings


def get_embedding_client(settings: Settings):
    if settings.backend == "azure":
        return AzureOpenAIEmbeddings(settings.azure)
    if settings.backend == "github":
        return GitHubModelsEmbeddings(settings.github)
    return LocalHashingEmbeddings()
