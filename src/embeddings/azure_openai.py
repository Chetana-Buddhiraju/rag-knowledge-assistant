from __future__ import annotations

import numpy as np
from openai import AzureOpenAI

from ..config import AzureConfig


class AzureOpenAIEmbeddings:
    """Production embedding client: Azure OpenAI `text-embedding-3-large` (3072-dim)
    by default. This is the path documented in docs/ARCHITECTURE.md and is what
    the Azure AI Search vector index is built against in azure mode."""

    def __init__(self, cfg: AzureConfig):
        if not cfg.is_openai_configured():
            raise RuntimeError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_API_KEY in .env, or run with BACKEND=local."
            )
        self.client = AzureOpenAI(
            azure_endpoint=cfg.openai_endpoint,
            api_key=cfg.openai_api_key,
            api_version=cfg.openai_api_version,
        )
        self.deployment = cfg.embedding_deployment
        self.dim = cfg.embedding_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Azure OpenAI embeddings endpoint accepts batches; keep batches modest
        # to stay under per-request token limits for long chunks.
        vectors: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.deployment, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
