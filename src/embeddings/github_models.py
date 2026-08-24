from __future__ import annotations

import numpy as np
from openai import OpenAI

from ..config import GitHubModelsConfig


class GitHubModelsEmbeddings:
    """Free-tier embedding client via GitHub Models (OpenAI-API-compatible).

    Same interface as AzureOpenAIEmbeddings — this exists so the "github" backend
    gets real semantic embeddings (unlike the "local" hashed fallback) without
    needing an Azure subscription. Rate limits are low (fine for this KB's ~150
    chunks and a ~30-question eval run, not for production traffic).
    """

    def __init__(self, cfg: GitHubModelsConfig):
        if not cfg.is_configured():
            raise RuntimeError(
                "GitHub Models is not configured. Set GITHUB_MODELS_TOKEN (a GitHub "
                "personal access token) in .env, or run with BACKEND=local."
            )
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.token)
        self.model = cfg.embedding_model
        self.dim = cfg.embedding_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
