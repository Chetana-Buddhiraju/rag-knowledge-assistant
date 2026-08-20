from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingClient(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
        ...
