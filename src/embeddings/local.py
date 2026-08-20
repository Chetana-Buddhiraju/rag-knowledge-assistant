from __future__ import annotations

import re
import zlib

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalHashingEmbeddings:
    """Dependency-free fallback embedding for local/offline development and for
    running the evaluation harness without cloud credentials.

    This is NOT the production embedding model — it's a deterministic hashed
    bag-of-words/char-n-gram vectorizer (word 1-2 grams + char 3-grams, hashed
    into a fixed-size vector, log-tf weighted, L2-normalized). It captures
    enough lexical overlap for hybrid search + the eval harness to exercise the
    full retrieval pipeline (chunking, metadata filtering, reranking,
    conversation rewriting) without any network call. Swap BACKEND=azure to use
    real Azure OpenAI embeddings for production quality semantic recall.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vectorize_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())

        grams: list[str] = []
        grams.extend(tokens)
        grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        joined = " ".join(tokens)
        grams.extend(joined[i : i + 3] for i in range(0, max(len(joined) - 2, 0)))

        for g in grams:
            h = zlib.crc32(g.encode("utf-8")) % self.dim
            vec[h] += 1.0

        vec = np.log1p(vec)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vectorize_one(t) for t in texts])
