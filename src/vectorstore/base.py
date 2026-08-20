from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class SearchHit:
    record: dict[str, Any]
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0


class VectorStore(Protocol):
    def build(self, records: list[dict[str, Any]], embeddings: np.ndarray) -> None: ...

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int,
        department_filter: list[str] | None = None,
        use_hybrid: bool = True,
    ) -> list[SearchHit]: ...
