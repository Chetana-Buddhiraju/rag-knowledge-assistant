"""Production vector store: Azure AI Search.

Index schema: one field per metadata column used for filtering/faceting
(department, doc_id, doc_family, effective_date, ...) plus a vector field
(HNSW, cosine) and a semantic configuration over title/section/text. A single
query call does hybrid retrieval (BM25 `search_text` + vector `vector_queries`
fused server-side) and, when `query_type=semantic`, reranks the fused
candidates with the managed semantic reranker (L2 cross-encoder) before
returning results with `@search.reranker_score`.

department_filter is passed as an OData `search.in(department, '...')` filter,
i.e. enforced server-side before results are returned — this is what makes it
usable as a real security-trimming boundary (see docs/ARCHITECTURE.md, Q4)
rather than an app-side post-filter that a bug could accidentally skip.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ..config import AzureConfig
from .base import SearchHit


class AzureAISearchStore:
    def __init__(self, cfg: AzureConfig, embedding_dim: int):
        if not cfg.is_search_configured():
            raise RuntimeError(
                "Azure AI Search is not configured. Set AZURE_SEARCH_ENDPOINT and "
                "AZURE_SEARCH_API_KEY in .env, or run with BACKEND=local."
            )
        self.cfg = cfg
        self.embedding_dim = embedding_dim
        credential = AzureKeyCredential(cfg.search_api_key)
        self.index_client = SearchIndexClient(cfg.search_endpoint, credential)
        self.search_client = SearchClient(cfg.search_endpoint, cfg.search_index_name, credential)

    def _index_definition(self) -> SearchIndex:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="doc_family", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="department", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="section", type=SearchFieldDataType.String),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, sortable=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="effective_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="effective_end", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="version", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="supersedes", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source_path", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dim,
                vector_search_profile_name="kb-vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="kb-hnsw")],
            profiles=[VectorSearchProfile(name="kb-vector-profile", algorithm_configuration_name="kb-hnsw")],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=self.cfg.search_semantic_config,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="text")],
                        keywords_fields=[SemanticField(field_name="section")],
                    ),
                )
            ]
        )

        return SearchIndex(
            name=self.cfg.search_index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

    def create_index_if_needed(self, recreate: bool = False) -> None:
        existing = [i.name for i in self.index_client.list_indexes()]
        if self.cfg.search_index_name in existing:
            if not recreate:
                return
            self.index_client.delete_index(self.cfg.search_index_name)
        self.index_client.create_index(self._index_definition())

    def build(self, records: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        self.create_index_if_needed(recreate=True)
        docs = []
        for record, vector in zip(records, embeddings):
            doc = dict(record)
            doc["embedding"] = vector.tolist()
            docs.append(doc)

        batch_size = 500
        for i in range(0, len(docs), batch_size):
            self.search_client.upload_documents(docs[i : i + batch_size])

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int,
        department_filter: list[str] | None = None,
        use_hybrid: bool = True,
        use_semantic_ranking: bool = True,
    ) -> list[SearchHit]:
        odata_filter = None
        if department_filter:
            odata_filter = "search.in(department, '{}', ',')".format(",".join(department_filter))

        vector_queries = [
            VectorizedQuery(vector=query_vector.tolist(), k_nearest_neighbors=top_k, fields="embedding")
        ]

        kwargs: dict[str, Any] = dict(
            search_text=query_text if use_hybrid else None,
            vector_queries=vector_queries,
            filter=odata_filter,
            top=top_k,
        )
        if use_semantic_ranking:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = self.cfg.search_semantic_config

        results = self.search_client.search(**kwargs)

        hits = []
        for r in results:
            score = r.get("@search.reranker_score") if use_semantic_ranking else r.get("@search.score", 0.0)
            record = {k: v for k, v in r.items() if not k.startswith("@search") and k != "embedding"}
            hits.append(SearchHit(record=record, score=float(score or 0.0)))
        return hits
