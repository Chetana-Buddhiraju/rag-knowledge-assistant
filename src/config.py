"""Central configuration, loaded from environment variables (.env).

Two independent axes of configuration:
  1. Backend selection (BACKEND=local|github|azure) — which vector store / embeddings / LLM
     to use. "azure" is the documented production path (Azure OpenAI + Azure AI Search).
     "github" is a free, no-credit-card fallback (GitHub Models for real chat + embeddings,
     paired with the same local BM25+cosine hybrid search "local" uses, since GitHub Models
     doesn't include a managed search service). "local" is fully offline (no network calls
     at all) for zero-dependency development and the evaluation harness.
  2. Pipeline profile (baseline|improved) — which retrieval/generation behavior to use.
     The profile is what Step 3/Step 4 of the assignment is about: the same backends,
     two different pipeline configurations, evaluated against each other.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
CATALOG_PATH = DATA_DIR / "catalog.json"


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AzureConfig:
    openai_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    openai_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    chat_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"))
    embedding_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("AZURE_OPENAI_EMBEDDING_DIM", "3072")))

    search_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_ENDPOINT", ""))
    search_api_key: str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_API_KEY", ""))
    search_index_name: str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_INDEX_NAME", "kb-chunks"))
    search_semantic_config: str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "kb-semantic-config"))

    def is_openai_configured(self) -> bool:
        return bool(self.openai_endpoint and self.openai_api_key)

    def is_search_configured(self) -> bool:
        return bool(self.search_endpoint and self.search_api_key)


@dataclass
class GitHubModelsConfig:
    """GitHub Models: free, rate-limited access to hosted models (GPT-4o family,
    embedding models, etc.) via an OpenAI-API-compatible endpoint, authenticated
    with a GitHub personal access token — no Azure subscription or card needed.

    Model IDs and the exact base URL occasionally shift as GitHub evolves this
    product; if the defaults below 404, open the model's page at
    https://github.com/marketplace/models, click "Use this model", and copy the
    exact base_url/model id shown there into your .env.
    """

    token: str = field(default_factory=lambda: os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN", ""))
    base_url: str = field(default_factory=lambda: os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"))
    chat_model: str = field(default_factory=lambda: os.getenv("GITHUB_MODELS_CHAT_MODEL", "openai/gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("GITHUB_MODELS_EMBEDDING_MODEL", "openai/text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("GITHUB_MODELS_EMBEDDING_DIM", "1536")))

    def is_configured(self) -> bool:
        return bool(self.token)


@dataclass
class PipelineProfile:
    """A named bundle of retrieval/generation knobs. `baseline` reproduces the naive
    RAG failure modes from the assignment; `improved` is the fixed version."""

    name: str
    chunk_strategy: str = "improved"        # "baseline" | "improved"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k_retrieve: int = 12
    top_k_final: int = 5
    use_hybrid_search: bool = True
    use_reranking: bool = True
    use_query_rewrite: bool = True
    use_department_acl: bool = True
    use_version_resolution: bool = True
    use_multi_query_expansion: bool = True
    ambiguity_detection: bool = True
    min_confidence: float = 0.22            # below this, answer "insufficient evidence"
    conversation_window: int = 4            # number of prior turns folded into rewrite


BASELINE_PROFILE = PipelineProfile(
    name="baseline",
    chunk_strategy="baseline",
    chunk_size=300,
    chunk_overlap=0,
    top_k_retrieve=3,
    top_k_final=3,
    use_hybrid_search=False,
    use_reranking=False,
    use_query_rewrite=False,
    use_department_acl=False,
    use_version_resolution=False,
    use_multi_query_expansion=False,
    ambiguity_detection=False,
    min_confidence=0.0,
    conversation_window=0,
)

IMPROVED_PROFILE = PipelineProfile(name="improved")

PROFILES = {"baseline": BASELINE_PROFILE, "improved": IMPROVED_PROFILE}


@dataclass
class Settings:
    backend: str = field(default_factory=lambda: os.getenv("BACKEND", "local").lower())  # "local" | "github" | "azure"
    profile_name: str = field(default_factory=lambda: os.getenv("PROFILE", "improved").lower())
    azure: AzureConfig = field(default_factory=AzureConfig)
    github: GitHubModelsConfig = field(default_factory=GitHubModelsConfig)
    debug: bool = field(default_factory=lambda: _bool("DEBUG", False))

    @property
    def profile(self) -> PipelineProfile:
        return PROFILES[self.profile_name]

    def use_azure(self) -> bool:
        """True only for the full production stack (Azure OpenAI + Azure AI Search).
        Both "local" and "github" backends use the local BM25+cosine hybrid vector
        store and the local lexical reranker/confidence heuristic — they only differ
        in whether generation/embeddings hit a real model or a local fallback."""
        return self.backend == "azure"


# Departments recognized by the system (== raw data subfolders == ACL scopes).
DEPARTMENTS = ["Finance", "HR", "IT", "Legal", "Sales"]

# Role -> allowed departments. "admin" sees everything. This is the enforcement
# table for Scenario/Question 4 (access-controlled RAG).
ROLE_DEPARTMENT_ACCESS = {
    "admin": DEPARTMENTS,
    "finance": ["Finance"],
    "hr": ["HR"],
    "it": ["IT"],
    "legal": ["Legal"],
    "sales": ["Sales"],
    "engineering": [],  # matches the assignment's example: no department maps to Engineering, so it sees nothing
}

settings = Settings()
