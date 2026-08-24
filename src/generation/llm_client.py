"""Chat LLM wrapper. Azure OpenAI in production (BACKEND=azure); a small
extractive fallback when no LLM is configured, so the pipeline is still fully
runnable end-to-end (retrieval, ACL, versioning, eval harness) with zero cloud
credentials. The fallback is clearly not a substitute for real generation —
see README for what it can and can't do — but it means "git clone && pip
install && python scripts/ingest.py && streamlit run app.py" works with no
setup at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import AzureConfig, GitHubModelsConfig


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    def is_available(self) -> bool:
        raise NotImplementedError

    def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        raise NotImplementedError

    def last_usage(self) -> TokenUsage:
        raise NotImplementedError


class AzureOpenAIChatClient(LLMClient):
    def __init__(self, cfg: AzureConfig):
        from openai import AzureOpenAI

        if not cfg.is_openai_configured():
            raise RuntimeError("Azure OpenAI is not configured.")
        self.client = AzureOpenAI(
            azure_endpoint=cfg.openai_endpoint,
            api_key=cfg.openai_api_key,
            api_version=cfg.openai_api_version,
        )
        self.deployment = cfg.chat_deployment
        self._usage = TokenUsage()

    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        usage = resp.usage
        self._usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return resp.choices[0].message.content or ""

    def last_usage(self) -> TokenUsage:
        return self._usage


class GitHubModelsChatClient(LLMClient):
    """Free-tier chat client via GitHub Models (OpenAI-API-compatible endpoint,
    authenticated with a GitHub personal access token). No Azure subscription
    or credit card needed — rate-limited, so fine for a demo/eval run, not for
    production traffic. See GitHubModelsConfig for how to fix the model id/base
    URL if GitHub has changed them since this was written."""

    def __init__(self, cfg: GitHubModelsConfig):
        from openai import OpenAI

        if not cfg.is_configured():
            raise RuntimeError("GitHub Models is not configured.")
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.token)
        self.model = cfg.chat_model
        self._usage = TokenUsage()

    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        usage = resp.usage
        self._usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return resp.choices[0].message.content or ""

    def last_usage(self) -> TokenUsage:
        return self._usage


class ExtractiveFallbackClient(LLMClient):
    """No LLM configured: cannot synthesize prose, so it does not try to. It
    is used by generation/prompt.py only to signal "no LLM" — the pipeline's
    answer in that mode is built directly from the retrieved chunks (see
    generation/generator.py: extractive_answer)."""

    def is_available(self) -> bool:
        return False

    def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        raise RuntimeError("No LLM configured; use the extractive fallback path instead.")

    def last_usage(self) -> TokenUsage:
        return TokenUsage()


def get_llm_client(settings) -> LLMClient:
    if settings.backend == "azure" and settings.azure.is_openai_configured():
        return AzureOpenAIChatClient(settings.azure)
    if settings.backend == "github" and settings.github.is_configured():
        return GitHubModelsChatClient(settings.github)
    return ExtractiveFallbackClient()
