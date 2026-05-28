"""Model and embedding adapters for Agentic Memory."""

from __future__ import annotations

from config import Settings
from model.groq_client import DeterministicStructuredClient, GroqStructuredClient, StructuredLLMClient

__all__ = [
    "DeterministicStructuredClient",
    "GroqStructuredClient",
    "StructuredLLMClient",
    "create_llm_client",
    "actor_model_name",
    "critic_model_name",
]


def create_llm_client(settings: Settings) -> StructuredLLMClient:
    """Create the configured structured LLM client."""
    if settings.llm_provider == "deterministic":
        return DeterministicStructuredClient()
    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return GroqStructuredClient(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
        )
    if settings.llm_provider == "anthropic":
        from model.anthropic_client import AnthropicStructuredClient

        return AnthropicStructuredClient(
            api_key=settings.anthropic_api_key,
            actor_model=settings.anthropic_actor_model,
            critic_model=settings.anthropic_critic_model,
        )
    if settings.llm_provider == "openai":
        from model.openai_client import OpenAIStructuredClient

        return OpenAIStructuredClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            actor_model=settings.openai_actor_model,
            critic_model=settings.openai_critic_model,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def actor_model_name(settings: Settings) -> str:
    """Return the configured Actor model for the selected provider."""
    if settings.llm_provider == "anthropic":
        return settings.anthropic_actor_model
    if settings.llm_provider == "openai":
        return settings.openai_actor_model
    return settings.actor_model


def critic_model_name(settings: Settings) -> str:
    """Return the configured Critic model for the selected provider."""
    if settings.llm_provider == "anthropic":
        return settings.anthropic_critic_model
    if settings.llm_provider == "openai":
        return settings.openai_critic_model
    return settings.critic_model
