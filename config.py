"""Runtime configuration and structured logging for Agentic Memory."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON events for production ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON string containing standard and loop-specific fields."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "phase",
            "session_id",
            "latency_ms",
            "scope_hash",
            "attempt",
            "model",
            "input_tokens",
            "output_tokens",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class Settings(BaseSettings):
    """Validate all deployment settings before the FastAPI service starts."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Memory Backend ──────────────────────────────────────────────
    memory_backend: str = Field("sqlite", alias="MEMORY_BACKEND")
    sqlite_db_path: str = Field("agentic_memory.db", alias="SQLITE_DB_PATH")
    postgres_dsn: str = Field(
        "postgresql://postgres:postgres@localhost:5432/agentic_memory",
        alias="POSTGRES_DSN",
    )

    # ── LLM Provider ────────────────────────────────────────────────
    llm_provider: str = Field("deterministic", alias="LLM_PROVIDER")
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    groq_base_url: str | None = Field(default=None, alias="GROQ_BASE_URL")
    actor_model: str = Field("llama3-8b-8192", alias="ACTOR_MODEL")
    critic_model: str = Field("llama3-70b-8192", alias="CRITIC_MODEL")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    anthropic_actor_model: str = Field("claude-sonnet-4-5", alias="ANTHROPIC_ACTOR_MODEL")
    anthropic_critic_model: str = Field("claude-opus-4-5", alias="ANTHROPIC_CRITIC_MODEL")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_base_url: str = Field("", alias="OPENAI_BASE_URL")
    openai_actor_model: str = Field("gpt-4o-mini", alias="OPENAI_ACTOR_MODEL")
    openai_critic_model: str = Field("gpt-4o", alias="OPENAI_CRITIC_MODEL")

    # ── MCP ────────────────────────────────────────────────────────
    mcp_transport: str = Field("stdio", alias="MCP_TRANSPORT")
    mcp_http_port: int = Field(8001, ge=1, le=65535, alias="MCP_HTTP_PORT")

    # ── Reliability ────────────────────────────────────────────────
    critic_timeout_seconds: float = Field(8.0, gt=0.0, alias="CRITIC_TIMEOUT_SECONDS")
    critic_self_consistency_samples: int = Field(3, ge=1, le=9, alias="CRITIC_SELF_CONSISTENCY_SAMPLES")
    critic_self_consistency_temperature: float = Field(
        0.4, ge=0.0, le=2.0, alias="CRITIC_SELF_CONSISTENCY_TEMPERATURE"
    )

    # ── Security ───────────────────────────────────────────────────
    rate_limit_rpm: int = Field(60, ge=0, alias="RATE_LIMIT_RPM")
    debug_key: str = Field("", alias="DEBUG_KEY")

    # ── Brave Search ────────────────────────────────────────────────
    brave_search_api_key: SecretStr | None = Field(default=None, alias="BRAVE_SEARCH_API_KEY")
    brave_search_endpoint: str = Field(
        "https://api.search.brave.com/res/v1/web/search", alias="BRAVE_SEARCH_ENDPOINT"
    )
    brave_search_country: str = Field("us", alias="BRAVE_SEARCH_COUNTRY")
    brave_search_lang: str = Field("en", alias="BRAVE_SEARCH_LANG")
    brave_search_count: int = Field(5, ge=1, le=20, alias="BRAVE_SEARCH_COUNT")
    brave_search_timeout_seconds: float = Field(10.0, gt=0.0, le=60.0, alias="BRAVE_SEARCH_TIMEOUT_SECONDS")

    # ── Embeddings ──────────────────────────────────────────────────
    embedding_backend: str = Field("sentence-transformer", alias="EMBEDDING_BACKEND")
    embedding_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME")
    hash_embedding_dimensions: int = Field(256, alias="HASH_EMBEDDING_DIMENSIONS")

    # ── Memory Tuning ───────────────────────────────────────────────
    memory_window_turns: int = Field(10, ge=2, alias="MEMORY_WINDOW_TURNS")
    semantic_dedup_threshold: float = Field(0.92, ge=0.0, le=1.0, alias="SEMANTIC_DEDUP_THRESHOLD")
    semantic_memory_ttl_days: int = Field(180, ge=0, alias="SEMANTIC_MEMORY_TTL_DAYS")
    failure_similarity_threshold: float = Field(0.80, ge=0.0, le=1.0, alias="FAILURE_SIMILARITY_THRESHOLD")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── Tool Settings ───────────────────────────────────────────────
    tool_workspace_root: str = Field(".", alias="TOOL_WORKSPACE_ROOT")

    # ── Backend validators ──────────────────────────────────────────

    @field_validator("memory_backend")
    @classmethod
    def validate_memory_backend(cls, value: str) -> str:
        """Restrict memory backend names to supported implementations."""
        normalized = value.strip().lower()
        if normalized not in {"sqlite", "postgres"}:
            raise ValueError("MEMORY_BACKEND must be 'sqlite' or 'postgres'")
        return normalized

    @field_validator("embedding_backend")
    @classmethod
    def validate_embedding_backend(cls, value: str) -> str:
        """Restrict embedding backend names to the supported explicit implementations."""
        normalized = value.strip().lower()
        if normalized not in {"sentence-transformer", "hash"}:
            raise ValueError("EMBEDDING_BACKEND must be 'sentence-transformer' or 'hash'")
        return normalized

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        """Restrict LLM providers to explicit structured-client adapters."""
        normalized = value.strip().lower()
        if normalized not in {"groq", "anthropic", "openai", "deterministic"}:
            raise ValueError("LLM_PROVIDER must be 'groq', 'anthropic', 'openai', or 'deterministic'")
        return normalized

    @field_validator("mcp_transport")
    @classmethod
    def validate_mcp_transport(cls, value: str) -> str:
        """Restrict MCP transport names to supported FastMCP transports."""
        normalized = value.strip().lower()
        if normalized not in {"stdio", "http"}:
            raise ValueError("MCP_TRANSPORT must be 'stdio' or 'http'")
        return normalized

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def empty_groq_key_to_none(cls, value: object) -> object:
        """Treat an empty optional Groq key as unset."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("groq_base_url")
    @classmethod
    def empty_base_url_to_none(cls, value: str | None) -> str | None:
        """Treat an empty optional Groq base URL as unset."""
        if value is None or value.strip() == "":
            return None
        return value.rstrip("/")

    @field_validator("openai_base_url")
    @classmethod
    def clean_openai_base_url(cls, value: str) -> str:
        """Normalize the optional OpenAI-compatible base URL."""
        return value.strip().rstrip("/")

    @field_validator("brave_search_api_key", mode="before")
    @classmethod
    def empty_brave_key_to_none(cls, value: object) -> object:
        """Treat an empty optional Brave API key as unset."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("brave_search_endpoint")
    @classmethod
    def validate_brave_endpoint(cls, value: str) -> str:
        """Ensure the Brave endpoint is an HTTP URL before tool registration."""
        if not value.startswith(("http://", "https://")):
            raise ValueError("BRAVE_SEARCH_ENDPOINT must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_backend_dependencies(self) -> "Settings":
        """Cross-validate that required credentials are set for the chosen backends."""
        if self.llm_provider == "groq":
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        return self


def configure_logging(level: str) -> None:
    """Install JSON logging for every loop phase and provider retry event."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def load_settings() -> Settings:
    """Load settings after applying the JSON logger so startup errors are clear."""
    settings = Settings()  # type: ignore[call-arg]
    configure_logging(settings.log_level)
    return settings
