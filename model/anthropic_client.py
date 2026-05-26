"""Anthropic structured LLM client for Actor-Critic JSON calls."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from core.models import LLMMessage
ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)


class AnthropicStructuredClient:
    """Anthropic SDK adapter for structured JSON completions."""

    def __init__(
        self,
        actor_model: str = "claude-sonnet-4-5",
        critic_model: str = "claude-opus-4-5",
        api_key: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        """Create an Anthropic async client using ANTHROPIC_API_KEY from the environment."""
        from anthropic import AsyncAnthropic

        self.actor_model = actor_model
        self.critic_model = critic_model
        self._client = AsyncAnthropic(
            api_key=api_key,
        )
        self._max_attempts = max_attempts

    async def complete_json(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
        model: str,
        temperature: float = 0.0,
    ) -> ModelT:
        """Complete chat messages and parse fenced-or-plain JSON into a Pydantic model."""
        from anthropic import RateLimitError

        system_prompt = "\n\n".join(message.content for message in messages if message.role == "system")
        chat_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        ]
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.messages.create(
                    model=model,
                    system=cast(Any, system_prompt) if system_prompt else cast(Any, None),
                    messages=cast(Any, chat_messages),
                    max_tokens=4096,
                    temperature=temperature,
                )
                text = "".join(getattr(block, "text", "") for block in response.content)
                logger.debug(
                    "anthropic_structured_completion",
                    extra={
                        "model": model,
                        "input_tokens": getattr(response.usage, "input_tokens", 0),
                        "output_tokens": getattr(response.usage, "output_tokens", 0),
                    },
                )
                parsed = _parse_structured_json(text, response_model)
                return parsed
            except RateLimitError as exc:
                last_error = exc
                logger.warning("anthropic_rate_limit_retry", extra={"attempt": attempt, "model": model}, exc_info=exc)
                if attempt < self._max_attempts:
                    await asyncio.sleep(2.0)
            except Exception as exc:
                raise RuntimeError("Anthropic structured completion failed") from exc
        raise RuntimeError(f"Anthropic structured completion failed after {self._max_attempts} attempts") from last_error


def _parse_structured_json(content: str, response_model: type[ModelT]) -> ModelT:
    """Strip markdown fences and parse provider JSON into the requested schema."""
    cleaned = re.sub(r"```(?:json)?|```", "", content).strip()
    try:
        return response_model.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return response_model.model_validate(json.loads(cleaned[start : end + 1]))
