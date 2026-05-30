"""OpenAI-compatible structured LLM client for Actor-Critic JSON calls."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from core.models import LLMMessage
ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)


class OpenAIStructuredClient:
    """OpenAI SDK adapter for OpenAI and OpenAI-compatible JSON completions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        """Create an OpenAI async client using OPENAI_API_KEY and optional OPENAI_BASE_URL."""
        from openai import AsyncOpenAI

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=resolved_base_url if resolved_base_url else None,
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
        from openai import RateLimitError

        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=cast(Any, [message.model_dump() for message in messages]),
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                usage = response.usage
                logger.debug(
                    "openai_structured_completion",
                    extra={
                        "model": model,
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                    },
                )
                parsed = _parse_structured_json(content, response_model)
                return parsed
            except RateLimitError as exc:
                last_error = exc
                logger.warning("openai_rate_limit_retry", extra={"attempt": attempt, "model": model}, exc_info=exc)
                if attempt < self._max_attempts:
                    await asyncio.sleep(2.0)
            except Exception as exc:
                raise RuntimeError("OpenAI structured completion failed") from exc
        raise RuntimeError(f"OpenAI structured completion failed after {self._max_attempts} attempts") from last_error


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
