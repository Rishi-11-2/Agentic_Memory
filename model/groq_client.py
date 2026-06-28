"""Model-agnostic structured LLM client with a Groq implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from core.models import ActorLLMOutput, CriticEvaluation, LLMMessage, ToolInvocation
ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredLLMClient(Protocol):
    """Protocol for any provider that can return JSON matching a Pydantic schema."""

    async def complete_json(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
        model: str,
        temperature: float = 0.0,
    ) -> ModelT:
        """Complete chat messages and parse the response into a Pydantic model."""
        ...


class GroqStructuredClient:
    """Groq SDK adapter for Actor and Critic structured JSON calls."""

    def __init__(self, api_key: str, base_url: str | None = None, max_attempts: int = 3) -> None:
        """Create an async Groq client with retry settings."""
        from groq import AsyncGroq

        self._client = AsyncGroq(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )
        self._max_attempts = max_attempts

    async def complete_json(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
        model: str,
        temperature: float = 0.0,
    ) -> ModelT:
        """Call Groq with exponential backoff and strict Pydantic parsing."""
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
                parsed = _parse_structured_json(content, response_model)
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "structured_llm_retry",
                    extra={"attempt": attempt, "model": model},
                    exc_info=exc,
                )
                if attempt < self._max_attempts:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        raise RuntimeError(f"Groq structured completion failed after {self._max_attempts} attempts") from last_error


class DeterministicStructuredClient:
    """Offline deterministic client for demos and tests without external LLM calls."""

    async def complete_json(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
        model: str,
        temperature: float = 0.0,
    ) -> ModelT:
        """Return schema-valid Actor or Critic outputs from simple local rules."""
        del model, temperature
        prompt = _last_user_message(messages)
        if response_model is ActorLLMOutput:
            return cast(ModelT, self._actor_output(prompt))
        if response_model is CriticEvaluation:
            return cast(ModelT, self._critic_output(prompt))
        raise TypeError(f"Deterministic client cannot synthesize {response_model.__name__}")

    def _actor_output(self, prompt: str) -> ActorLLMOutput:
        """Create a deterministic actor result with calculator/search/new tool calls."""
        normalized = prompt.lower()
        calls: list[ToolInvocation] = []
        
        # 1. calculator
        expression = _extract_expression(prompt)
        if expression is not None:
            calls.append(
                ToolInvocation(
                    tool_name="calculator",
                    input_parameters={"expression": expression},
                    input_summary=expression,
                )
            )
            
        # 2. web_search
        if any(word in normalized for word in ("search web", "web search", "latest")):
            calls.append(
                ToolInvocation(
                    tool_name="web_search",
                    input_parameters={"query": prompt},
                    input_summary=prompt[:120],
                )
            )
            
        # 3. file_search
        if any(word in normalized for word in ("file", "folder", "directory", "glob")):
            calls.append(
                ToolInvocation(
                    tool_name="file_search",
                    input_parameters={"pattern": "*.py", "max_results": 20},
                    input_summary="*.py in root",
                )
            )
            
        # 4. document_search
        if any(word in normalized for word in ("search in files", "grep", "find in code", "content search")):
            calls.append(
                ToolInvocation(
                    tool_name="document_search",
                    input_parameters={"query": "import", "file_pattern": "*.py"},
                    input_summary="Search 'import' in *.py",
                )
            )
            
        # 5. memory_search
        if any(word in normalized for word in ("remember", "what do you know", "memory", "retrieve")):
            calls.append(
                ToolInvocation(
                    tool_name="memory_search",
                    input_parameters={"query": prompt, "top_k": 5},
                    input_summary=prompt[:120],
                )
            )
            
        # 6. python_executor
        if any(word in normalized for word in ("run", "execute", "python", "code")):
            calls.append(
                ToolInvocation(
                    tool_name="python_executor",
                    input_parameters={"code": "print('hello')", "timeout_seconds": 10},
                    input_summary="print('hello')",
                )
            )
            
        # 7. shell_executor
        if any(word in normalized for word in ("shell", "terminal", "command", "list files")):
            calls.append(
                ToolInvocation(
                    tool_name="shell_executor",
                    input_parameters={"command": "ls"},
                    input_summary="ls",
                )
            )

        return ActorLLMOutput(
            reasoning="Deterministic demo plan generated from prompt keywords.",
            tool_calls=calls,
            final_response="I handled the request using the available memory context and tools.",
        )

    def _critic_output(self, prompt: str) -> CriticEvaluation:
        """Create a deterministic critic result that can exercise consolidation.

        Preference mining is intentionally left to the MCP client agent or a real
        LLM Critic; this offline client only supplies stable scoring signals.
        """
        del prompt
        return CriticEvaluation(
            factual_accuracy=8.0,
            preference_adherence=8.0,
            tool_efficiency=8.0,
            hallucination_risk=8.0,
            workflow_quality=8.0,
            new_semantic_facts=[],
            save_workflow=True,
            failure_summary=None,
        )


def _parse_structured_json(content: str, response_model: type[ModelT]) -> ModelT:
    """Parse provider JSON, tolerating fenced or prefixed JSON objects."""
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    try:
        return response_model.model_validate_json(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        extracted = content[start : end + 1]
        parsed = json.loads(extracted)
        return response_model.model_validate(parsed)


def _last_user_message(messages: list[LLMMessage]) -> str:
    """Return the final user message content from a chat prompt."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _extract_expression(prompt: str) -> str | None:
    """Extract a conservative arithmetic expression from a natural-language prompt."""
    candidates: list[str] = re.findall(r"[0-9][0-9\s+\-*/().%^]+[0-9)]", prompt)
    if not candidates:
        return None
    expression: str = max(candidates, key=len).strip()
    return expression
