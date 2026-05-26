"""OpenAI-compatible chat completions adapter for memory-augmented turns."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from core.access_scope import AccessScope

router = APIRouter()


class ChatMessage(BaseModel):
    """Subset of OpenAI chat message shape needed by the adapter."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> dict[str, object]:
    """Run a memory-augmented turn and return an OpenAI ChatCompletion-shaped response."""
    del payload.stream, payload.temperature, payload.max_tokens
    user_message = _last_user_message(payload.messages)
    session_id = request.headers.get("X-Session-Id", "default-session")
    scope = AccessScope(
        application_id=request.headers.get("X-Application-Id", "default-application"),
        tenant_id=request.headers.get("X-Tenant-Id", "default-tenant"),
        user_id=request.headers.get("X-User-Id", "default-user"),
    )
    components = request.app.state.components
    result = await components.loop.run_turn(user_message, session_id, scope)
    return {
        "id": f"chatcmpl-{uuid4()}",
        "object": "chat.completion",
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.final_response},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _last_user_message(messages: list[ChatMessage]) -> str:
    """Return the final user message from an OpenAI chat request."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
