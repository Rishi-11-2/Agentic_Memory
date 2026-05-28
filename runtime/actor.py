"""Actor execution for the self-learning loop."""

from __future__ import annotations

import traceback
from time import perf_counter

from core.models import ActorLLMOutput, ActorResult, LLMMessage, MemoryContext, ToolInvocation
from model import StructuredLLMClient
from runtime.tools import ToolDefinition, ToolRegistry


class Actor:
    """Run the primary model, parse tool calls, and execute tools audibly."""

    def __init__(self, llm_client: StructuredLLMClient, model: str, tool_registry: ToolRegistry) -> None:
        """Create an Actor with a model-agnostic client and explicit tool registry."""
        self._llm_client = llm_client
        self._model = model
        self._tool_registry = tool_registry

    def available_tool_names(self) -> list[str]:
        """Return tool names available to this Actor."""
        return self._tool_registry.names()

    def available_tool_definitions(self) -> list[ToolDefinition]:
        """Return full tool definitions for prompt assembly."""
        return self._tool_registry.definitions()

    async def execute(self, prompt: str, memory_context: MemoryContext, available_tools: list[str]) -> ActorResult:
        """Execute the Actor phase and return final response plus real tool traces."""
        started = perf_counter()
        messages = [
            LLMMessage(role="system", content=self._system_prompt(memory_context, available_tools)),
            LLMMessage(role="user", content=prompt),
        ]
        llm_output = await self._llm_client.complete_json(
            messages=messages,
            response_model=ActorLLMOutput,
            model=self._model,
            temperature=0.2,
        )
        executed_tools = []
        for tool_call in llm_output.tool_calls:
            executed_tools.append(await self._execute_tool_safely(tool_call))
        return ActorResult(
            reasoning=llm_output.reasoning,
            tool_calls=executed_tools,
            final_response=llm_output.final_response,
            latency_ms=int((perf_counter() - started) * 1000),
        )

    async def _execute_tool_safely(self, invocation: ToolInvocation) -> ToolInvocation:
        """Execute a tool call and convert exceptions into critic-visible traces."""
        try:
            return await self._tool_registry.execute(invocation)
        except Exception as exc:
            invocation.success = False
            invocation.output_summary = str(exc)
            invocation.error_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6))
            invocation.critic_flagged = True
            return invocation

    def _system_prompt(self, memory_context: MemoryContext, available_tools: list[str]) -> str:
        """Build the Actor system prompt with memory context and JSON instructions."""
        tool_defs = "\n".join(
            f"- {definition.name}: {definition.description} input_schema={definition.input_schema}"
            for definition in self.available_tool_definitions()
            if definition.name in available_tools
        )
        return (
            "You are the Actor in a self-learning agentic loop. Use the injected memory context exactly as durable "
            "guidance. Return only valid JSON matching this schema: "
            '{"reasoning": "brief private rationale", "tool_calls": ['
            '{"tool_name": "calculator", "input_parameters": {"expression": "2+2"}, '
            '"input_summary": "2+2"}], "final_response": "user-facing answer"}. '
            "Do not expose the reasoning field in the final response. Available tools:\n"
            f"{tool_defs}\n\n"
            f"{memory_context.rendered_context}"
        )
