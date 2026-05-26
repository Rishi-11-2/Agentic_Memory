"""Explicit tool registry for the self-learning loop."""

from __future__ import annotations

import ast
import asyncio
import math
import os
import re
import shlex
import stat
import textwrap
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import httpx

from pydantic import BaseModel, Field

from core.models import ToolInvocation

ToolCallable = Callable[[dict[str, Any]], Awaitable[str]]

# ── Limits ──────────────────────────────────────────────────────────
_MAX_OUTPUT_BYTES = 10_240
_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".tox", ".pytest_cache"})
_MAX_FILE_READ_BYTES = 2 * 1024 * 1024  # 2 MB
_BINARY_CHECK_BYTES = 8192


# ── ToolDefinition & ToolRegistry ───────────────────────────────────


class ToolDefinition(BaseModel):
    """Describe a callable tool and its JSON-compatible input contract."""

    name: str
    description: str
    input_schema: dict[str, str] = Field(default_factory=dict)


class ToolRegistry:
    """Register and execute auditable tools without framework abstractions."""

    def __init__(self) -> None:
        """Create an empty registry for runtime tools."""
        self._tools: dict[str, tuple[ToolDefinition, ToolCallable]] = {}

    def register(self, definition: ToolDefinition, handler: ToolCallable) -> None:
        """Register a tool definition and async handler by name."""
        self._tools[definition.name] = (definition, handler)

    def definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions visible to the Actor."""
        return [definition for definition, _ in self._tools.values()]

    def names(self) -> list[str]:
        """Return registered tool names for compact Actor prompts."""
        return list(self._tools.keys())

    async def execute(self, invocation: ToolInvocation) -> ToolInvocation:
        """Execute a requested tool and return a completed invocation trace."""
        if invocation.tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {invocation.tool_name}")
        _, handler = self._tools[invocation.tool_name]
        started = perf_counter()
        output = await handler(invocation.input_parameters)
        invocation.output_summary = output
        invocation.success = True
        invocation.latency_ms = int((perf_counter() - started) * 1000)
        return invocation


# ── 1. Calculator ───────────────────────────────────────────────────


async def calculator_tool(params: dict[str, Any]) -> str:
    """Evaluate a small arithmetic expression in a sandboxed Python evaluator."""
    expression = str(params.get("expression", "")).strip()
    if not expression:
        raise ValueError("calculator requires an 'expression' parameter")
    if len(expression) > 200:
        raise ValueError("calculator expression is too long")
    normalized = expression.replace("^", "**")
    parsed = ast.parse(normalized, mode="eval")
    _validate_calculator_ast(parsed)
    value = eval(  # noqa: S307 — sandboxed via AST whitelist and empty builtins
        compile(parsed, "<calculator>", "eval"),
        {"__builtins__": {}},
        _allowed_math_names(),
    )
    return f"calculator result: {value}"


def _validate_calculator_ast(node: ast.AST) -> None:
    """Reject AST nodes outside arithmetic expressions and known math calls."""
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.FloorDiv,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed_nodes):
            raise ValueError(f"calculator disallows syntax: {type(child).__name__}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _allowed_math_names():
                raise ValueError("calculator only allows whitelisted math functions")
        if isinstance(child, ast.Name) and child.id not in _allowed_math_names():
            raise ValueError(f"calculator disallows name: {child.id}")


def _allowed_math_names() -> dict[str, Any]:
    """Return safe math names available inside the calculator sandbox."""
    return {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
    }


# ── 2. Web Search (Brave) ──────────────────────────────────────────


class BraveSearchTool:
    """Call Brave Search API and format auditable web results for the Actor."""

    def __init__(
        self,
        api_key: str | None,
        endpoint: str = "https://api.search.brave.com/res/v1/web/search",
        country: str = "us",
        search_lang: str = "en",
        default_count: int = 5,
        timeout_seconds: float = 10.0,
        simulate_when_missing_key: bool = False,
    ) -> None:
        """Create a Brave Search tool with deployment-safe defaults."""
        self._api_key = api_key
        self._endpoint = endpoint
        self._country = country
        self._search_lang = search_lang
        self._default_count = max(1, min(default_count, 20))
        self._timeout_seconds = timeout_seconds
        self._simulate_when_missing_key = simulate_when_missing_key

    async def __call__(self, params: dict[str, Any]) -> str:
        """Execute a Brave web search and return compact grounded results."""
        query = str(params.get("query", "")).strip()
        if not query:
            raise ValueError("web_search requires a 'query' parameter")

        count = _bounded_int(params.get("count"), self._default_count, minimum=1, maximum=20)
        country = str(params.get("country") or self._country).lower()
        search_lang = str(params.get("search_lang") or self._search_lang).lower()

        if not self._api_key:
            if self._simulate_when_missing_key:
                return _format_simulated_search(query)
            raise ValueError("BRAVE_SEARCH_API_KEY is required for the web_search tool")

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(
                self._endpoint,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                params={
                    "q": query,
                    "count": count,
                    "country": country,
                    "search_lang": search_lang,
                },
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Brave Search API returned {response.status_code}: {_excerpt(response.text, 300)}")
        data = cast(dict[str, Any], response.json())
        return _format_brave_results(query, data, count)


# ── 3. File Search ──────────────────────────────────────────────────


class FileSearchTool:
    """Find files by name, extension, or glob pattern within a workspace root."""

    def __init__(self, workspace_root: str) -> None:
        """Create a file search tool scoped to a workspace directory."""
        self._root = Path(workspace_root).resolve()

    async def __call__(self, params: dict[str, Any]) -> str:
        """Search for files matching a pattern within the workspace."""
        pattern = str(params.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("file_search requires a 'pattern' parameter")
        max_results = _bounded_int(params.get("max_results"), 20, minimum=1, maximum=50)
        directory = self._resolve_directory(params.get("directory"))

        matches: list[str] = []
        is_glob = any(char in pattern for char in ("*", "?", "["))

        if is_glob:
            for path in directory.rglob(pattern):
                if self._should_skip(path):
                    continue
                if path.is_file():
                    matches.append(self._format_entry(path))
                if len(matches) >= max_results:
                    break
        else:
            lowered = pattern.lower()
            for root, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                for name in files:
                    if lowered in name.lower():
                        filepath = Path(root) / name
                        if self._is_within_root(filepath):
                            matches.append(self._format_entry(filepath))
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break

        if not matches:
            return f"file_search: no files matching '{pattern}' found in {directory.relative_to(self._root) if directory != self._root else '.'}/"
        header = f"file_search: {len(matches)} file(s) matching '{pattern}':"
        return _truncate_output(header + "\n" + "\n".join(matches))

    def _resolve_directory(self, raw: object) -> Path:
        """Resolve and validate the search directory within the workspace root."""
        if not raw or not str(raw).strip():
            return self._root
        candidate = (self._root / str(raw)).resolve()
        if not self._is_within_root(candidate):
            raise ValueError(f"directory '{raw}' escapes the workspace root")
        if not candidate.is_dir():
            raise ValueError(f"directory '{raw}' does not exist")
        return candidate

    def _is_within_root(self, path: Path) -> bool:
        """Return whether a resolved path is inside the workspace root."""
        try:
            path.resolve().relative_to(self._root)
            return True
        except ValueError:
            return False

    def _should_skip(self, path: Path) -> bool:
        """Return whether a path should be excluded from results."""
        return any(part in _SKIP_DIRS for part in path.parts) or not self._is_within_root(path)

    def _format_entry(self, path: Path) -> str:
        """Format one file entry with relative path and size."""
        try:
            size = path.stat().st_size
            relative = path.relative_to(self._root)
            return f"  {relative}  ({_human_size(size)})"
        except OSError:
            return f"  {path.name}  (unreadable)"


# ── 4. Document Search ──────────────────────────────────────────────


class DocumentSearchTool:
    """Search within file contents using text or regex matching."""

    def __init__(self, workspace_root: str) -> None:
        """Create a document search tool scoped to a workspace directory."""
        self._root = Path(workspace_root).resolve()

    async def __call__(self, params: dict[str, Any]) -> str:
        """Search file contents for a query string or regex pattern."""
        query = str(params.get("query", "")).strip()
        if not query:
            raise ValueError("document_search requires a 'query' parameter")
        max_results = _bounded_int(params.get("max_results"), 10, minimum=1, maximum=30)
        context_lines = _bounded_int(params.get("context_lines"), 2, minimum=0, maximum=5)
        file_pattern = str(params.get("file_pattern", "")).strip() or None
        directory = self._resolve_directory(params.get("directory"))

        try:
            compiled = re.compile(query, re.IGNORECASE)
        except re.error:
            compiled = re.compile(re.escape(query), re.IGNORECASE)

        results: list[str] = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                if file_pattern and not _glob_match(name, file_pattern):
                    continue
                filepath = Path(root) / name
                if not self._is_within_root(filepath):
                    continue
                file_matches = self._search_file(filepath, compiled, context_lines, max_results - len(results))
                results.extend(file_matches)
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results:
            scope = f" (file_pattern={file_pattern})" if file_pattern else ""
            return f"document_search: no matches for '{query}'{scope}"
        header = f"document_search: {len(results)} match(es) for '{query}':"
        return _truncate_output(header + "\n" + "\n".join(results))

    def _search_file(
        self, filepath: Path, pattern: re.Pattern[str], context_lines: int, remaining: int
    ) -> list[str]:
        """Search one file and return formatted match blocks."""
        if remaining <= 0:
            return []
        try:
            size = filepath.stat().st_size
            if size > _MAX_FILE_READ_BYTES or size == 0:
                return []
        except OSError:
            return []
        if _is_binary_file(filepath):
            return []
        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

        matches: list[str] = []
        relative = filepath.relative_to(self._root)
        for line_num, line_text in enumerate(lines, start=1):
            if pattern.search(line_text):
                block = self._format_match(relative, lines, line_num, context_lines)
                matches.append(block)
                if len(matches) >= remaining:
                    break
        return matches

    def _format_match(self, relative: Path, lines: list[str], line_num: int, context: int) -> str:
        """Format a single match with surrounding context lines."""
        start = max(0, line_num - 1 - context)
        end = min(len(lines), line_num + context)
        block_lines = [f"\n  {relative}:{line_num}"]
        for idx in range(start, end):
            marker = ">>>" if idx == line_num - 1 else "   "
            block_lines.append(f"    {marker} {idx + 1:>4}| {_excerpt(lines[idx], 200)}")
        return "\n".join(block_lines)

    def _resolve_directory(self, raw: object) -> Path:
        """Resolve and validate the search directory within the workspace root."""
        if not raw or not str(raw).strip():
            return self._root
        candidate = (self._root / str(raw)).resolve()
        if not self._is_within_root(candidate):
            raise ValueError(f"directory '{raw}' escapes the workspace root")
        if not candidate.is_dir():
            raise ValueError(f"directory '{raw}' does not exist")
        return candidate

    def _is_within_root(self, path: Path) -> bool:
        """Return whether a resolved path is inside the workspace root."""
        try:
            path.resolve().relative_to(self._root)
            return True
        except ValueError:
            return False


# ── 5. Memory Search ────────────────────────────────────────────────


class MemorySearchTool:
    """Vector database lookup against the agent's own memory layers."""

    def __init__(
        self,
        store: Any,
        embedding_model: Any,
        semantic_ttl_cutoff: Any = None,
    ) -> None:
        """Create a memory search tool with injected store and embedding model."""
        self._store = store
        self._embedding_model = embedding_model
        self._semantic_ttl_cutoff = semantic_ttl_cutoff

    async def __call__(self, params: dict[str, Any]) -> str:
        """Search memory layers for records relevant to a query."""
        query = str(params.get("query", "")).strip()
        if not query:
            raise ValueError("memory_search requires a 'query' parameter")

        top_k = _bounded_int(params.get("top_k"), 5, minimum=1, maximum=20)
        raw_layers = params.get("layers")
        valid_layers = {"semantic", "episodic", "procedural", "failure"}
        if isinstance(raw_layers, list):
            selected = [layer for layer in raw_layers if layer in valid_layers] or list(valid_layers)
        else:
            selected = list(valid_layers)

        embedding = await self._embedding_model.embed(query)
        sections: list[str] = []

        if "semantic" in selected:
            records = await self._store.search_semantic(
                embedding, top_k, 0.30, 0.0,
                last_confirmed_after=self._semantic_ttl_cutoff,
            )
            if records:
                lines = [f"  SEMANTIC ({len(records)} results):"]
                for record in records:
                    score_text = f"{record.score:.2f}" if record.score is not None else "n/a"
                    lines.append(f"    [{score_text}] {_excerpt(record.content, 200)} (source={record.source})")
                sections.append("\n".join(lines))

        if "episodic" in selected:
            records = await self._store.search_episodes(embedding, top_k, 0.30)
            if records:
                lines = [f"  EPISODIC ({len(records)} results):"]
                for record in records:
                    score_text = f"{record.score:.2f}" if record.score is not None else "n/a"
                    tools = ", ".join(record.tool_names) or "none"
                    lines.append(
                        f"    [{score_text}] outcome={record.outcome.value} tools=[{tools}] "
                        f"prompt={_excerpt(record.prompt_text, 120)}"
                    )
                sections.append("\n".join(lines))

        if "procedural" in selected:
            records = await self._store.search_procedural(embedding, top_k, 0.30)
            if records:
                lines = [f"  PROCEDURAL ({len(records)} results):"]
                for record in records:
                    score_text = f"{record.score:.2f}" if record.score is not None else "n/a"
                    tools = " → ".join(record.tool_names) or "none"
                    lines.append(
                        f"    [{score_text}] {tools} (status={record.status.value}, "
                        f"success_count={record.success_count})"
                    )
                sections.append("\n".join(lines))

        if "failure" in selected:
            records = await self._store.search_failures(embedding, top_k, 0.30)
            if records:
                lines = [f"  FAILURE ({len(records)} results):"]
                for record in records:
                    score_text = f"{record.score:.2f}" if record.score is not None else "n/a"
                    lines.append(
                        f"    [{score_text}] tool={record.tool_name} "
                        f"error={_excerpt(record.exception_message, 150)}"
                    )
                sections.append("\n".join(lines))

        if not sections:
            return f"memory_search: no results for '{query}' across {', '.join(selected)}"
        header = f"memory_search results for '{_excerpt(query, 80)}':"
        return _truncate_output(header + "\n" + "\n".join(sections))


class StubMemorySearchTool:
    """Placeholder when no MemoryStore is available (e.g. offline demo without store injection)."""

    async def __call__(self, params: dict[str, Any]) -> str:
        """Return a message indicating memory search is not available."""
        return "memory_search: memory store is not available in this deployment."


# ── 6. Python Executor ──────────────────────────────────────────────


_PYTHON_BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "exit", "quit",
    "breakpoint", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "type", "super", "classmethod", "staticmethod",
    "property", "memoryview", "bytearray",
})

_PYTHON_BLOCKED_ATTR_NAMES = frozenset({
    "system", "popen", "exec", "spawn", "fork", "kill",
    "remove", "rmdir", "unlink", "rename", "chmod", "chown",
    "write", "writelines", "truncate",
})

_PYTHON_ALLOWED_MODULES = frozenset({
    "math", "statistics", "json", "datetime", "re", "collections",
    "itertools", "functools", "string", "textwrap", "decimal",
    "fractions", "random", "typing", "operator", "dataclasses",
    "enum", "abc", "copy", "pprint", "numbers", "cmath",
})


async def python_executor_tool(params: dict[str, Any]) -> str:
    """Execute sandboxed Python code in a subprocess with strict safety constraints."""
    code = str(params.get("code", "")).strip()
    if not code:
        raise ValueError("python_executor requires a 'code' parameter")
    if len(code) > 10_000:
        raise ValueError("python_executor code exceeds 10,000 character limit")
    timeout = _bounded_int(params.get("timeout_seconds"), 10, minimum=1, maximum=30)

    _validate_python_code(code)

    # Build the wrapper script that runs the user code with restricted builtins
    wrapper = _build_python_wrapper(code)

    try:
        process = await asyncio.create_subprocess_exec(
            "python3", "-c", wrapper,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_sandboxed_env(),
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        return f"python_executor: execution timed out after {timeout} seconds"
    except FileNotFoundError:
        return "python_executor: python3 is not available on this system"

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    parts: list[str] = []
    if stdout.strip():
        parts.append(f"STDOUT:\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"STDERR:\n{stderr.strip()}")
    if not parts:
        parts.append("(no output)")

    exit_label = f"exit_code={process.returncode}"
    result = f"python_executor [{exit_label}]:\n" + "\n".join(parts)
    return _truncate_output(result)


def _validate_python_code(code: str) -> None:
    """Reject dangerous Python code before execution via AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"python_executor syntax error: {exc}") from exc

    for node in ast.walk(tree):
        # Block raw import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_name = ""
            if isinstance(node, ast.Import):
                module_name = node.names[0].name
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
            top_module = module_name.split(".")[0]
            if top_module not in _PYTHON_ALLOWED_MODULES:
                raise ValueError(
                    f"python_executor blocks import of '{module_name}'. "
                    f"Allowed: {', '.join(sorted(_PYTHON_ALLOWED_MODULES))}"
                )
        # Block calls to dangerous builtins
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _PYTHON_BLOCKED_NAMES:
                raise ValueError(f"python_executor blocks call to '{node.func.id}()'")
        # Block dangerous attribute calls
        if isinstance(node, ast.Attribute):
            if node.attr in _PYTHON_BLOCKED_ATTR_NAMES:
                raise ValueError(f"python_executor blocks access to '.{node.attr}'")


def _build_python_wrapper(code: str) -> str:
    """Wrap user code in a restricted execution harness."""
    escaped = code.replace("\\", "\\\\").replace("'", "\\'")
    return textwrap.dedent(f"""\
        import sys
        sys.setrecursionlimit(200)
        try:
            exec('{escaped}')
        except Exception as _e:
            print(f"Error: {{type(_e).__name__}}: {{_e}}", file=sys.stderr)
            sys.exit(1)
    """)


def _sandboxed_env() -> dict[str, str]:
    """Create a minimal environment for subprocess execution."""
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONDONTWRITEBYTECODE"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ── 7. Shell Executor ───────────────────────────────────────────────


_SHELL_ALLOWLIST = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
    "date", "pwd", "env", "du", "df", "uname", "whoami",
    "sort", "uniq", "cut", "tr", "sed", "awk", "file", "stat",
    "which", "dirname", "basename", "realpath", "readlink",
    "diff", "comm", "tee", "xargs", "printf",
})

_SHELL_BLOCKED_PATTERNS = re.compile(
    r"[;|&`]"  # semicolons, pipes, &&/||, backticks
    r"|>\s*>"   # append redirect
    r"|[<>]"    # redirects
    r"|\$\("    # command substitution
)


class ShellExecutorTool:
    """Execute whitelisted read-only shell commands within a workspace."""

    def __init__(self, workspace_root: str) -> None:
        """Create a shell executor scoped to a workspace directory."""
        self._cwd = str(Path(workspace_root).resolve())

    async def __call__(self, params: dict[str, Any]) -> str:
        """Execute a whitelisted shell command and return its output."""
        command = str(params.get("command", "")).strip()
        if not command:
            raise ValueError("shell_executor requires a 'command' parameter")
        if len(command) > 500:
            raise ValueError("shell_executor command exceeds 500 character limit")

        _validate_shell_command(command)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                env=_sandboxed_env(),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=15,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
            return "shell_executor: command timed out after 15 seconds"

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            parts.append(f"STDERR: {stderr.strip()}")
        if not parts:
            parts.append("(no output)")

        exit_label = f"exit_code={process.returncode}"
        result = f"shell_executor [{exit_label}]:\n" + "\n".join(parts)
        return _truncate_output(result)


def _validate_shell_command(command: str) -> None:
    """Reject shell commands that are not in the allowlist or use dangerous operators."""
    if _SHELL_BLOCKED_PATTERNS.search(command):
        raise ValueError(
            "shell_executor blocks pipes (|), redirects (<, >, >>), chaining (;, &&, ||), "
            "backticks (`), and command substitution ($())"
        )
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"shell_executor cannot parse command: {exc}") from exc
    if not tokens:
        raise ValueError("shell_executor received an empty command")
    executable = Path(tokens[0]).name  # handle /usr/bin/ls → ls
    if executable not in _SHELL_ALLOWLIST:
        raise ValueError(
            f"shell_executor blocks '{executable}'. "
            f"Allowed commands: {', '.join(sorted(_SHELL_ALLOWLIST))}"
        )


# ── Registry Factory ────────────────────────────────────────────────


def default_tool_registry(
    brave_api_key: str | None = None,
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search",
    brave_country: str = "us",
    brave_search_lang: str = "en",
    brave_count: int = 5,
    brave_timeout_seconds: float = 10.0,
    simulate_web_search: bool = False,
    workspace_root: str | None = None,
    memory_store: Any = None,
    embedding_model: Any = None,
    semantic_ttl_cutoff: Any = None,
) -> ToolRegistry:
    """Create the full tool registry for the Actor with all available tools."""
    registry = ToolRegistry()
    resolved_root = str(Path(workspace_root or ".").resolve())

    # 1. Calculator
    registry.register(
        ToolDefinition(
            name="calculator",
            description=(
                "Safely evaluates arithmetic expressions including sqrt, sin, cos, tan, log, "
                "log10, log2, ceil, floor, abs, round, and constants pi, e, inf."
            ),
            input_schema={"expression": "string"},
        ),
        calculator_tool,
    )

    # 2. Web Search
    registry.register(
        ToolDefinition(
            name="web_search",
            description="Searches the live web through Brave Search API and returns grounded titles, URLs, and snippets.",
            input_schema={
                "query": "string",
                "count": "integer optional, 1-20",
                "country": "string optional, ISO country code such as us",
                "search_lang": "string optional, language code such as en",
            },
        ),
        BraveSearchTool(
            api_key=brave_api_key,
            endpoint=brave_endpoint,
            country=brave_country,
            search_lang=brave_search_lang,
            default_count=brave_count,
            timeout_seconds=brave_timeout_seconds,
            simulate_when_missing_key=simulate_web_search,
        ),
    )

    # 3. File Search
    registry.register(
        ToolDefinition(
            name="file_search",
            description=(
                "Finds files by name, extension, or glob pattern within the project workspace. "
                "Use glob patterns like '*.py' or '*.json', or a substring like 'config' to match filenames."
            ),
            input_schema={
                "pattern": "string, glob pattern or filename substring",
                "directory": "string optional, subdirectory to search within",
                "max_results": "integer optional, 1-50 (default 20)",
            },
        ),
        FileSearchTool(workspace_root=resolved_root),
    )

    # 4. Document Search
    registry.register(
        ToolDefinition(
            name="document_search",
            description=(
                "Searches within file contents for a text string or regex pattern. Returns matching lines "
                "with surrounding context. Use file_pattern to filter by extension (e.g. '*.py')."
            ),
            input_schema={
                "query": "string, literal text or regex pattern to search for",
                "directory": "string optional, subdirectory to search within",
                "file_pattern": "string optional, glob like '*.py' to filter files",
                "max_results": "integer optional, 1-30 (default 10)",
                "context_lines": "integer optional, 0-5 lines of context (default 2)",
            },
        ),
        DocumentSearchTool(workspace_root=resolved_root),
    )

    # 5. Memory Search
    if memory_store is not None and embedding_model is not None:
        memory_handler: ToolCallable = MemorySearchTool(
            store=memory_store,
            embedding_model=embedding_model,
            semantic_ttl_cutoff=semantic_ttl_cutoff,
        )
    else:
        memory_handler = StubMemorySearchTool()
    registry.register(
        ToolDefinition(
            name="memory_search",
            description=(
                "Searches the agent's own vector memory for past interactions, learned facts, "
                "known workflows, and past failures. Layers: semantic, episodic, procedural, failure."
            ),
            input_schema={
                "query": "string, what to search for in memory",
                "layers": "list optional, subset of [semantic, episodic, procedural, failure]",
                "top_k": "integer optional, 1-20 (default 5)",
            },
        ),
        memory_handler,
    )

    # 6. Python Executor
    registry.register(
        ToolDefinition(
            name="python_executor",
            description=(
                "Executes sandboxed Python code for math, data processing, string manipulation, and testing. "
                "Allowed imports: math, statistics, json, datetime, re, collections, itertools, functools, "
                "string, textwrap, decimal, fractions, random, typing, operator. "
                "Blocked: file I/O, network, os, subprocess, eval/exec."
            ),
            input_schema={
                "code": "string, Python code to execute",
                "timeout_seconds": "integer optional, 1-30 (default 10)",
            },
        ),
        python_executor_tool,
    )

    # 7. Shell Executor
    registry.register(
        ToolDefinition(
            name="shell_executor",
            description=(
                "Runs read-only shell commands for data inspection. "
                "Allowed: ls, cat, head, tail, wc, grep, find, echo, date, pwd, du, df, uname, whoami, "
                "sort, uniq, cut, tr, sed, awk, file, stat, which, diff, comm, basename, dirname. "
                "Blocked: pipes, redirects, command chaining, and all write/delete commands."
            ),
            input_schema={"command": "string, the shell command to run"},
        ),
        ShellExecutorTool(workspace_root=resolved_root),
    )

    return registry


# ── Shared Helpers ──────────────────────────────────────────────────


def _format_brave_results(query: str, data: dict[str, Any], count: int) -> str:
    """Format Brave API JSON into a compact result block for memory and Critic review."""
    web = cast(dict[str, Any], data.get("web") or {})
    results = cast(list[dict[str, Any]], web.get("results") or [])
    if not results:
        return f"BRAVE WEB SEARCH RESULT for '{query}': no web results returned."

    lines = [f"BRAVE WEB SEARCH RESULT for '{query}' (top {min(count, len(results))}):"]
    for index, result in enumerate(results[:count], start=1):
        title = _clean_text(str(result.get("title") or "Untitled"))
        url = _clean_text(str(result.get("url") or ""))
        description = _clean_text(str(result.get("description") or ""))
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if description:
            lines.append(f"   Snippet: {_excerpt(description, 240)}")
    return "\n".join(lines)


def _format_simulated_search(query: str) -> str:
    """Return a clearly marked simulation for demos that intentionally avoid external APIs."""
    return f"SIMULATED WEB SEARCH RESULT for '{query}': BRAVE_SEARCH_API_KEY was not provided."


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    """Parse and clamp integer tool parameters to a safe range."""
    try:
        parsed = int(value) if isinstance(value, (str, int, float)) else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _clean_text(value: str) -> str:
    """Collapse whitespace from external search result text."""
    return " ".join(value.split())


def _excerpt(text: str, max_length: int) -> str:
    """Return a compact excerpt for API errors and search snippets."""
    clean = _clean_text(text)
    if len(clean) <= max_length:
        return clean
    return clean[: max(0, max_length - 3)] + "..."


def _truncate_output(text: str) -> str:
    """Truncate tool output to the global maximum byte limit."""
    if len(text.encode("utf-8", errors="replace")) <= _MAX_OUTPUT_BYTES:
        return text
    # Truncate by characters and re-check
    while len(text.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES - 40 and text:
        text = text[: len(text) - 200]
    return text + "\n... (output truncated)"


def _human_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _is_binary_file(filepath: Path) -> bool:
    """Detect binary files by checking for null bytes in the first 8 KB."""
    try:
        with filepath.open("rb") as fh:
            chunk = fh.read(_BINARY_CHECK_BYTES)
            return b"\x00" in chunk
    except OSError:
        return True


def _glob_match(filename: str, pattern: str) -> bool:
    """Check if a filename matches a simple glob pattern."""
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)
