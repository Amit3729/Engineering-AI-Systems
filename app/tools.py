"""Tools the model may call, exposed as OpenAI-compatible function schemas.

The registry is an allow-list: the model can only name a tool declared in
``TOOL_SCHEMAS``, arguments are validated before execution, and every tool
returns a JSON string so the result can be fed straight back as a tool message.
"""

from __future__ import annotations

import ast
import asyncio
import json
import operator
import re
from datetime import datetime, timezone
from typing import Any, Callable

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search over the ingested engineering documents. Call this "
                "whenever the question could be answered by project documentation, and "
                "call it again with a reworded query if the first results look thin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How many chunks to return."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression exactly. Use it instead of doing mental arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic only, e.g. '12 * 4' or '(1200/8) ** 2'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Current UTC timestamp in ISO-8601. Use it for anything date or time relative.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class ToolError(Exception):
    """Raised for a bad call; surfaced to the model so it can correct itself."""


def calculator(expression: str) -> float:
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ToolError("Only basic arithmetic (+ - * / // % **) is supported")

    try:
        return evaluate(ast.parse(expression, mode="eval").body)
    except ToolError:
        raise
    except (SyntaxError, ValueError, TypeError) as error:
        raise ToolError(f"Could not parse expression: {error}") from error
    except ZeroDivisionError as error:
        raise ToolError("Division by zero") from error


class ToolRegistry:
    """Binds tool names to implementations, with the retriever injected."""

    def __init__(self, retriever: Any):
        self._retriever = retriever
        self._handlers: dict[str, Callable[..., Any]] = {
            "search_knowledge_base": self._search,
            "calculator": lambda expression: {"expression": expression, "result": calculator(expression)},
            "current_time": lambda: {"utc": datetime.now(timezone.utc).isoformat()},
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def _search(self, query: str, top_k: int = 4) -> dict[str, Any]:
        matches = self._retriever.search(query, limit=max(1, min(int(top_k), 10)))
        return {"query": query, "matches": matches}

    async def execute(self, name: str, raw_arguments: str) -> tuple[str, dict[str, Any] | None]:
        """Run one model-requested tool call.

        Returns the JSON payload to hand back to the model, plus the structured
        matches when the call was a retrieval (so the API can cite sources).
        """
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool '{name}'"}), None

        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ToolError("Tool arguments must be a JSON object")
            result = await asyncio.to_thread(handler, **arguments)
        except ToolError as error:
            return json.dumps({"error": str(error)}), None
        except (json.JSONDecodeError, TypeError) as error:
            return json.dumps({"error": f"Invalid arguments for '{name}': {error}"}), None
        except Exception as error:  # a failing tool must not fail the request
            return json.dumps({"error": f"Tool '{name}' failed: {error}"}), None

        matches = result.get("matches") if name == "search_knowledge_base" else None
        return json.dumps(result, default=str), matches


def heuristic_tool_calls(question: str, registry: ToolRegistry) -> tuple[list[dict[str, Any]], str]:
    """Offline stand-in for model-driven tool selection.

    Only used by the keyless ``local`` provider, which has no model to do the
    choosing. Real providers go through ``ToolRegistry.execute``.
    """
    match = re.search(r"(?:calculate|compute|what is)\s+([0-9.+*/%()\-\s]+?)\s*[?.]?$", question.lower())
    if not match:
        return [], ""
    expression = match.group(1).strip()
    if not any(character.isdigit() for character in expression):
        return [], ""
    try:
        result = calculator(expression)
    except ToolError:
        return [], ""
    return [{"name": "calculator", "arguments": {"expression": expression}, "result": result}], f"Calculator result: {result}"
