"""Provider adapters plus the model-driven tool-calling loop."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from .config import Settings
from .tools import ToolRegistry, heuristic_tool_calls

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise engineering assistant for a retrieval-augmented system.

Rules:
- Prefer the supplied context and tool results over prior knowledge.
- Call `search_knowledge_base` when the answer may live in the project documents, and search again with different wording if the first results are thin.
- Call `calculator` for any arithmetic. Never compute it yourself.
- If the retrieved material does not answer the question, say so plainly instead of guessing.

Reply with a single JSON object and nothing else, using exactly these keys:
  "answer": string, the response for the user.
  "confidence": number between 0 and 1, how well the evidence supports the answer.
  "citations": array of strings, the document names you actually relied on."""

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class Completion:
    answer: str
    confidence: float = 0.5
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "local"
    model: str = "offline"


@dataclass(frozen=True)
class Target:
    label: str
    base_url: str
    api_key: str
    model: str


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code in RETRYABLE_STATUS
    return False


class LLMService:
    def __init__(self, settings: Settings, registry: ToolRegistry):
        self.settings = settings
        self.registry = registry
        self._clients: dict[tuple[str, str], AsyncOpenAI] = {}
        # Bounds in-flight upstream calls so a burst queues instead of stampeding.
        self._gate = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    def _client(self, target: Target) -> AsyncOpenAI:
        key = (target.base_url, target.api_key)
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                api_key=target.api_key or "EMPTY",
                base_url=target.base_url,
                timeout=self.settings.request_timeout_seconds,
                max_retries=0,  # retries are handled here so backoff is observable
            )
        return self._clients[key]

    def _primary(self) -> Target | None:
        if self.settings.provider == "vllm":
            return Target("vllm", self.settings.vllm_base_url, "EMPTY", self.settings.vllm_model)
        if self.settings.provider == "openai":
            return Target("openai", self.settings.openai_base_url, self.settings.openai_api_key, self.settings.openai_model)
        return None

    def _fallback(self, primary: Target) -> Target | None:
        if self.settings.fallback_provider == "openai" and self.settings.openai_api_key:
            candidate = Target(
                "fallback-openai", self.settings.openai_base_url, self.settings.openai_api_key, self.settings.fallback_model
            )
        elif self.settings.fallback_provider == "vllm":
            candidate = Target("fallback-vllm", self.settings.vllm_base_url, "EMPTY", self.settings.vllm_model)
        else:
            return None
        if (candidate.base_url, candidate.model) == (primary.base_url, primary.model):
            return None
        return candidate

    async def complete(self, question: str, context: str, temperature: float, top_p: float) -> Completion:
        primary = self._primary()
        if primary is None:
            return self._offline(question, context)

        try:
            return await self._run(primary, question, context, temperature, top_p)
        except Exception as primary_error:
            fallback = self._fallback(primary)
            if fallback is None:
                raise
            logger.warning("primary provider %s failed (%s); trying %s", primary.label, primary_error, fallback.label)
            try:
                return await self._run(fallback, question, context, temperature, top_p)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"primary {primary.label} failed ({primary_error}); fallback {fallback.label} failed ({fallback_error})"
                ) from fallback_error

    async def _run(self, target: Target, question: str, context: str, temperature: float, top_p: float) -> Completion:
        user_content = f"Retrieved context:\n{context or '(nothing retrieved yet - use search_knowledge_base)'}\n\nQuestion: {question}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        executed: list[dict[str, Any]] = []
        retrieved: list[dict[str, Any]] = []

        for iteration in range(self.settings.max_tool_iterations):
            last_round = iteration == self.settings.max_tool_iterations - 1
            message = await self._chat(target, messages, temperature, top_p, use_tools=not last_round)

            if not getattr(message, "tool_calls", None):
                return self._parse(message.content, executed, retrieved, target)

            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                payload, matches = await self.registry.execute(call.function.name, call.function.arguments)
                executed.append(
                    {
                        "name": call.function.name,
                        "arguments": _safe_json(call.function.arguments),
                        "result": _safe_json(payload),
                    }
                )
                if matches:
                    retrieved.extend(matches)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})

        # Tool budget spent: ask once more with tools off so a final answer is produced.
        message = await self._chat(target, messages, temperature, top_p, use_tools=False)
        return self._parse(message.content, executed, retrieved, target)

    async def _chat(self, target: Target, messages: list[dict[str, Any]], temperature: float, top_p: float, use_tools: bool):
        extra: dict[str, Any] = {"tools": self.registry.schemas, "tool_choice": "auto"} if use_tools else {}
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries):
            try:
                async with self._gate:
                    response = await self._client(target).chat.completions.create(
                        model=target.model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        response_format={"type": "json_object"},
                        **extra,
                    )
                return response.choices[0].message
            except Exception as error:
                last_error = error
                if attempt == self.settings.max_retries - 1 or not _is_retryable(error):
                    break
                # Exponential backoff with jitter so retries do not synchronise.
                delay = 0.4 * (2**attempt) * (1 + random.random() * 0.25)
                logger.warning("%s attempt %d failed (%s); retrying in %.2fs", target.label, attempt + 1, error, delay)
                await asyncio.sleep(delay)

        raise RuntimeError(f"{target.label} request failed after {self.settings.max_retries} attempts: {last_error}") from last_error

    def _parse(
        self, content: str | None, executed: list[dict[str, Any]], retrieved: list[dict[str, Any]], target: Target
    ) -> Completion:
        payload: dict[str, Any]
        try:
            parsed = json.loads(content or "{}")
            payload = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # Graceful degradation: a non-JSON reply is still worth returning.
            logger.warning("%s returned non-JSON content; passing it through verbatim", target.label)
            payload = {"answer": (content or "").strip(), "confidence": 0.3}

        confidence = payload.get("confidence", 0.5)
        citations = payload.get("citations") or []
        return Completion(
            answer=str(payload.get("answer") or "The model returned an empty answer."),
            confidence=min(max(float(confidence), 0.0), 1.0) if isinstance(confidence, (int, float)) else 0.5,
            citations=[str(item) for item in citations] if isinstance(citations, list) else [],
            tool_calls=executed,
            retrieved=retrieved,
            provider=target.label,
            model=target.model,
        )

    def _offline(self, question: str, context: str) -> Completion:
        """Keyless demo path. No model, so tool selection is heuristic."""
        tool_calls, tool_context = heuristic_tool_calls(question, self.registry)
        if tool_context:
            return Completion(answer=tool_context, confidence=1.0, tool_calls=tool_calls)
        if not context:
            return Completion(answer="I do not have enough indexed context to answer that question.", confidence=0.0)
        first = context.splitlines()[0]
        return Completion(answer=f"Based on the indexed documents: {first}", confidence=0.4)


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
