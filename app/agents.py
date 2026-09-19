"""Multi-agent research mode: planner, parallel researchers, synthesiser.

Why this exists
---------------
The knowledge base is now the whole CPython 3.14 manual: 545 pages, ~16.5k
chunks. The W15 single-agent loop answers a question by appending every
``search_knowledge_base`` result to one message list verbatim. At ``top_k`` 5
that is roughly 1.2k tokens per search, and a question that spans two modules
needs three or four searches, so by the last turn the original question is
sitting under 5k tokens of manual text and the model starts answering from
whatever it read most recently rather than from what is most relevant.

Two things fix that, and they are separable:

* **Context isolation.** Each sub-question is researched by its own agent with
  its own message list. Raw chunks live and die inside that agent.
* **Compaction.** What crosses an agent boundary is a list of
  ``(claim, citation)`` findings, not chunk text. The full chunks are carried
  out-of-band on the ``Digest`` so the API can still return them as sources and
  the user can still read them - they simply never enter another model's prompt.

Compaction is also applied *inside* a researcher: once a tool result is no
longer the most recent one, its payload is rewritten to the list of documents it
cited. The researcher can still see what it has already looked at without
paying for the text a second time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .llm import Completion, LLMService, Usage, _safe_json
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You plan documentation research. The library is the CPython 3.14 manual plus a few project notes.

Split the user's question into the smallest set of independent lookups that together answer it.
Two lookups are independent when neither needs the other's result - they will be run in parallel.

Rules:
- One lookup is the right answer for a question about a single function, class or concept. Do not invent extra work.
- Use a separate lookup per module or topic when a question spans several (for example datetime and zoneinfo).
- Never produce more than {max_tasks} lookups.
- Return an EMPTY array when no documentation is needed at all - arithmetic, the current
  time, or anything answerable without a lookup. An empty array is a valid, useful answer.

Reply with a single JSON object and nothing else:
  "tasks": array of objects, each {{"question": string, "source": string}}.
           "source" is "python-3.14-docs-html" for the CPython manual, "root" for project notes,
           or "" when it could be either."""

RESEARCHER_PROMPT = """You are a documentation researcher. You answer ONE narrow question by searching, and you report findings for another agent to write up.

Rules:
- Search first. Search again with different wording if the first results are thin or off-target.
- Report only what the retrieved text actually says. Never fill a gap from prior knowledge.
- Every finding must carry the document it came from, as "path#anchor" exactly as it appears in the results.
- If the documents do not answer the question, return an empty "findings" array and say so in "gaps". That is a useful result, not a failure.

Reply with a single JSON object and nothing else:
  "findings": array of objects, each {"claim": string, "citation": string}. Keep each claim to one sentence.
  "gaps": string, what the question still needs and the retrieved text did not cover. Empty string if nothing is missing."""

SYNTHESISER_PROMPT = """You write the final answer from research findings supplied by other agents.

Rules:
- The findings are your only evidence. You cannot search, and you must not add facts from prior knowledge.
- If the findings do not answer the question, say plainly what is missing. Do not paper over a gap.
- Report a low confidence when the findings are thin or a researcher flagged a gap.
- Call `calculator` for any arithmetic. Never compute it yourself.

Reply with a single JSON object and nothing else, using exactly these keys:
  "answer": string, the response for the user.
  "confidence": number between 0 and 1, how well the findings support the answer.
  "citations": array of strings, the documents you actually relied on."""


@dataclass
class Digest:
    """What one researcher hands to the synthesiser.

    ``findings`` is what crosses the boundary. ``retrieved`` is the evidence
    behind it, kept for the API response and deliberately kept out of any
    further prompt.
    """

    task: str
    findings: list[dict[str, str]] = field(default_factory=list)
    gaps: str = ""
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str = ""

    def as_prompt_block(self) -> str:
        lines = [f"### Lookup: {self.task}"]
        if self.error:
            lines.append(f"- FAILED: {self.error}")
        for finding in self.findings:
            lines.append(f"- {finding['claim']} [{finding['citation']}]")
        if not self.findings and not self.error:
            lines.append("- (no supporting text was found)")
        if self.gaps:
            lines.append(f"- GAP: {self.gaps}")
        return "\n".join(lines)


def compact_history(messages: list[dict[str, Any]], keep_raw: int) -> int:
    """Replace all but the newest ``keep_raw`` tool payloads with their citations.

    Returns the number of messages rewritten. Operates in place on the list the
    caller is about to send, so the saving lands on the very next request.
    """
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    stale = tool_indexes[:-keep_raw] if keep_raw else tool_indexes
    for index in stale:
        message = messages[index]
        if message.get("_compacted"):
            continue
        message["content"] = _citations_only(message["content"])
        message["_compacted"] = True
    return len([index for index in stale if messages[index].get("_compacted")])


def _citations_only(payload: str) -> str:
    """A one-line stand-in for a retrieval payload: what it cited, not what it said."""
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return "[earlier tool result omitted to save context]"
    if not isinstance(parsed, dict) or "matches" not in parsed:
        return payload if len(payload) < 200 else payload[:200] + " …[truncated]"

    seen: list[str] = []
    for match in parsed.get("matches") or []:
        anchor = match.get("anchor") or ""
        reference = f"{match.get('document', '?')}{'#' + anchor if anchor else ''}"
        if reference not in seen:
            seen.append(reference)
    query = parsed.get("query", "")
    return json.dumps({"note": "earlier search, text dropped to save context", "query": query, "documents": seen})


def with_default_top_k(raw_arguments: str, top_k: int) -> str:
    """Fill in a researcher's retrieval breadth when the model did not pick one.

    A researcher's budget is a property of the pattern, not of the model's mood,
    so ``RESEARCHER_TOP_K`` is applied here rather than hoped for in a prompt.
    An explicit choice by the model is left alone.
    """
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return raw_arguments  # let the registry report the malformed call
    if not isinstance(arguments, dict) or "top_k" in arguments:
        return raw_arguments
    return json.dumps(arguments | {"top_k": top_k})


def _strip_internal(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop bookkeeping keys the provider would reject."""
    return [{key: value for key, value in message.items() if not key.startswith("_")} for message in messages]


def _parse_json(content: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(content or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class ResearchTeam:
    """Planner -> N parallel researchers -> synthesiser."""

    def __init__(self, settings: Settings, registry: ToolRegistry, llm: LLMService):
        self.settings = settings
        self.registry = registry
        self.llm = llm

    # ---------------------------------------------------------------- public

    async def answer(self, question: str, context: str, temperature: float, top_p: float) -> Completion:
        usage = Usage()
        try:
            plan, target = await self._plan(question, temperature, top_p, usage)
        except Exception as error:
            # The planner is a single point of failure for this pattern, so it
            # degrades to the pattern that does not need it.
            logger.warning("planner failed (%s); falling back to the single-agent loop", error)
            plan = []

        if not plan:
            # Nothing to research: fall back to the single-agent loop, which
            # still has the calculator and the clock. Paying three agents to
            # answer "what is 12 * 4" would be coordination cost for nothing.
            logger.info("planner found no research tasks; using the single-agent loop")
            completion = await self.llm.complete(question, context, temperature, top_p)
            completion.usage.merge(usage)
            completion.agent_mode = "multi->single"
            completion.iterations += usage.model_calls  # the planner turn counts too
            return completion

        digests = await asyncio.gather(
            *(self._research(task, temperature, top_p) for task in plan),
            return_exceptions=False,
        )
        for digest in digests:
            usage.merge(digest.usage)

        completion = await self._synthesise(question, digests, temperature, top_p, usage, target)
        completion.plan = [
            {
                "task": digest.task,
                "source": task.get("source", ""),
                "findings": len(digest.findings),
                "iterations": digest.iterations,
                "tokens": digest.usage.total_tokens,
                "gaps": digest.gaps,
                "error": digest.error,
            }
            for task, digest in zip(plan, digests)
        ]
        return completion

    # --------------------------------------------------------------- planner

    async def _plan(
        self, question: str, temperature: float, top_p: float, usage: Usage
    ) -> tuple[list[dict[str, str]], Any]:
        messages = [
            {"role": "system", "content": PLANNER_PROMPT.format(max_tasks=self.settings.max_researchers)},
            {"role": "user", "content": question},
        ]
        message, target = await self.llm.call(messages, temperature=temperature, top_p=top_p, usage=usage)
        payload = _parse_json(message.content)

        # The task list is the decision. An earlier version also asked for a
        # "needs_research" boolean and got a flat contradiction back - qwen2.5
        # returned false alongside two perfectly good lookups, and every
        # documentation question silently fell through to the single-agent loop.
        # One signal cannot disagree with itself.
        tasks: list[dict[str, str]] = []
        for raw in payload.get("tasks") or []:
            if isinstance(raw, dict) and str(raw.get("question", "")).strip():
                tasks.append({"question": str(raw["question"]).strip(), "source": str(raw.get("source", "") or "")})
        return tasks[: self.settings.max_researchers], target

    # ------------------------------------------------------------ researcher

    async def _research(self, task: dict[str, str], temperature: float, top_p: float) -> Digest:
        digest = Digest(task=task["question"])
        hint = f"\nSearch the '{task['source']}' corpus." if task.get("source") else ""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RESEARCHER_PROMPT},
            {"role": "user", "content": f"Question: {task['question']}{hint}"},
        ]
        tools = [schema for schema in self.registry.schemas if schema["function"]["name"] == "search_knowledge_base"]
        if not tools:
            # The search tool has been withdrawn (fault injection). Report the
            # gap rather than letting the researcher answer from memory.
            digest.error = "search_knowledge_base is unavailable, so nothing could be retrieved"
            return digest

        try:
            for iteration in range(self.settings.researcher_max_iterations):
                digest.iterations = iteration + 1
                last_round = iteration == self.settings.researcher_max_iterations - 1
                compact_history(messages, self.settings.compaction_keep_raw)
                outgoing = _strip_internal(messages)
                message, target = await self.llm.call(
                    outgoing,
                    tools=None if last_round else tools,
                    temperature=temperature,
                    top_p=top_p,
                    usage=digest.usage,
                )

                if not getattr(message, "tool_calls", None):
                    # Without the envelope there are no findings to report at all.
                    message = await self.llm.structured(
                        outgoing, message, target, temperature=temperature, top_p=top_p, usage=digest.usage
                    )
                    return self._finish(digest, message.content)

                messages.append(message.model_dump(exclude_none=True))
                for call in message.tool_calls:
                    arguments = with_default_top_k(call.function.arguments, self.settings.researcher_top_k)
                    payload, matches = await self.registry.execute(call.function.name, arguments)
                    digest.tool_calls.append(
                        {
                            "name": call.function.name,
                            "arguments": _safe_json(arguments),
                            "result": _safe_json(payload),
                        }
                    )
                    if matches:
                        digest.retrieved.extend(matches)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})

            digest.iterations += 1
            compact_history(messages, self.settings.compaction_keep_raw)
            message, _ = await self.llm.call(
                _strip_internal(messages), temperature=temperature, top_p=top_p, usage=digest.usage
            )
            return self._finish(digest, message.content)
        except Exception as error:  # one researcher failing must not sink the answer
            logger.warning("researcher for %r failed: %s", task["question"], error)
            digest.error = str(error)
            return digest

    @staticmethod
    def _finish(digest: Digest, content: str | None) -> Digest:
        payload = _parse_json(content)
        for raw in payload.get("findings") or []:
            if isinstance(raw, dict) and str(raw.get("claim", "")).strip():
                digest.findings.append(
                    {"claim": str(raw["claim"]).strip(), "citation": str(raw.get("citation", "") or "")}
                )
        digest.gaps = str(payload.get("gaps") or "")
        return digest

    # ----------------------------------------------------------- synthesiser

    async def _synthesise(
        self,
        question: str,
        digests: list[Digest],
        temperature: float,
        top_p: float,
        usage: Usage,
        target: Any,
    ) -> Completion:
        brief = "\n\n".join(digest.as_prompt_block() for digest in digests)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYNTHESISER_PROMPT},
            {"role": "user", "content": f"Research findings:\n\n{brief}\n\nQuestion: {question}"},
        ]
        tools = [schema for schema in self.registry.schemas if schema["function"]["name"] != "search_knowledge_base"]

        executed: list[dict[str, Any]] = []
        iterations = 0
        message = None
        for iteration in range(2):
            iterations = iteration + 1
            outgoing = list(messages)
            message, target = await self.llm.call(
                outgoing,
                tools=tools if iteration == 0 else None,
                temperature=temperature,
                top_p=top_p,
                usage=usage,
            )
            if not getattr(message, "tool_calls", None):
                message = await self.llm.structured(
                    outgoing, message, target, temperature=temperature, top_p=top_p, usage=usage
                )
                break
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                payload, _ = await self.registry.execute(call.function.name, call.function.arguments)
                executed.append(
                    {
                        "name": call.function.name,
                        "arguments": _safe_json(call.function.arguments),
                        "result": _safe_json(payload),
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})

        payload = _parse_json(message.content if message else None)
        confidence = payload.get("confidence", 0.5)
        citations = payload.get("citations") or []
        retrieved: list[dict[str, Any]] = []
        for digest in digests:
            retrieved.extend(digest.retrieved)
            executed = digest.tool_calls + executed

        answer = str(payload.get("answer") or (message.content if message else "") or "The model returned an empty answer.")
        return Completion(
            answer=answer,
            confidence=min(max(float(confidence), 0.0), 1.0) if isinstance(confidence, (int, float)) else 0.5,
            citations=[str(item) for item in citations] if isinstance(citations, list) else [],
            tool_calls=executed,
            retrieved=retrieved,
            provider=target.label,
            model=target.model,
            # Trajectory length for the whole team: the planner turn, the longest
            # researcher (they run in parallel), and the synthesiser's turns.
            iterations=1 + max((digest.iterations for digest in digests), default=0) + iterations,
            usage=usage,
            agent_mode="multi",
        )
