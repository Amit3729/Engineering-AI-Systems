"""Evaluation harness, written from scratch for this assistant.

It drives the same service objects the API uses - ``Retriever``, ``ToolRegistry``,
``LLMService``, ``ResearchTeam`` - but builds its own stack rather than going
through ``/chat``, for two reasons: the response cache would make a second run of
the same query free and meaningless, and each configuration needs its own
settings (agent mode, injected fault) without restarting a process.

What it measures, per query:

* **Task completion** - did the run do what the case asked? For an answerable
  case that means the expected documents were actually cited and any required
  content is present; for an unanswerable one it means the assistant declined.
* **Tool-call correctness** - were the expected tools called, were forbidden
  ones avoided, and does every argument object validate against the tool's own
  JSON schema (required keys, types, bounds, no stray keys)?
* **Trajectory length** - loop turns consumed.
* **Tokens** - prompt and completion tokens over every upstream call, which is
  what makes the multi-agent coordination cost visible against the baseline.
* **Failures** - classified as hard, soft, or cascading soft.

Usage::

    python -m eval.harness --modes single multi
    python -m eval.harness --modes multi --fault malformed_retrieval --tag fault
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents import ResearchTeam
from app.config import Settings, get_settings
from app.llm import Completion, LLMService
from app.retrieval import Retriever
from app.tools import TOOL_SCHEMAS, ToolRegistry

from .dataset import SEARCH, Case, select

# Phrases an assistant uses when it is declining for lack of evidence. This is a
# heuristic, and it is deliberately narrow: it must not fire on a real answer
# that merely contains a caveat.
DECLINE_PATTERNS = re.compile(
    r"(do(es)? not (appear|contain|cover|include|have)|don't have|do not have|no (relevant|information|mention)"
    r"|not (found|covered|available|present|in the)|could not find|couldn't find|unable to (find|answer)"
    r"|cannot (answer|find)|insufficient|nothing (was )?(found|retrieved)|no (documents|results))",
    re.IGNORECASE,
)

FAILURE_TAXONOMY = {
    "hard": "No usable answer was produced: an exception, or an empty response.",
    "soft": "An answer was produced and looks fine, but is wrong on its own terms: "
            "the wrong tool, invalid arguments, a missing requirement, or an ungrounded citation.",
    "cascading_soft": "An upstream step failed or returned nothing usable, and the agent "
                      "still produced a confident answer on top of it.",
}


# --------------------------------------------------------------------- scoring


def validate_arguments(name: str, arguments: Any) -> list[str]:
    """Check one tool call's arguments against that tool's declared schema."""
    schema = next((s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == name), None)
    if schema is None:
        return [f"{name}: not an allow-listed tool"]
    if isinstance(arguments, str):  # the model emitted a string that never parsed
        return [f"{name}: arguments were not a JSON object"]
    if not isinstance(arguments, dict):
        return [f"{name}: arguments were {type(arguments).__name__}, expected object"]

    parameters = schema.get("parameters", {})
    properties: dict[str, Any] = parameters.get("properties", {})
    problems: list[str] = []

    for required in parameters.get("required", []):
        if required not in arguments:
            problems.append(f"{name}: missing required argument '{required}'")
    if parameters.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                problems.append(f"{name}: unexpected argument '{key}'")

    expected_types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for key, value in arguments.items():
        spec = properties.get(key)
        if not spec:
            continue
        python_type = expected_types.get(spec.get("type", ""))
        if python_type and not isinstance(value, python_type):
            problems.append(f"{name}: '{key}' should be {spec['type']}, got {type(value).__name__}")
            continue
        if spec.get("type") == "integer" and isinstance(value, int):
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"{name}: '{key}'={value} below minimum {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                problems.append(f"{name}: '{key}'={value} above maximum {spec['maximum']}")
    return problems


def looks_declined(answer: str, confidence: float) -> bool:
    return bool(DECLINE_PATTERNS.search(answer)) or confidence <= 0.2


def active_model(settings: Settings) -> str:
    """The model that will actually answer, whichever provider is configured."""
    return {
        "openai": settings.openai_model,
        "vllm": settings.vllm_model,
        "ollama": settings.ollama_model,
    }.get(settings.provider, "offline")


@dataclass
class Result:
    case_id: str
    kind: str
    mode: str
    fault: str
    # What actually ran. "multi->single" means the planner found nothing to
    # research and handed the question back to the single-agent loop.
    agent_mode: str = ""
    answered: bool = False
    completed: bool = False
    grounded: bool = False
    declined: bool = False
    tools_correct: bool = False
    tools_called: list[str] = field(default_factory=list)
    argument_problems: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    retrieval_broke: bool = False
    documents: list[str] = field(default_factory=list)
    matched_documents: list[str] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    latency_ms: int = 0
    confidence: float = 0.0
    researchers: int = 0
    failure: str = ""
    reason: str = ""
    answer: str = ""


def score(
    case: Case,
    completion: Completion | None,
    error: str,
    latency_ms: int,
    mode: str,
    fault: str,
    prefetched: list[dict[str, Any]] | None = None,
) -> Result:
    """Grade one run.

    ``prefetched`` is the context the API seeds the prompt with before the agent
    runs. It is reported as a source, but it is deliberately *excluded* from the
    grounding check: finding the right page is the retriever's job, and citing it
    is the agent's. Counting pre-retrieval towards grounding would give every
    agent credit for work it did not do.
    """
    result = Result(case_id=case.id, kind=case.kind, mode=mode, fault=fault, latency_ms=latency_ms)

    if completion is None:
        result.failure, result.reason = "hard", f"the run raised: {error}"
        return result

    answer = (completion.answer or "").strip()
    result.answer = answer
    result.confidence = completion.confidence
    result.iterations = completion.iterations
    result.prompt_tokens = completion.usage.prompt_tokens
    result.completion_tokens = completion.usage.completion_tokens
    result.total_tokens = completion.usage.total_tokens
    result.model_calls = completion.usage.model_calls
    result.researchers = len(completion.plan)
    result.agent_mode = completion.agent_mode
    result.answered = bool(answer)

    if not result.answered:
        result.failure, result.reason = "hard", "the assistant returned an empty answer"
        return result

    # ---- tool-call correctness
    bad_arguments: list[str] = []
    for call in completion.tool_calls:
        result.tools_called.append(call["name"])
        result.argument_problems.extend(validate_arguments(call["name"], call.get("arguments")))
        payload = call.get("result")
        if not (isinstance(payload, dict) and payload.get("error")):
            continue
        result.tool_errors.append(f"{call['name']}: {payload['error']}")
        # An argument the tool rejected is the agent's mistake; a tool that is
        # down or withdrawn is not, and must not be scored as one.
        if payload.get("kind") == "arguments":
            bad_arguments.append(f"{call['name']} rejected the arguments: {payload['error']}")
        if call["name"] == SEARCH:
            result.retrieval_broke = True
    missing = [tool for tool in case.expect_tools if tool not in result.tools_called]
    forbidden = [tool for tool in case.forbid_tools if tool in result.tools_called]
    result.tools_correct = not (missing or forbidden or result.argument_problems or bad_arguments)

    # ---- grounding
    prefetched = prefetched or []
    attributable = {str(citation) for citation in completion.citations}
    attributable |= {match["document"] for match in completion.retrieved}
    result.documents = sorted(attributable | {match["document"] for match in prefetched})
    result.matched_documents = [
        expected for expected in case.expect_documents if any(expected in document for document in attributable)
    ]
    result.grounded = len(result.matched_documents) >= case.required_documents

    result.declined = looks_declined(answer, completion.confidence)
    lowered = answer.lower()
    missing_content = [needle for needle in case.must_include if needle.lower() not in lowered]

    # ---- completion
    problems: list[str] = []
    if case.expect_decline:
        result.completed = result.declined
        if not result.completed:
            problems.append("answered a question the corpus cannot support instead of declining")
    else:
        if missing_content:
            problems.append(f"answer is missing {missing_content}")
        if case.expect_documents and not result.grounded:
            problems.append(
                f"cited {result.matched_documents or 'nothing expected'}, "
                f"needed {case.required_documents} of {list(case.expect_documents)}"
            )
        if result.declined:
            problems.append("declined a question the corpus does support")
        result.completed = not problems
    if missing:
        problems.append(f"never called {missing}")
    if forbidden:
        problems.append(f"called forbidden {forbidden}")
    if result.argument_problems:
        problems.append(f"invalid arguments: {result.argument_problems}")
    problems.extend(bad_arguments)

    if result.completed and result.tools_correct:
        return result

    # ---- failure taxonomy
    # "Usable" means a chunk that actually carries text and cleared the score
    # floor. Corrupted retrieval returns matches, so counting rows is not enough.
    usable = [
        match
        for match in list(completion.retrieved) + list(prefetched)
        if str(match.get("text", "")).strip() and float(match.get("score", 0)) > 0
    ]
    evidence_broke = result.retrieval_broke or (bool(case.expect_documents) and not usable)
    confident = completion.confidence >= 0.5 and not result.declined
    result.failure = "cascading_soft" if evidence_broke and confident else "soft"
    result.reason = "; ".join(problems) or "tool selection was wrong"
    return result


# ---------------------------------------------------------------------- runner


# A 7B model on a laptop answers in tens of seconds, and a queued request waits
# behind the ones ahead of it. The 30s default is a hosted-API figure; keeping it
# here would score the machine's speed as a stream of hard failures.
LOCAL_PROVIDER_DEFAULTS = {"request_timeout_seconds": 900.0, "max_retries": 2}


class Stack:
    """One configured assistant: retriever, tools, model, and both agent modes."""

    def __init__(self, **overrides: Any):
        base = get_settings()
        if overrides.get("provider", base.provider) in {"ollama", "vllm"}:
            overrides = LOCAL_PROVIDER_DEFAULTS | overrides
        self.settings: Settings = base.model_copy(update=overrides)
        if self.settings.provider not in {"openai", "vllm", "ollama"}:
            raise SystemExit(
                "The harness needs a real model provider; 'local' has no model to evaluate. "
                "Set PROVIDER=ollama in .env for a free local run, or pass --provider ollama."
            )
        self.retriever = Retriever(self.settings)
        self.registry = ToolRegistry(self.retriever, self.settings)
        self.llm = LLMService(self.settings, self.registry)
        self.team = ResearchTeam(self.settings, self.registry, self.llm)

    async def run(self, case: Case, mode: str) -> tuple[Completion | None, list[dict[str, Any]], str, int]:
        started = time.perf_counter()
        try:
            # Mirrors /chat: pre-retrieve, then hand off to the configured pattern.
            try:
                matches = await asyncio.to_thread(self.retriever.search, case.question)
            except Exception:
                matches = []  # /chat degrades the same way rather than failing
            prefetched = [m for m in matches if m["score"] >= self.settings.min_score]
            context = "\n\n".join(f"[{m['document']}] {m['text']}" for m in prefetched)

            if mode == "multi":
                completion = await self.team.answer(case.question, context, 0.2, 0.9)
            else:
                completion = await self.llm.complete(case.question, context, 0.2, 0.9)
                completion.agent_mode = "single"
            return completion, prefetched, "", int((time.perf_counter() - started) * 1000)
        except Exception as error:
            return None, [], f"{type(error).__name__}: {error}", int((time.perf_counter() - started) * 1000)


async def run_suite(cases: list[Case], mode: str, fault: str, concurrency: int, **overrides: Any) -> list[Result]:
    stack = Stack(agent_mode=mode, fault_injection=fault, **overrides)
    gate = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> Result:
        async with gate:
            completion, prefetched, error, latency = await stack.run(case, mode)
            result = score(case, completion, error, latency, mode, fault or "none", prefetched)
            flag = "ok " if result.completed and result.tools_correct else result.failure.upper()
            print(f"  [{flag:>14}] {case.id:<24} {result.iterations} turns  {result.total_tokens:>6} tok")
            return result

    return list(await asyncio.gather(*(one(case) for case in cases)))


# ---------------------------------------------------------------------- report


def summarise(results: list[Result]) -> dict[str, Any]:
    total = len(results)
    completed = [r for r in results if r.completed]
    tokens = [r.total_tokens for r in results]
    return {
        "runs": total,
        "completion_rate": len(completed) / total if total else 0.0,
        "tool_correctness_rate": sum(r.tools_correct for r in results) / total if total else 0.0,
        "mean_iterations": statistics.mean([r.iterations for r in results]) if total else 0.0,
        "max_iterations": max([r.iterations for r in results], default=0),
        "mean_tokens": statistics.mean(tokens) if total else 0.0,
        "total_tokens": sum(tokens),
        "mean_latency_ms": statistics.mean([r.latency_ms for r in results]) if total else 0.0,
        "hard": sum(r.failure == "hard" for r in results),
        "soft": sum(r.failure == "soft" for r in results),
        "cascading_soft": sum(r.failure == "cascading_soft" for r in results),
    }


def _table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render(runs: dict[str, list[Result]], out_dir: Path, settings: Settings) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Evaluation report",
        "",
        f"Generated {stamp} · provider `{settings.provider}` · model `{active_model(settings)}` · "
        f"top_k {settings.top_k} · researcher top_k {settings.researcher_top_k} · "
        f"researcher turns {settings.researcher_max_iterations}",
        "",
        "## Summary by configuration",
        "",
    ]

    headers = ["configuration", "n", "completion", "tool-calls", "mean turns", "mean tokens", "total tokens",
               "hard", "soft", "cascading"]
    rows = []
    for label, results in runs.items():
        s = summarise(results)
        rows.append([
            f"`{label}`", str(s["runs"]), f"{s['completion_rate']:.0%}", f"{s['tool_correctness_rate']:.0%}",
            f"{s['mean_iterations']:.1f}", f"{s['mean_tokens']:,.0f}", f"{s['total_tokens']:,}",
            str(s["hard"]), str(s["soft"]), str(s["cascading_soft"]),
        ])
    parts += [_table(rows, headers), ""]

    # Compaction saving: the same mode run with and without it.
    paired = [label for label in runs if label.startswith("multi")]
    on = next((label for label in paired if "no-compaction" not in label), None)
    off = next((label for label in paired if "no-compaction" in label), None)
    if on and off:
        a, b = summarise(runs[on]), summarise(runs[off])
        saved = 1 - (a["mean_tokens"] / b["mean_tokens"]) if b["mean_tokens"] else 0
        parts += [
            "## What compaction saves",
            "",
            f"Same pattern, same queries, compaction on (`{on}`) versus off (`{off}`): "
            f"**{a['mean_tokens']:,.0f} vs {b['mean_tokens']:,.0f}** mean tokens per query, "
            f"a **{saved:.0%}** reduction, at "
            f"{a['completion_rate']:.0%} versus {b['completion_rate']:.0%} completion.",
            "",
        ]

    baseline = runs.get("single")
    multi = runs.get("multi")
    if baseline and multi:
        b, m = summarise(baseline), summarise(multi)
        ratio = m["mean_tokens"] / b["mean_tokens"] if b["mean_tokens"] else 0
        parts += [
            "## Coordination cost",
            "",
            f"The multi-agent mode spends **{ratio:.2f}x** the tokens of the single-agent baseline "
            f"({m['mean_tokens']:,.0f} vs {b['mean_tokens']:,.0f} per query) and completes "
            f"{m['completion_rate']:.0%} of cases against the baseline's {b['completion_rate']:.0%}.",
            "",
        ]

    degraded = {
        label: sum(1 for r in results if r.agent_mode == "multi->single")
        for label, results in runs.items()
        if any(r.agent_mode == "multi->single" for r in results)
    }
    if degraded:
        parts += [
            "The planner declined to fan out on some queries and handed them back to the "
            "single-agent loop: "
            + ", ".join(f"{count} in `{label}`" for label, count in degraded.items())
            + ". Those rows are reported as `multi->single`.",
            "",
        ]

    parts += ["## Completion by query kind", ""]
    kind_rows = []
    kinds = sorted({r.kind for results in runs.values() for r in results})
    for kind in kinds:
        row = [kind]
        for label, results in runs.items():
            subset = [r for r in results if r.kind == kind]
            row.append(f"{sum(r.completed for r in subset)}/{len(subset)}" if subset else "-")
        kind_rows.append(row)
    parts += [_table(kind_rows, ["kind"] + [f"`{label}`" for label in runs]), ""]

    parts += ["## Per-query results", ""]
    detail_headers = ["query", "config", "ran as", "completed", "tools ok", "turns", "tokens", "confidence", "failure"]
    detail_rows = []
    for label, results in runs.items():
        for r in sorted(results, key=lambda r: r.case_id):
            detail_rows.append([
                f"`{r.case_id}`", f"`{label}`", r.agent_mode or "-", "yes" if r.completed else "no",
                "yes" if r.tools_correct else "no", str(r.iterations), f"{r.total_tokens:,}",
                f"{r.confidence:.2f}", r.failure or "-",
            ])
    parts += [_table(detail_rows, detail_headers), ""]

    failures = [(label, r) for label, results in runs.items() for r in results if r.failure]
    parts += ["## Failure log", ""]
    if not failures:
        parts += ["No failures recorded.", ""]
    else:
        for taxon, description in FAILURE_TAXONOMY.items():
            subset = [(label, r) for label, r in failures if r.failure == taxon]
            parts += [f"### {taxon.replace('_', ' ')} ({len(subset)})", "", f"*{description}*", ""]
            if not subset:
                parts += ["None.", ""]
                continue
            for label, r in subset:
                parts += [f"- **`{r.case_id}`** (`{label}`, {r.kind}) — {r.reason}"]
                if r.tool_errors:
                    parts += [f"  - tool errors: {r.tool_errors}"]
                parts += [f"  - confidence {r.confidence:.2f}, {r.iterations} turns, answer: "
                          f"\"{r.answer[:160].replace(chr(10), ' ')}…\""]
            parts += [""]

    report = "\n".join(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({label: [asdict(r) for r in results] for label, results in runs.items()}, indent=2),
        encoding="utf-8",
    )
    return report


# ------------------------------------------------------------------------ cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the assistant's agentic modes.")
    parser.add_argument("--modes", nargs="+", default=["single", "multi"], choices=["single", "multi"])
    parser.add_argument("--fault", default="", help="inject a fault: tool_unavailable | malformed_retrieval | retrieval_timeout")
    parser.add_argument("--kinds", nargs="*", default=[], help="restrict to these query kinds")
    parser.add_argument("--ids", nargs="*", default=[], help="restrict to these case ids")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of queries (cost control)")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--provider", default=None, help="override the configured provider")
    parser.add_argument(
        "--keep-raw", type=int, default=None,
        help="COMPACTION_KEEP_RAW override; a large value effectively turns compaction off, "
             "which is how the technique's saving is measured",
    )
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--tag", default="", help="suffix for the configuration labels, e.g. 'fault'")
    args = parser.parse_args()

    cases = select(tuple(args.kinds), tuple(args.ids), args.limit)
    if not cases:
        raise SystemExit("no cases selected")
    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.keep_raw is not None:
        overrides["compaction_keep_raw"] = args.keep_raw

    runs: dict[str, list[Result]] = {}
    for mode in args.modes:
        label = f"{mode}+{args.tag}" if args.tag else mode
        print(f"\n== {label}: {len(cases)} queries ==")
        runs[label] = asyncio.run(run_suite(cases, mode, args.fault, args.concurrency, **overrides))

    out_dir = Path(args.out)
    report = render(runs, out_dir, Stack(**overrides).settings)
    print(f"\nwrote {out_dir / 'report.md'} and {out_dir / 'results.json'}\n")
    print(report.split("## Per-query results")[0])


if __name__ == "__main__":
    main()
