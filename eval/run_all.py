"""The full evaluation, in one process, into one report.

Six configurations, chosen so that each number in the report answers a specific
question rather than just existing:

* ``single`` vs ``multi`` - what does the extra coordination buy, and cost?
* ``multi`` vs ``multi+no-compaction`` - the same pattern with the context
  technique switched off, which is the only honest way to attribute a token
  saving to it.
* the three fault configurations - does the assistant notice broken evidence?
  These run a retrieval-dependent subset, because they are a behaviour probe and
  not a benchmark; running all twenty would cost more and show the same thing.

    python -m eval.run_all
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .dataset import select
from .harness import Stack, render, run_suite

# Queries that cannot be answered without working retrieval, which is what the
# fault configurations are probing.
FAULT_PROBE_IDS = ("json-indent", "lru-cache", "tz-convert", "dataclass-json", "project-serving", "mmap")

NO_COMPACTION = 99  # larger than any researcher's tool-result count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="ollama", help="ollama (free, local) | openai | vllm")
    # Ollama serves one request at a time by default, so extra concurrency only
    # deepens the queue. Two keeps the model fed without starving a long request.
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="cap the main suites (cost control)")
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args()

    overrides = {"provider": args.provider}
    main_cases = select(limit=args.limit)
    probe_cases = select(ids=FAULT_PROBE_IDS)

    plan = [
        ("single", "single", "", main_cases, {}),
        ("multi", "multi", "", main_cases, {}),
        ("multi+no-compaction", "multi", "", main_cases, {"compaction_keep_raw": NO_COMPACTION}),
    ]
    if not args.skip_faults:
        plan += [
            ("single+malformed_retrieval", "single", "malformed_retrieval", probe_cases, {}),
            ("multi+malformed_retrieval", "multi", "malformed_retrieval", probe_cases, {}),
            ("multi+tool_unavailable", "multi", "tool_unavailable", probe_cases, {}),
            ("multi+retrieval_timeout", "multi", "retrieval_timeout", probe_cases, {}),
        ]

    runs = {}
    for label, mode, fault, cases, extra in plan:
        print(f"\n== {label}: {len(cases)} queries ==")
        runs[label] = asyncio.run(run_suite(cases, mode, fault, args.concurrency, **(overrides | extra)))

    out_dir = Path(args.out)
    report = render(runs, out_dir, Stack(**overrides).settings)
    print(f"\nwrote {out_dir / 'report.md'} and {out_dir / 'results.json'}\n")
    print(report.split("## Per-query results")[0])


if __name__ == "__main__":
    main()
