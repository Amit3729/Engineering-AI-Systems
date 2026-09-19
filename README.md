# Engineering AI Assistant

A retrieval-augmented assistant built to the two-part brief: an applied-AI task
(LLM integration, prompt engineering, structured output, tool calling, a RAG
pipeline, local serving, containerisation) and a productionisation task (web UI,
inference optimisation, concurrency, caching, retries, rate limiting, fallback,
graceful degradation, deployment).

It runs with **no API key and no model download configured out of the box**:
`PROVIDER=local` answers from retrieved context so the retrieval stack can be
demonstrated before any credentials exist.

---

## Architecture

The diagram source is [architecture.mmd](architecture.mmd) (Mermaid — GitHub
renders it, or paste it into <https://mermaid.live>). It shows both agentic
modes and the coordination structure of the multi-agent one. The request path is:

```
Streamlit → FastAPI → rate limit → cache → pre-retrieval (Chroma) → agent_mode?
  single → tool-calling loop ⇄ {search_knowledge_base, calculator, current_time}
  multi  → planner → ⟨researcher × N in parallel⟩ ═compaction═> synthesiser
         → provider (openai | vllm) with retry and fallback
         → merged sources + structured JSON + usage + trajectory
```

| Layer | Choice |
| --- | --- |
| API | FastAPI, fully async handlers |
| Vector database | ChromaDB, persistent, HNSW, cosine |
| Embeddings | `BAAI/bge-small-en-v1.5` on **ONNX Runtime** via fastembed (384d) |
| Generation | Any OpenAI-compatible endpoint: OpenAI, or vLLM serving Llama 3 / Mistral |
| UI | Streamlit |
| Packaging | Docker + Docker Compose |

---

## W16: the agentic feature

*Reasoning, worked examples and weaknesses: [eval/ANALYSIS.md](eval/ANALYSIS.md).
Generated tables and the full failure log: [eval/results/report.md](eval/results/report.md).*

### a. Context engineering technique — compaction of retrieved context

One `search_knowledge_base` call over the CPython manual returns ~1.2k tokens of
prose. The W15 loop appended every result to one message list verbatim and never
removed it, so a question spanning two modules buried the question itself under
~4k tokens of near-duplicate documentation. Nothing crashed: the model began
answering from the page retrieved *most recently* rather than the most relevant.

Compaction runs where that text accumulates — not at the start, which discards
evidence unread, and not at the end, by which point every intermediate turn has
already paid:

1. **Inside a researcher's loop** — `compact_history()` in
   [app/agents.py](app/agents.py), at the top of every turn, rewrites all but the
   newest tool payload to `{"query": …, "documents": [...]}`. The agent still
   knows what it looked at; it stops paying for the prose twice.
2. **At the researcher → synthesiser boundary** — researchers return
   `(claim, citation)` findings. Chunk text rides out-of-band and is returned to
   the caller as `sources`, but never enters another model's prompt.

**Measured: 7% fewer tokens** than the same pattern with compaction off — modest,
because a researcher usually searches only once or twice. Its larger effect was
not tokens; see (3).

### b. Agentic pattern — multi-agent, single loop kept as a live baseline

**Planner → N parallel researchers → synthesiser.** `AGENT_MODE` and a
per-request `mode` select at runtime, so the baseline is a configuration, not a
deleted branch. Chosen for **context isolation**, targeting **context
saturation** — each researcher owns its message list, so raw chunks live and die
inside it — and for **parallelization**, removing a **sequential bottleneck**:
"aware datetime, then convert it" needs `datetime` and `zoneinfo`, and neither
lookup depends on the other. Not for **specialization**; the roles differ by
prompt and tool access, not domain skill, so this is not skill dilution solved.

**Measured: 85% completion against the baseline's 65%, for 1.22x the tokens.**
The baseline's losses are almost all one structural behaviour — handed
pre-retrieved context it often answers on a *single turn without searching at
all*, at confidence 0.90–1.00. A researcher cannot: issuing a lookup is its only
action.

It does **not** solve the **self-verification paradox** — the synthesiser sees
findings, never source text, so it can only propagate a claim, not check it. A
verifier agent is the fix and is not implemented. Nor is the pattern always
right: an empty planner task list routes arithmetic back to the single loop
(3 of 20 queries), so one planner turn is its entire overhead there.

### c. Evaluation harness

From scratch in [eval/](eval/), no framework. 20 cases (`single_hop`,
`multi_hop`, `arithmetic`, `unanswerable`), each naming documents that exist in
the index so a miss is a retrieval failure, not a typo. Per query: **completion**
(expected documents cited and required content present; for an unanswerable case,
that it declined), **tool-call correctness** (expected tools called, forbidden
ones avoided, arguments validated against each tool's JSON schema),
**trajectory length**, **tokens**, and a **failure class** — *hard* (no usable
answer), *soft* (looks fine, wrong on its own terms), *cascading soft* (an
upstream step returned nothing usable and the agent answered confidently anyway).

Pre-retrieved chunks count as sources but not as grounding — finding the page is
the retriever's job, citing it is the agent's. Tool errors are labelled
`arguments` or `infrastructure` so an injected fault is not charged to the agent.

```bash
python -m eval.run_all      # six configurations, one report; local model by default
```

**84 runs, `qwen2.5:7b` on Ollama, no API cost:**

| configuration | n | completion | tool-calls | turns | mean tokens | hard | soft | cascading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single` | 20 | 65% | 80% | 2.1 | 6,337 | 0 | 8 | 0 |
| `multi` | 20 | **85%** | 85% | 4.8 | 7,727 | 0 | 5 | 0 |
| `multi+no-compaction` | 20 | 80% | 90% | 4.7 | 8,332 | 0 | 4 | 1 |
| `single+malformed_retrieval` | 6 | 0% | 100% | 3.7 | 2,825 | 0 | 0 | **6** |
| `multi+malformed_retrieval` | 6 | 0% | 100% | 4.0 | 2,860 | 0 | 5 | **1** |
| `multi+tool_unavailable` | 6 | 0% | 0% | 2.2 | 1,058 | 0 | 6 | 0 |
| `multi+retrieval_timeout` | 6 | 0% | 100% | 4.2 | 2,739 | 0 | 6 | 0 |

Trajectory length is reasonable against its ceiling: 2.1 turns of a possible 4
for the loop, 4.8 of a possible 6 for the team.

### Additional requirements

**1. Skill vs. Agent.** The *researcher* could not have been a Skill — a Skill is
instructions injected into the calling context, and the point of this capability
is keeping its chunks *out* of that context. The *planner* plausibly could have
been one, and was not, because that returns the decomposition and the results to
a single window, which is the saturation being avoided.

**2. Token and cost accounting.** `Usage` in [app/llm.py](app/llm.py) sums
`prompt_tokens` / `completion_tokens` / `model_calls` across every upstream call
including sub-agents, returned on `/chat` and recorded per query: **7,727 tokens
for the team against 6,337 for the baseline — 1.22x for +20 points of
completion** — and 8,332 with compaction off, which the 7% is measured against.

**3. Failure injection.** `tool_unavailable`, `malformed_retrieval` (rows with
unreadable text and empty metadata) and `retrieval_timeout`. The retrieval faults
live in the `Retriever`, not the tool, so pre-retrieval degrades *with* it — a
fault only half the system sees is not a useful test.

| fed corrupted retrieval | confident answer on bad evidence | admitted the gap |
| --- | --- | --- |
| `single+malformed_retrieval` | **6 / 6** | 0 |
| `multi+malformed_retrieval` | **1 / 6** | 5 |

Handed mojibake the single loop wrote fluent documentation at confidence
0.70–0.80 every time, once recommending **pytz** — not in the corpus, never
retrieved. The team answered *"the findings do not provide information about…"*
at confidence 0.00 on five of six. The cause is the compaction boundary: the
synthesiser sees `(no supporting text was found)`, not the garbage. Text that
*looks* like documentation invites a model to keep writing; a statement of
absence does not. Isolating context protected truthfulness, not just budget.
`tool_unavailable` and `retrieval_timeout` each produced **0 cascading of 6**.

**4. Tool vs. agent boundary.** The retriever is multi-step and stateful — a
persisted Chroma collection, an HNSW index and a manifest, with an ingest path
that mutates all three — and is modelled as a **bounded tool call**. Its state is
not *conversational*: it is the same index for every user and every turn, so
there is nothing for a negotiated protocol to carry between calls, and one
validated request/response keeps model access inside the allow-list. An
agent-to-agent framing would only earn its complexity if the retriever could ask
clarifying questions or re-index on its own; it can do neither, and should not.
Ingestion — 22 minutes of embedding, the genuinely long-running stateful
operation — is kept off the tool surface entirely and exposed only as an operator
endpoint (`POST /ingest`), so no model can trigger a rebuild.

---

## Quickstart

```bash
cp .env.example .env
pip install -r requirements.txt

uvicorn app.main:app --reload          # API  → http://localhost:8000/docs
streamlit run streamlit_app.py         # UI   → http://localhost:8501
```

### Building the index — do this before the first run

**`data/chroma/` is not in this repository.** It is a 197 MB generated artefact,
so it is deliberately left out and has to be built once on your machine:

```bash
curl -X POST 'http://localhost:8000/ingest?force=true'   # or just start the API
```

Starting the API builds it automatically if it is missing. Budget for it: the
first start also downloads the ~130 MB ONNX embedding model, and embedding 544
documents into 13,657 chunks takes **roughly 22 minutes** on a 4-performance-core
laptop. Every later start reuses both — the corpus fingerprint is unchanged, so
ingestion is skipped and the API is serving in seconds.

The corpus itself (`data/python-3.14-docs-html/`, the 74 MB Sphinx build) is also
untracked; download it from <https://docs.python.org/3/download.html> and unpack
it there. Without it the assistant still runs, on the three project notes alone.

Once the index exists, try:

- `How does chunk overlap work?` — exercises retrieval
- `What is 128 * 47?` — exercises the calculator tool
- `What happens when the primary provider fails?` — exercises retrieval over the serving docs

### Docker

```bash
cp .env.example .env
docker compose up --build
```

API on `:8000`, UI on `:8501`. The embedding weights are baked into the image at
build time, so a cold container serves its first request without downloading
anything. The UI waits on the API's health check before starting.

---

## Configuration

Every setting in [.env.example](.env.example) is an environment variable. The
ones that matter most:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROVIDER` | `local` | `local` (keyless demo), `ollama` (free, on your machine), `openai`, or `vllm` |
| `FALLBACK_PROVIDER` | `openai` | Used when the primary fails; `none` disables |
| `EMBEDDING_PROVIDER` | `onnx` | `onnx`, `openai`, or `hash` (test double) |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `180` / `40` | Chunking |
| `TOP_K` / `MIN_SCORE` | `4` / `0.15` | Retrieval breadth and relevance floor |
| `MAX_TOOL_ITERATIONS` | `4` | Tool-loop budget |
| `AGENT_MODE` | `single` | `single` (W15 loop) or `multi` (planner + researchers + synthesiser) |
| `MAX_RESEARCHERS` | `3` | Parallel researchers the planner may fan out to |
| `RESEARCHER_MAX_ITERATIONS` | `3` | Turns per researcher: two searches plus a forced report |
| `RESEARCHER_TOP_K` | `5` | Chunks per researcher search, when the model does not choose |
| `COMPACTION_KEEP_RAW` | `1` | Tool results kept verbatim; older ones are compacted to citations |
| `FAULT_INJECTION` | `` | `tool_unavailable`, `malformed_retrieval`, or `retrieval_timeout` |
| `MAX_RETRIES` | `3` | Attempts per provider before failing over |
| `RATE_LIMIT_REQUESTS` / `..._WINDOW_SECONDS` | `30` / `60` | Per-IP sliding window |
| `MAX_CONCURRENT_LLM_CALLS` | `8` | Upstream concurrency ceiling |

### Local model with Ollama — no key, no cost

```bash
ollama pull qwen2.5:7b
PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

Ollama serves the OpenAI protocol on `/v1`, so it needs no separate adapter —
only a different `base_url` and a placeholder key. This is the configuration the
evaluation in this README was produced with, so every number below is
reproducible without an account or a bill.

### Hosted model

```bash
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Local open-source model with vLLM

```bash
docker compose --profile local-model up --build
```

Then set `PROVIDER=vllm`. The vLLM service is started with
`--enable-auto-tool-choice --tool-call-parser llama3_json`, which is required
for function calling to work against a Llama-family model. It needs a
CUDA-capable Docker host; weights are mounted from the host Hugging Face cache
rather than copied into the image.

---

## RAG pipeline

**The corpus.** Two things live under `DATA_DIR`: a few Markdown notes about this
project, and the complete CPython 3.14 documentation as a Sphinx HTML build
(579 pages, 74 MB). Excluded from the index:

- **Build artefacts** — `genindex*`, `search.html`, `py-modindex.html`,
  `_static/`, `_downloads/`, `_images/`, and the redirect stubs this build ships
  where a page moved (`library/stdtypes.html` is a 476-byte `<meta refresh>` to
  `builtins/stdtypes.html`).
- **`whatsnew/changelog.html`** — one page, but **2,878 chunks, 17% of the whole
  index**, of entries like *"gh-12345: Fix a crash in json.dumps()"*. Release-note
  fragments match almost any "how does X work" question on surface wording while
  answering none of them: before this exclusion, a query about `json.dumps`
  ranked the changelog *above* `library/json.html`. The narrative
  `whatsnew/3.x.html` pages are kept — they are real documentation.

That leaves **544 files**. Exclusion patterns match anywhere under `DATA_DIR`,
not just directly beneath it, because the CPython build sits in its own
directory.

**Ingestion and chunking.** `.md` and `.txt` are split on paragraph boundaries,
then packed into 180-word chunks with a 40-word sliding overlap so a sentence
spanning a boundary stays retrievable from either side.

HTML goes through [app/documents.py](app/documents.py) first, and is chunked
*per section* rather than per page. `library/datetime.html` is 12.8k words; a
blind 180-word window straddling two unrelated APIs retrieves badly for both,
while Sphinx has already marked the right boundaries as `<section id="...">`.
Each section keeps its heading trail (`datetime — Basic date and time types >
Aware and naive objects`) and its anchor, so a citation points at
`library/json.html#json.dumps` rather than at the page as a whole. Two details
matter for retrieval quality:

- **Signatures are rendered without separators.** `get_text(" ")` over Sphinx's
  span soup produces `json . dump ( obj , fp`, which no longer matches the symbol
  anyone would search for. Those subtrees are flattened to
  `json.dump(obj, fp, *, skipkeys=False)` first.
- **Code blocks keep their lines**, fenced, so `>>> 2 + 2` survives as an example
  instead of collapsing into the surrounding prose.

A chunk's metadata carries `source` — the corpus it came from — and
`search_knowledge_base` takes an optional `source` filter, so a question about
this project's serving stack is not drowned by 16.5k chunks of CPython manual.

**Fingerprinting.** The index records a hash over every document's bytes plus the
chunk size, overlap, embedding-model name, and parser version. An unchanged corpus skips
re-embedding entirely on restart; changing the embedding model forces a rebuild,
because vectors from two models are not comparable. A rebuild recreates the
collection so deleted documents actually leave the index.

**Embeddings.** `BAAI/bge-small-en-v1.5` executed by ONNX Runtime — no API key,
no PyTorch. Swappable for hosted `text-embedding-3-small`, or a deterministic
hashing stand-in used by the tests. Sequences are embedded 32 at a time and
upserted in the same batches: every sequence in an ONNX batch is padded to the
longest one in it, so a wide batch cost 5.7 GB of activations for no extra
throughput, against 1.5 GB at batch 32. A full build of this corpus takes roughly
half an hour on a 4-performance-core laptop and is then skipped on every restart.

**Storage and search.** ChromaDB in persistent mode with an HNSW index over
cosine distance. Distances are converted to similarity (`1 - distance`) and
anything below `MIN_SCORE` is dropped rather than padding the prompt with noise.

---

## Tool calling

Tools are declared as OpenAI function schemas in [app/tools.py](app/tools.py) and
**the model chooses which to call**. The loop appends each `tool` result to the
message list and re-invokes the model until it stops requesting tools.

| Tool | Purpose |
| --- | --- |
| `search_knowledge_base` | Semantic search; the model can re-query with different wording, and can restrict to one corpus with `source` |
| `calculator` | Arithmetic over a restricted AST — no imports, attributes, or calls |
| `current_time` | UTC timestamp |

Safety and robustness properties:

- The registry is an **allow-list**; an invented tool name returns an error object.
- Arguments are parsed and validated; a malformed call returns an error *to the model* so it can correct itself instead of failing the request.
- The loop is bounded by `MAX_TOOL_ITERATIONS`, and the final turn is issued with tools disabled so an answer must be produced.
- Tool execution runs in a worker thread, keeping the event loop free.

Retrieval happens twice by design: a cheap pre-retrieval seeds the prompt (saving
a round-trip), and the model may still call `search_knowledge_base` for more.
Both result sets are merged and de-duplicated into `sources`.

---

## Structured output

Every call sets `response_format={"type": "json_object"}` and the system prompt
fixes the schema. The response is validated by Pydantic before it leaves the API:

```json
{
  "answer": "vLLM raises throughput with paged attention and continuous batching.",
  "confidence": 0.9,
  "citations": ["serving.md"],
  "sources": [{"document": "serving.md", "chunk_id": 0, "score": 0.735, "text": "..."}],
  "tool_calls": [{"name": "search_knowledge_base", "arguments": {...}, "result": {...}}],
  "provider": "openai",
  "model": "gpt-4o-mini",
  "cached": false,
  "latency_ms": 344,
  "agent_mode": "multi",
  "iterations": 5,
  "usage": {"prompt_tokens": 6412, "completion_tokens": 498, "total_tokens": 6910, "model_calls": 6},
  "plan": [{"task": "...", "source": "python-3.14-docs-html", "findings": 3, "iterations": 2, "tokens": 2180, "gaps": "", "error": ""}]
}
```

A reply that is not valid JSON is passed through with a lowered confidence rather
than failing the request.

---

## Performance

- **Async end to end.** Handlers are `async`; embedding, vector search, ingestion, and tool execution run via `asyncio.to_thread` so CPU work never blocks the event loop.
- **Bounded upstream concurrency.** A semaphore caps in-flight provider calls so a burst queues instead of stampeding.
- **Caching.** Bounded TTL + LRU keyed on question and sampling parameters. Bounded matters: an unbounded dict keyed on user input is a memory leak.
- **Ingestion is skipped** when the corpus fingerprint is unchanged.
- **ONNX Runtime** for embeddings: quantised graph, no PyTorch in the image, CPU/CoreML execution providers.

Measured locally (24 concurrent requests against a stub provider, one uvicorn
worker): **~100 req/s, p50 184 ms, p95 199 ms**; cached repeats return in ~0 ms.

### On ONNX conversion

ONNX **is** applied — to the embedding model, which is the part of this system
that actually runs locally. The generation model is not converted, and that is a
deliberate call rather than an omission: generation is served either by a hosted
API or by vLLM, whose throughput comes from paged attention and continuous
batching. Exporting a 3B+ decoder to ONNX would *lose* those and regress
throughput. Scale out with `--workers` or more replicas.

---

## Reliability

| Mechanism | Behaviour |
| --- | --- |
| Retry | Exponential backoff with jitter, `MAX_RETRIES` per provider |
| Selective retry | Only 408/409/425/429/5xx, timeouts, connection errors. A 401 fails immediately instead of burning the budget |
| Fallback | A configured second provider serves the request; the response reports which one answered |
| Rate limiting | Per-IP sliding window, `429` with `Retry-After`; `/health` is exempt so orchestrators can still probe a throttled instance |
| Tool failure | Returned to the model as an error object, never a 500 |
| Retrieval failure | Degrades to a context-free answer, request still succeeds |
| Bad JSON | Passed through with lowered confidence |
| Total provider failure | `503` with the underlying cause |

---

## API

```bash
curl http://localhost:8000/health
curl -X POST 'http://localhost:8000/ingest?force=true'
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"What does vLLM provide?","temperature":0.2,"top_p":0.9}'
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Status, provider, index stats, cache stats |
| `POST /ingest?force=` | Rebuild the index (`force=true` bypasses the fingerprint) |
| `POST /chat` | Ask a question; optional `mode` of `single` or `multi` |

---

## Tests

```bash
pytest -q        # 58 tests in the repository, 103 on the author's machine
```

Test files are excluded from version control by `.gitignore`, so a clone carries
the 58 tests in the originally tracked modules; the 45 added for W16 — Sphinx
extraction, the multi-agent team, and the harness's own scoring — live alongside
them locally. The suite runs offline: no API key, no network, no weight download
(`EMBEDDING_PROVIDER=hash`, stubbed provider clients). It covers chunking and
overlap, ingest/search round-trips and re-index behaviour, tool-schema validity,
calculator sandboxing, allow-list rejection, the multi-turn tool loop, retry and
fallback, JSON parsing and degradation, the rate limiter, and the API contract.
W16 adds coverage of Sphinx extraction (section splitting, signature flattening,
artefact exclusion), injected retrieval faults, the multi-agent team (fan-out,
compaction, context isolation, degradation under a withdrawn tool), and the
harness's own scoring and failure classification.

---

## Layout

```
app/
  main.py        FastAPI app, middleware, endpoints
  config.py      Environment-backed settings
  llm.py         Provider adapters, tool-calling loop, retry, fallback, token accounting
  agents.py      W16: planner, parallel researchers, synthesiser, compaction
  tools.py       Function schemas, allow-listed registry, fault injection
  documents.py   W16: Sphinx HTML -> sections
  retrieval.py   Chunking, ingestion, ChromaDB search
  embeddings.py  ONNX / OpenAI / hashing backends
  cache.py       Bounded TTL + LRU cache
  schemas.py     Request and response models
eval/
  dataset.py     20 evaluation cases
  harness.py     Scoring, failure taxonomy, report generation
  run_all.py     Six-configuration driver
  ANALYSIS.md    Reading of the results
  results/       report.md and results.json (generated)
data/            Corpus; data/chroma/ is the persisted index
tests/           Offline test suite
```
