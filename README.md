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
renders it, or paste it into <https://mermaid.live>). The request path is:

```
Streamlit → FastAPI → rate limit → cache → pre-retrieval (Chroma)
         → tool-calling loop ⇄ {search_knowledge_base, calculator, current_time}
         → provider (openai | vllm) with retry and fallback
         → merged sources + structured JSON
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

## Quickstart

```bash
cp .env.example .env
pip install -r requirements.txt

uvicorn app.main:app --reload          # API  → http://localhost:8000/docs
streamlit run streamlit_app.py         # UI   → http://localhost:8501
```

The first start downloads the ~130 MB ONNX embedding model and builds the index.
Subsequent starts reuse both. Try:

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
| `PROVIDER` | `local` | `local` (keyless demo), `openai`, or `vllm` |
| `FALLBACK_PROVIDER` | `openai` | Used when the primary fails; `none` disables |
| `EMBEDDING_PROVIDER` | `onnx` | `onnx`, `openai`, or `hash` (test double) |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `180` / `40` | Chunking |
| `TOP_K` / `MIN_SCORE` | `4` / `0.15` | Retrieval breadth and relevance floor |
| `MAX_TOOL_ITERATIONS` | `4` | Tool-loop budget |
| `MAX_RETRIES` | `3` | Attempts per provider before failing over |
| `RATE_LIMIT_REQUESTS` / `..._WINDOW_SECONDS` | `30` / `60` | Per-IP sliding window |
| `MAX_CONCURRENT_LLM_CALLS` | `8` | Upstream concurrency ceiling |

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

**Ingestion and chunking.** `.md` and `.txt` files under `DATA_DIR` are split on
paragraph boundaries, then packed into 180-word chunks with a 40-word sliding
overlap so a sentence spanning a boundary stays retrievable from either side.

**Fingerprinting.** The index records a hash over every document's bytes plus the
chunk size, overlap, and embedding-model name. An unchanged corpus skips
re-embedding entirely on restart; changing the embedding model forces a rebuild,
because vectors from two models are not comparable. A rebuild recreates the
collection so deleted documents actually leave the index.

**Embeddings.** `BAAI/bge-small-en-v1.5` executed by ONNX Runtime — no API key,
no PyTorch. Swappable for hosted `text-embedding-3-small`, or a deterministic
hashing stand-in used by the tests.

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
| `search_knowledge_base` | Semantic search; the model can re-query with different wording |
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
  "latency_ms": 344
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
| `POST /chat` | Ask a question |

---

## Tests

```bash
pytest -q        # 44 tests
```

The suite runs offline: no API key, no network, no weight download
(`EMBEDDING_PROVIDER=hash`, stubbed provider clients). It covers chunking and
overlap, ingest/search round-trips and re-index behaviour, tool-schema validity,
calculator sandboxing, allow-list rejection, the multi-turn tool loop, retry and
fallback, JSON parsing and degradation, the rate limiter, and the API contract.

---

## Layout

```
app/
  main.py        FastAPI app, middleware, endpoints
  config.py      Environment-backed settings
  llm.py         Provider adapters, tool-calling loop, retry, fallback
  tools.py       Function schemas, allow-listed registry
  retrieval.py   Chunking, ingestion, ChromaDB search
  embeddings.py  ONNX / OpenAI / hashing backends
  cache.py       Bounded TTL + LRU cache
  schemas.py     Request and response models
data/            Corpus; data/chroma/ is the persisted index
tests/           Offline test suite
```
