# Engineering AI Assistant

A production-shaped AI assistant reference implementation for the two-part problem set. It includes document ingestion and chunking, deterministic local vector retrieval, OpenAI-compatible hosted or vLLM model adapters, JSON structured output, an allow-listed tool, a Streamlit UI, retries, provider fallback, rate limiting, and TTL caching.

## Architecture

The architecture source is in [architecture.mmd](architecture.mmd). The request path is:

`Streamlit -> FastAPI -> rate limit -> cache -> retrieval -> LLM/tool -> structured response`

Documents in `data/` are split into 180-word chunks and stored with normalized hashing embeddings in `data/index.json`. This lightweight index avoids requiring a separate database for the assignment demo. Replace `app/retrieval.py` with pgvector, Qdrant, or Chroma for a larger corpus.

## Run locally

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The UI is at `http://localhost:8501`; API docs are at `http://localhost:8000/docs`. With the default `PROVIDER=local`, no key or model download is needed. Try `calculate 12 * 4` to exercise tool calling, or ask about RAG and vLLM to exercise retrieval.

## Hosted or local model

Set `PROVIDER=openai` and `OPENAI_API_KEY` for OpenAI. The adapter uses `response_format={"type":"json_object"}` and the Pydantic response schema enforces the external API contract. Set `PROVIDER=vllm` to use the OpenAI-compatible local endpoint configured by `VLLM_BASE_URL` and `VLLM_MODEL`.

The optional GPU service can be started with:

```bash
docker compose --profile local-model up --build
```

This requires a CUDA-capable Docker host and a Hugging Face model-access token where applicable. Model weights are intentionally mounted from the host cache rather than copied into the image.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The API is at `http://localhost:8000` and the UI is at `http://localhost:8501`. The compose health check prevents the UI from starting before the API is ready.

## API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/chat \
	-H 'content-type: application/json' \
	-d '{"question":"What does vLLM provide?","temperature":0.2,"top_p":0.9}'
```

The chat response is JSON with `answer`, `sources`, `tool_calls`, `provider`, and `cached`. Requests are limited per client IP, transient provider failures retry with exponential backoff, and non-primary providers are used as a fallback when configured.

## Validation and optimization

```bash
pytest -q
```

ONNX conversion is not applied to the generation model because this implementation consumes remote OpenAI-compatible APIs or a vLLM server; there is no local PyTorch checkpoint in this repository to convert. vLLM is the applicable serving optimization for the local path, while async FastAPI handlers, bounded provider retries, cache hits, and stateless containers improve throughput and latency.
