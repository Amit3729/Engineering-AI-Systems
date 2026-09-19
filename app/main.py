import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .agents import ResearchTeam
from .cache import TTLCache
from .config import get_settings
from .llm import LLMService
from .retrieval import Retriever
from .schemas import (
    AssistantResponse,
    ChatRequest,
    HealthResponse,
    IngestResponse,
    ResearchTask,
    Source,
    ToolCall,
)
from .tools import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
retriever = Retriever(settings)
registry = ToolRegistry(retriever, settings)
llm = LLMService(settings, registry)
team = ResearchTeam(settings, registry, llm)
cache = TTLCache(settings.cache_max_entries, settings.cache_ttl_seconds)
requests_by_ip: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Embedding the corpus is CPU work; keep it off the event loop.
    documents, chunks = await asyncio.to_thread(retriever.ingest)
    logger.info("index ready: %d documents, %d chunks, embedder=%s", documents, chunks, retriever.embedder.name)
    yield


app = FastAPI(title="Engineering AI Assistant", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in {"/health", "/docs", "/openapi.json"}:
        return await call_next(request)

    now = time.monotonic()
    window_start = now - settings.rate_limit_window_seconds
    client = request.client.host if request.client else "unknown"
    bucket = requests_by_ip[client]
    while bucket and bucket[0] <= window_start:
        bucket.popleft()

    if len(bucket) >= settings.rate_limit_requests:
        retry_after = max(1, int(bucket[0] + settings.rate_limit_window_seconds - now))
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded; retry shortly"},
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)
    # Drop idle clients so the limiter's own bookkeeping stays bounded.
    if len(requests_by_ip) > 10_000:
        for key in [key for key, value in requests_by_ip.items() if not value]:
            del requests_by_ip[key]
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        index = await asyncio.to_thread(retriever.stats)
    except Exception as error:  # degraded, not dead
        index = {"error": str(error)}
    return HealthResponse(
        status="ok",
        provider=settings.provider,
        index=index
        | {
            "cache": cache.stats(),
            "agent_mode": settings.agent_mode,
            "sources": await asyncio.to_thread(retriever.sources),
            "fault_injection": settings.fault_injection or "none",
        },
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(force: bool = False) -> IngestResponse:
    documents, chunks = await asyncio.to_thread(retriever.ingest, force)
    return IngestResponse(documents=documents, chunks=chunks, embedder=retriever.embedder.name)


def _merge_sources(*groups: list[dict[str, Any]]) -> list[Source]:
    """De-duplicate chunks across pre-retrieval and model-issued searches."""
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for group in groups:
        for match in group:
            key = (match["document"], match["chunk_id"])
            if key not in best or match["score"] > best[key]["score"]:
                best[key] = match
    return [Source(**match) for match in sorted(best.values(), key=lambda item: item["score"], reverse=True)]


@app.post("/chat", response_model=AssistantResponse)
async def chat(request: ChatRequest) -> AssistantResponse:
    started = time.perf_counter()
    mode = request.mode or settings.agent_mode
    key = hashlib.sha256(f"{request.question}|{request.temperature}|{request.top_p}|{mode}".encode()).hexdigest()

    hit = cache.get(key)
    if hit is not None:
        return hit.model_copy(update={"cached": True, "latency_ms": int((time.perf_counter() - started) * 1000)})

    try:
        matches = await asyncio.to_thread(retriever.search, request.question)
    except Exception:
        # Degrade to a context-free answer rather than failing the request.
        logger.exception("retrieval failed; continuing without pre-retrieved context")
        matches = []

    prefetched = [match for match in matches if match["score"] >= settings.min_score]
    context = "\n\n".join(f"[{match['document']}] {match['text']}" for match in prefetched)

    try:
        if mode == "multi":
            # The team does its own retrieval per sub-question, so the prefetched
            # context is only a fallback for the no-research path.
            completion = await team.answer(request.question, context, request.temperature, request.top_p)
        else:
            completion = await llm.complete(request.question, context, request.temperature, request.top_p)
    except Exception as error:
        logger.exception("all providers failed")
        raise HTTPException(status_code=503, detail=f"Assistant temporarily unavailable: {error}") from error

    response = AssistantResponse(
        answer=completion.answer,
        confidence=completion.confidence,
        citations=completion.citations,
        sources=_merge_sources(prefetched, completion.retrieved),
        tool_calls=[ToolCall(**call) for call in completion.tool_calls],
        provider=completion.provider,
        model=completion.model,
        latency_ms=int((time.perf_counter() - started) * 1000),
        agent_mode=completion.agent_mode,
        iterations=completion.iterations,
        usage=completion.usage.as_dict(),
        plan=[ResearchTask(**task) for task in completion.plan],
    )
    cache.set(key, response)
    return response
