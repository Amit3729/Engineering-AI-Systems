import hashlib
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from .config import get_settings
from .llm import LLMService
from .retrieval import Retriever
from .schemas import AssistantResponse, ChatRequest, IngestResponse, Source
from .tools import run_tools

settings = get_settings()
retriever = Retriever(settings)
llm = LLMService(settings)
cache: dict[str, tuple[float, AssistantResponse]] = {}
requests_by_ip: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    retriever.ingest()
    yield


app = FastAPI(title="Engineering AI Assistant", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    now = time.monotonic()
    bucket = requests_by_ip[request.client.host if request.client else "unknown"]
    while bucket and bucket[0] <= now - settings.rate_limit_window_seconds:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; retry shortly")
    bucket.append(now)
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": settings.provider}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    documents, chunks = retriever.ingest()
    return IngestResponse(documents=documents, chunks=chunks)


@app.post("/chat", response_model=AssistantResponse)
async def chat(request: ChatRequest) -> AssistantResponse:
    key = hashlib.sha256(f"{request.question}|{request.temperature}|{request.top_p}".encode()).hexdigest()
    cached = cache.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1].model_copy(update={"cached": True})
    matches = retriever.search(request.question)
    sources = [Source(**match) for match in matches if match["score"] > 0]
    context = "\n".join(f"[{source.document}] {source.text}" for source in sources)
    tool_calls, tool_context = run_tools(request.question)
    try:
        result, provider = await llm.complete(request.question, context, tool_context, request.temperature, request.top_p)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Assistant temporarily unavailable: {error}") from error
    response = AssistantResponse(answer=str(result.get("answer", "No answer returned.")), sources=sources, tool_calls=tool_calls, provider=provider)
    cache[key] = (time.monotonic() + settings.cache_ttl_seconds, response)
    return response
