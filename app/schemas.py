from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="anonymous", max_length=100)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)


class Source(BaseModel):
    document: str
    chunk_id: int
    score: float
    text: str
    title: str = ""
    section: str = ""
    anchor: str = ""
    source: str = "root"


class ToolCall(BaseModel):
    name: str
    arguments: Any = None
    result: Any = None


class AssistantResponse(BaseModel):
    answer: str
    confidence: float = 0.5
    citations: list[str] = []
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    provider: str
    model: str
    cached: bool = False
    latency_ms: int = 0


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    embedder: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    index: dict[str, Any]
