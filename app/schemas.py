from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="anonymous", max_length=100)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    # "single" (the W15 loop) or "multi" (planner + researchers + synthesiser).
    # None uses the configured default, so the harness can A/B the two modes.
    mode: Literal["single", "multi"] | None = None


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


class ResearchTask(BaseModel):
    """One planner-issued lookup, with what it cost and what it found."""

    task: str
    source: str = ""
    findings: int = 0
    iterations: int = 0
    tokens: int = 0
    gaps: str = ""
    error: str = ""


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
    agent_mode: str = "single"
    # Trajectory length and token spend, so a caller can see the coordination cost.
    iterations: int = 0
    usage: dict[str, int] = {}
    plan: list[ResearchTask] = []


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    embedder: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    index: dict[str, Any]
