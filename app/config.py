from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


class Settings(BaseSettings):
    # Generation provider: "local" (no key, deterministic demo), "openai", "vllm",
    # or "ollama" (a local model, no key and no cost).
    provider: str = "local"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    fallback_provider: str = "openai"
    fallback_model: str = "gpt-4o-mini"
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    # Ollama speaks the OpenAI protocol on /v1, so it needs no separate adapter.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    # Embeddings: "onnx" (ONNX Runtime, no API key), "openai", or "hash" (test double).
    embedding_provider: str = "onnx"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_cache_dir: str = ".cache/fastembed"
    # Sequences per ONNX forward pass; wider batches cost memory, not throughput.
    embedding_batch_size: int = 32

    # Retrieval
    data_dir: str = "data"
    chroma_dir: str = "data/chroma"
    collection_name: str = "knowledge"
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    top_k: int = 4
    min_score: float = 0.15
    # Chunks embedded and written per batch during ingestion.
    ingest_batch_size: int = 512
    # Extra comma-separated globs (relative to data_dir) to keep out of the index.
    excluded_globs: str = ""

    # Tool calling
    max_tool_iterations: int = 4

    # Agentic pattern: "single" is the W15 tool-calling loop; "multi" adds a
    # planner, parallel researchers with isolated context, and a synthesiser.
    agent_mode: str = "single"
    max_researchers: int = 3
    # 3 turns = two search rounds plus a forced report. Two turns would allow
    # only one search, which is also one too few for compaction to ever fire.
    researcher_max_iterations: int = 3
    researcher_top_k: int = 5
    # Retrieval payloads older than this many tool results are compacted to
    # citations only, so a researcher's window does not fill with raw chunks.
    compaction_keep_raw: int = 1

    # Deliberate fault for the failure-injection test. One of "", "tool_unavailable",
    # "malformed_retrieval", "retrieval_timeout".
    fault_injection: str = ""
    tool_timeout_seconds: float = 20.0

    # Reliability / performance
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 512
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    max_concurrent_llm_calls: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
