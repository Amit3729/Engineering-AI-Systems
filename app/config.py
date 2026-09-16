from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


class Settings(BaseSettings):
    # Generation provider: "local" (no key, deterministic demo), "openai", or "vllm".
    provider: str = "local"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    fallback_provider: str = "openai"
    fallback_model: str = "gpt-4o-mini"
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Embeddings: "onnx" (ONNX Runtime, no API key), "openai", or "hash" (test double).
    embedding_provider: str = "onnx"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_cache_dir: str = ".cache/fastembed"

    # Retrieval
    data_dir: str = "data"
    chroma_dir: str = "data/chroma"
    collection_name: str = "knowledge"
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    top_k: int = 4
    min_score: float = 0.15

    # Tool calling
    max_tool_iterations: int = 4

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
