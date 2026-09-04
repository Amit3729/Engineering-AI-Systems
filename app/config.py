from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    provider: str = "local"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    fallback_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    data_dir: str = "data"
    cache_ttl_seconds: int = 300
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
