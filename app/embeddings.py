"""Embedding backends.

Three interchangeable implementations sit behind one protocol so the vector
store never has to care where vectors come from:

* ``onnx``   - BAAI/bge-small-en-v1.5 executed by ONNX Runtime via fastembed.
               No API key, no PyTorch, quantised graph, CPU/CoreML providers.
* ``openai`` - hosted text-embedding-3-small, for parity with a managed stack.
* ``hash``   - deterministic hashing bag-of-words. Not a real model; it exists
               so tests and CI can run without downloading weights.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class OnnxEmbedder:
    """Sentence embeddings from an ONNX graph run by ONNX Runtime."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding

        self.name = f"onnx:{model_name}"
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dimensions = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed batches internally, so a whole corpus goes through one call.
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [vector.tolist() for vector in self._model.query_embed([text])][0]


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.dimensions = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            batch = self._client.embeddings.create(model=self._model, input=texts[start : start + 128])
            vectors.extend(item.embedding for item in batch.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._client.embeddings.create(model=self._model, input=[text]).data[0].embedding


class HashingEmbedder:
    """Zero-dependency stand-in. Deterministic, offline, and deliberately weak."""

    def __init__(self, dimensions: int = 384):
        self.name = "hash:sha256"
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                matrix[row, int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0 if digest[4] % 2 else -1.0
        return _normalize(matrix).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=4)
def _build(provider: str, model: str, cache_dir: str, api_key: str, base_url: str) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model, api_key, base_url)
    if provider == "hash":
        return HashingEmbedder()
    return OnnxEmbedder(model, cache_dir)


def get_embedder(settings: Settings) -> Embedder:
    """Build (and memoise) the configured embedder.

    The ONNX backend falls back to hashing rather than taking the process down:
    a first run with no cached weights and no network should still serve.
    """
    model = settings.openai_embedding_model if settings.embedding_provider == "openai" else settings.embedding_model
    try:
        return _build(
            settings.embedding_provider,
            model,
            settings.embedding_cache_dir,
            settings.openai_api_key,
            settings.openai_base_url,
        )
    except Exception:
        if settings.embedding_provider == "hash":
            raise
        return _build("hash", "", "", "", "")
