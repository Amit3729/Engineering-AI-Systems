"""Document ingestion, chunking, and vector search backed by ChromaDB."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chromadb

from .config import Settings
from .embeddings import Embedder, get_embedder

SUPPORTED_SUFFIXES = (".md", ".txt")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into word-bounded chunks with a sliding overlap.

    Paragraph boundaries are preferred so a chunk rarely starts mid-thought;
    an oversized paragraph is then split by words. The overlap keeps a sentence
    that straddles a boundary retrievable from either side.
    """
    if overlap >= size:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    words: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            words.extend(paragraph.split())
            words.append("\n")
    while words and words[-1] == "\n":
        words.pop()

    chunks: list[str] = []
    step = size - overlap
    for start in range(0, max(len(words), 1), step):
        # A tail shorter than the overlap is already carried by the previous chunk.
        if start and len(words) - start <= overlap:
            break
        window = " ".join(words[start : start + size]).replace(" \n ", "\n").strip()
        if window:
            chunks.append(window)
        if start + size >= len(words):
            break
    return chunks


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._embedder: Embedder | None = None
        self._manifest_path = Path(settings.chroma_dir) / "manifest.json"

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    def _collection(self):
        return self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _corpus(self) -> list[Path]:
        root = Path(self.settings.data_dir)
        files = [path for path in sorted(root.rglob("*")) if path.suffix in SUPPORTED_SUFFIXES and path.is_file()]
        return [path for path in files if Path(self.settings.chroma_dir) not in path.parents]

    def _fingerprint(self, files: list[Path]) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.embedder.name}|{self.settings.chunk_size_words}|{self.settings.chunk_overlap_words}".encode())
        for path in files:
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def ingest(self, force: bool = False) -> tuple[int, int]:
        """(Re)build the index. Unchanged corpora skip re-embedding entirely."""
        files = self._corpus()
        fingerprint = self._fingerprint(files)

        if not force and self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("fingerprint") == fingerprint and self._collection().count() == manifest.get("chunks"):
                return manifest["documents"], manifest["chunks"]

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for path in files:
            chunks = chunk_text(
                path.read_text(encoding="utf-8"),
                self.settings.chunk_size_words,
                self.settings.chunk_overlap_words,
            )
            for chunk_id, chunk in enumerate(chunks):
                ids.append(f"{path.relative_to(self.settings.data_dir)}::{chunk_id}")
                texts.append(chunk)
                metadatas.append({"document": path.name, "chunk_id": chunk_id})

        # A full rebuild is the honest way to honour deletions and edited chunks.
        try:
            self._client.delete_collection(self.settings.collection_name)
        except Exception:
            pass
        collection = self._collection()

        if texts:
            embeddings = self.embedder.embed_documents(texts)
            for start in range(0, len(texts), 256):
                stop = start + 256
                collection.upsert(
                    ids=ids[start:stop],
                    documents=texts[start:stop],
                    embeddings=embeddings[start:stop],
                    metadatas=metadatas[start:stop],
                )

        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps({"fingerprint": fingerprint, "documents": len(files), "chunks": len(texts), "embedder": self.embedder.name})
        )
        return len(files), len(texts)

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        collection = self._collection()
        if collection.count() == 0:
            self.ingest()
            collection = self._collection()
        if collection.count() == 0:
            return []

        limit = limit or self.settings.top_k
        result = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches: list[dict[str, Any]] = []
        for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            matches.append(
                {
                    "document": str(metadata.get("document", "unknown")),
                    "chunk_id": int(metadata.get("chunk_id", 0)),
                    # Chroma returns cosine distance; report similarity instead.
                    "score": round(1.0 - float(distance), 4),
                    "text": text,
                }
            )
        return matches

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "chromadb",
            "collection": self.settings.collection_name,
            "chunks": self._collection().count(),
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
        }
