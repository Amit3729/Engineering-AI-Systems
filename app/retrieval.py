"""Document ingestion, chunking, and vector search backed by ChromaDB."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb

from . import documents
from .config import Settings
from .embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = documents.SUPPORTED_SUFFIXES


def corrupt_matches() -> list[dict[str, Any]]:
    """Retrieval output with the contract broken: no text, no usable metadata."""
    return [
        {"document": "\ufffd\ufffd\ufffd", "chunk_id": -1, "score": 0.0, "text": "\ufffd\ufffd<<truncated>>\ufffd",
         "title": "", "section": "", "anchor": "", "source": ""},
        {"document": "", "chunk_id": -1, "score": 0.0, "text": "", "title": "", "section": "", "anchor": "", "source": ""},
    ]


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
        self._sources: list[str] | None = None

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
        chroma = Path(self.settings.chroma_dir)
        excludes = documents.DEFAULT_EXCLUDES + tuple(
            pattern.strip() for pattern in self.settings.excluded_globs.split(",") if pattern.strip()
        )
        files = [path for path in sorted(root.rglob("*")) if documents.is_indexable(path, root, excludes)]
        return [path for path in files if chroma not in path.parents]

    def _fingerprint(self, files: list[Path]) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"{self.embedder.name}|{self.settings.chunk_size_words}|{self.settings.chunk_overlap_words}"
            f"|extractor={documents.EXTRACTOR_VERSION}".encode()
        )
        for path in files:
            digest.update(str(path.relative_to(self.settings.data_dir)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def ingest(self, force: bool = False) -> tuple[int, int]:
        """(Re)build the index. Unchanged corpora skip re-embedding entirely."""
        self._sources = None
        files = self._corpus()
        fingerprint = self._fingerprint(files)

        if not force and self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("fingerprint") == fingerprint and self._collection().count() == manifest.get("chunks"):
                return manifest["documents"], manifest["chunks"]

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        root = Path(self.settings.data_dir)
        for path in files:
            relative = path.relative_to(root).as_posix()
            # The first path component names the corpus a chunk came from, so a
            # caller can tell a CPython manual page from a project note.
            parts = path.relative_to(root).parts
            source = parts[0] if len(parts) > 1 else "root"
            document = documents.load(path)
            chunk_id = 0
            for section in document.sections:
                for chunk in chunk_text(section.text, self.settings.chunk_size_words, self.settings.chunk_overlap_words):
                    ids.append(f"{relative}::{chunk_id}")
                    texts.append(chunk)
                    metadatas.append(
                        {
                            "document": relative,
                            "chunk_id": chunk_id,
                            "title": document.title,
                            "section": section.heading,
                            "anchor": section.anchor,
                            "source": source,
                        }
                    )
                    chunk_id += 1

        # A full rebuild is the honest way to honour deletions and edited chunks.
        try:
            self._client.delete_collection(self.settings.collection_name)
        except Exception:
            pass
        collection = self._collection()

        # Embed and write one batch at a time: a 16k-chunk corpus held as Python
        # float lists all at once costs far more memory than the vectors do.
        batch = max(1, self.settings.ingest_batch_size)
        for start in range(0, len(texts), batch):
            stop = start + batch
            collection.upsert(
                ids=ids[start:stop],
                documents=texts[start:stop],
                embeddings=self.embedder.embed_documents(texts[start:stop]),
                metadatas=metadatas[start:stop],
            )
            logger.info("indexed %d/%d chunks", min(stop, len(texts)), len(texts))

        self._sources = sorted({str(metadata["source"]) for metadata in metadatas})
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "documents": len(files),
                    "chunks": len(texts),
                    "embedder": self.embedder.name,
                    "sources": self._sources,
                }
            )
        )
        return len(files), len(texts)

    def search(self, query: str, limit: int | None = None, source: str | None = None) -> list[dict[str, Any]]:
        # Injected faults live here rather than in the tool, so that the API's
        # pre-retrieval degrades with the tool instead of quietly compensating
        # for it. A fault that only half the system sees is not a useful test.
        if self.settings.fault_injection == "retrieval_timeout":
            raise TimeoutError("vector search did not return in time")
        if self.settings.fault_injection == "malformed_retrieval":
            return corrupt_matches()

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
            where={"source": source} if source else None,
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
                    "title": str(metadata.get("title", "")),
                    "section": str(metadata.get("section", "")),
                    "anchor": str(metadata.get("anchor", "")),
                    "source": str(metadata.get("source", "root")),
                }
            )
        return matches

    def sources(self) -> list[str]:
        """Distinct corpora present in the index.

        Ingestion already knows the answer, so it is written to the manifest.
        The scan below is only for an index built before that was recorded, and
        it is a full pass over every metadata row - too slow to run per request,
        hence the memo.
        """
        if self._sources is not None:
            return self._sources
        try:
            if self._manifest_path.exists():
                recorded = json.loads(self._manifest_path.read_text()).get("sources")
                if recorded is not None:
                    self._sources = [str(item) for item in recorded]
                    return self._sources
            metadatas = self._collection().get(include=["metadatas"])["metadatas"]
            self._sources = sorted({str(item.get("source", "root")) for item in metadatas or []})
        except Exception:
            return []
        return self._sources

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "chromadb",
            "collection": self.settings.collection_name,
            "chunks": self._collection().count(),
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
        }
