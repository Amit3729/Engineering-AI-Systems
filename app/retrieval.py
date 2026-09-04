import hashlib
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from .config import Settings


def _embed(text: str, dimensions: int = 384) -> list[float]:
    # Deterministic hashing keeps the demo usable without an embedding API.
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).tolist()


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.data_dir) / "index.json"
        self.records: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.records = json.loads(self.path.read_text())

    def ingest(self) -> tuple[int, int]:
        files = list(Path(self.settings.data_dir).glob("**/*.md")) + list(Path(self.settings.data_dir).glob("**/*.txt"))
        records = []
        for file in files:
            words = file.read_text(encoding="utf-8").split()
            for chunk_id, start in enumerate(range(0, len(words), 180)):
                text = " ".join(words[start:start + 180]).strip()
                if text:
                    records.append({"document": file.name, "chunk_id": chunk_id, "text": text, "embedding": _embed(text)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records), encoding="utf-8")
        self.records = records
        return len(files), len(records)

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if not self.records:
            self.ingest()
        query_vector = np.array(_embed(query))
        scored = []
        for record in self.records:
            score = float(np.dot(query_vector, np.array(record["embedding"])))
            scored.append({key: value for key, value in record.items() if key != "embedding"} | {"score": score})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
