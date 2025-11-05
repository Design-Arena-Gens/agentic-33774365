from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.logger import get_logger


@dataclass
class StoredDocument:
    id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)


class SemanticStore:
    def __init__(
        self,
        storage_path: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self.logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _read(self) -> List[StoredDocument]:
        raw = self.storage_path.read_text(encoding="utf-8")
        records = json.loads(raw)
        documents: List[StoredDocument] = []
        for record in records:
            documents.append(
                StoredDocument(
                    id=record["id"],
                    content=record["content"],
                    metadata=record.get("metadata", {}),
                    embedding=record.get("embedding", []),
                )
            )
        return documents

    def _write(self, documents: List[StoredDocument]) -> None:
        serialized = []
        for doc in documents:
            serialized.append(
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding,
                }
            )
        self.storage_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def add_document(self, content: str, metadata: Optional[Dict[str, str]] = None) -> str:
        metadata = metadata or {}
        model = self._load_model()
        embedding_vector = model.encode(content).tolist()
        document = StoredDocument(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata,
            embedding=embedding_vector,
        )
        with self._lock:
            docs = self._read()
            docs.append(document)
            self._write(docs)
        self.logger.info(
            "Added document %s to semantic store (%s)", document.id, metadata.get("source", "unknown")
        )
        return document.id

    def similarity_search(self, query: str, top_k: int = 5) -> List[StoredDocument]:
        if not query.strip():
            return []
        model = self._load_model()
        query_vec = model.encode(query)
        with self._lock:
            docs = self._read()
        if not docs:
            return []
        embeddings = np.array([doc.embedding for doc in docs], dtype=np.float32)
        query_vec = np.array(query_vec, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        similarities = (
            np.dot(embeddings, query_vec) / np.clip(norms, a_min=1e-8, a_max=None)
        )
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [docs[idx] for idx in ranked_indices if not math.isnan(similarities[idx])]

    def rebuild_from_documents(self, documents: List[Dict[str, str]]) -> None:
        model = self._load_model()
        stored: List[StoredDocument] = []
        for doc in documents:
            embedding = model.encode(doc["content"]).tolist()
            stored.append(
                StoredDocument(
                    id=doc.get("id", str(uuid.uuid4())),
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                    embedding=embedding,
                )
            )
        with self._lock:
            self._write(stored)
        self.logger.info("Rebuilt semantic store with %d documents", len(stored))
