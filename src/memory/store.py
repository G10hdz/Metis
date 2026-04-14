"""ChromaDB persistent client wrapper for Metis knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.config.ollama import get_embedding_model

logger = logging.getLogger(__name__)


class MetisStore:
    """Thin wrapper around ChromaDB PersistentClient."""

    def __init__(self) -> None:
        self._chroma_settings = ChromaSettings(anonymized_telemetry=False)
        self._client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=self._chroma_settings,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedding_model()
        logger.info("ChromaDB initialized at %s collection=%s", settings.CHROMA_DIR, settings.CHROMA_COLLECTION)

        self._hypatia_collection = None
        if settings.HYPATIA_CHROMA_DIR:
            hp = Path(settings.HYPATIA_CHROMA_DIR).expanduser().resolve()
            if hp.is_dir():
                try:
                    h_client = chromadb.PersistentClient(path=str(hp), settings=self._chroma_settings)
                    self._hypatia_collection = h_client.get_collection(settings.HYPATIA_CHROMA_COLLECTION)
                    logger.info(
                        "Secondary vector store: path=%s collection=%s",
                        hp,
                        settings.HYPATIA_CHROMA_COLLECTION,
                    )
                except Exception as exc:
                    logger.warning("Could not open secondary Chroma (%s): %s", hp, exc)
            else:
                logger.warning("METIS_HYPATIA_CHROMA_DIR is not a directory: %s", hp)

    def add(self, documents: list[str], ids: list[str] | None = None, metadatas: list[dict] | None = None) -> None:
        """Add documents to the knowledge base."""
        if not documents:
            return
        if ids is None:
            ids = [f"doc_{i}" for i in range(self._collection.count() + 1, self._collection.count() + 1 + len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        self._collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info("Added %d documents to ChromaDB", len(documents))

    def query(self, query_text: str, n_results: int = 3) -> dict[str, Any]:
        """
        Query the knowledge base.
        Returns dict with keys: documents, distances, metadatas, ids.
        If a secondary Chroma store is configured, merges hits from both stores by distance.
        """
        fetch = max(n_results * 2, n_results + 2)
        results = self._collection.query(
            query_texts=[query_text],
            n_results=fetch,
        )
        pairs: list[tuple[str, float, dict, str]] = []

        def _extend(res: dict[str, Any], source: str) -> None:
            docs = res.get("documents") or [[]]
            dists = res.get("distances") or [[]]
            metas = res.get("metadatas") or [[]]
            ids_ = res.get("ids") or [[]]
            row_docs = docs[0] if docs else []
            row_dists = dists[0] if dists else [0.0] * len(row_docs)
            row_metas = metas[0] if metas else [{} for _ in row_docs]
            row_ids = ids_[0] if ids_ else [f"{source}_{i}" for i in range(len(row_docs))]
            for i, doc in enumerate(row_docs):
                d = float(row_dists[i]) if i < len(row_dists) else 0.0
                md = dict(row_metas[i]) if i < len(row_metas) else {}
                md.setdefault("rag_source", source)
                rid = row_ids[i] if i < len(row_ids) else f"{source}_{i}"
                pairs.append((doc, d, md, rid))

        _extend(results, "metis_kb")

        if self._hypatia_collection is not None:
            try:
                hyp = self._hypatia_collection.query(
                    query_texts=[query_text],
                    n_results=fetch,
                )
                _extend(hyp, "secondary_store")
            except Exception as exc:
                logger.warning("Secondary Chroma query failed: %s", exc)

        pairs.sort(key=lambda x: x[1])
        pairs = pairs[:n_results]

        return {
            "documents": [p[0] for p in pairs],
            "distances": [p[1] for p in pairs],
            "metadatas": [p[2] for p in pairs],
            "ids": [p[3] for p in pairs],
        }

    def delete(self, ids: list[str]) -> None:
        """Remove documents by ID."""
        if ids:
            self._collection.delete(ids=ids)

    @property
    def count(self) -> int:
        return self._collection.count()


# --- Singleton pattern ---
_instance: MetisStore | None = None


def get_store() -> MetisStore:
    """Return the ChromaDB store singleton (lazy)."""
    global _instance
    if _instance is None:
        _instance = MetisStore()
    return _instance


def reset_store() -> None:
    """Reset the store singleton (useful for testing)."""
    global _instance
    _instance = None
