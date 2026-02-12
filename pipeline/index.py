"""
Index Pipeline Stage
====================
Builds/updates vector index over cleaned chunks for retrieval.
Supports local sentence-transformers embeddings with TF-IDF fallback.
Forward compatible with OpenAI/Vertex embeddings.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol, Tuple
from abc import ABC, abstractmethod
import hashlib

import numpy as np

from .config import load_pipeline_config, ensure_week_dirs, INDEXES_DIR
from .database import (
    get_chunks_by_week, mark_chunks_embedded, db_session,
    get_artifacts_by_week
)


# =========================================================================
# Embedding Interface (Pluggable)
# =========================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, return numpy array of shape (n, dim)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, return numpy array of shape (dim,)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        return self._dimension


class TFIDFProvider(EmbeddingProvider):
    """Fallback TF-IDF based retrieval (no external API needed)."""

    def __init__(self, dimension: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._dimension = dimension
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=dimension)
        self._fitted = False
        self._corpus_vectors = None

    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer on corpus."""
        if not texts:
            return
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[1] > self._dimension:
            self._corpus_vectors = self.svd.fit_transform(tfidf_matrix)
        else:
            self._corpus_vectors = tfidf_matrix.toarray()
            self._dimension = self._corpus_vectors.shape[1]
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
            return self._corpus_vectors

        tfidf_matrix = self.vectorizer.transform(texts)
        if hasattr(self.svd, 'components_'):
            return self.svd.transform(tfidf_matrix)
        return tfidf_matrix.toarray()

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise ValueError("TF-IDF provider not fitted. Call embed() first.")
        tfidf = self.vectorizer.transform([query])
        if hasattr(self.svd, 'components_'):
            return self.svd.transform(tfidf)[0]
        return tfidf.toarray()[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(config: Dict[str, Any]) -> EmbeddingProvider:
    """Factory function to get embedding provider based on config."""
    provider_type = config.get("provider", "local")
    fallback = config.get("fallback_to_tfidf", True)

    if provider_type == "local":
        try:
            model_name = config.get("local_model", "all-MiniLM-L6-v2")
            return SentenceTransformerProvider(model_name)
        except ImportError:
            if fallback:
                print("WARNING: sentence-transformers not available, using TF-IDF fallback")
                return TFIDFProvider(config.get("dimension", 384))
            raise

    elif provider_type == "openai":
        # Placeholder for OpenAI embeddings (Phase 2)
        raise NotImplementedError("OpenAI embeddings not implemented yet")

    elif provider_type == "vertex":
        # Placeholder for Vertex AI embeddings (Phase 2)
        raise NotImplementedError("Vertex AI embeddings not implemented yet")

    else:
        if fallback:
            return TFIDFProvider(config.get("dimension", 384))
        raise ValueError(f"Unknown embedding provider: {provider_type}")


# =========================================================================
# Vector Store (Simple NumPy-based for MVP, Chroma-compatible interface)
# =========================================================================

class VectorStore:
    """Simple vector store using NumPy. Persists to disk."""

    def __init__(self, persist_dir: Path, dimension: int):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension

        self.vectors_path = self.persist_dir / "vectors.npy"
        self.metadata_path = self.persist_dir / "metadata.json"

        self.vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_id_to_idx: Dict[str, int] = {}

        self._load()

    def _load(self):
        """Load existing index from disk."""
        if self.vectors_path.exists() and self.metadata_path.exists():
            self.vectors = np.load(str(self.vectors_path))
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.chunk_id_to_idx = {m["chunk_id"]: i for i, m in enumerate(self.metadata)}
            print(f"Loaded {len(self.metadata)} vectors from {self.persist_dir}")

    def _save(self):
        """Save index to disk."""
        if self.vectors is not None:
            np.save(str(self.vectors_path), self.vectors)
            self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add vectors with metadata."""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        start_idx = len(self.metadata)
        for i, meta in enumerate(metadatas):
            self.metadata.append(meta)
            if "chunk_id" in meta:
                self.chunk_id_to_idx[meta["chunk_id"]] = start_idx + i

        self._save()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors. Returns list of (metadata, score) tuples."""
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # Compute cosine similarity
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(vectors_norm, query_norm)

        # Apply filter if provided
        if filter_fn:
            mask = np.array([filter_fn(m) for m in self.metadata])
            similarities = np.where(mask, similarities, -np.inf)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                results.append((self.metadata[idx], float(similarities[idx])))

        return results

    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata by chunk ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.metadata[idx]
        return None

    def count(self) -> int:
        """Return number of vectors in store."""
        return len(self.metadata)


# =========================================================================
# Keyword Search (for hybrid retrieval)
# =========================================================================

class KeywordIndex:
    """Simple inverted index for keyword search."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "keyword_index.json"

        self.inverted_index: Dict[str, List[str]] = {}  # term -> [chunk_ids]
        self.chunk_texts: Dict[str, str] = {}  # chunk_id -> text

        self._load()

    def _load(self):
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.inverted_index = data.get("inverted_index", {})
            self.chunk_texts = data.get("chunk_texts", {})

    def _save(self):
        data = {
            "inverted_index": self.inverted_index,
            "chunk_texts": self.chunk_texts
        }
        self.index_path.write_text(json.dumps(data), encoding="utf-8")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 2]

    def add(self, chunk_id: str, text: str):
        """Add a chunk to the keyword index."""
        self.chunk_texts[chunk_id] = text
        tokens = self._tokenize(text)
        for token in set(tokens):
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            if chunk_id not in self.inverted_index[token]:
                self.inverted_index[token].append(chunk_id)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """Search by keywords. Returns list of (chunk_id, match_count) tuples."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Count matches per chunk
        chunk_scores: Dict[str, int] = {}
        for token in query_tokens:
            for chunk_id in self.inverted_index.get(token, []):
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1

        # Sort by score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[:top_k]

    def save(self):
        self._save()


# =========================================================================
# Main Retriever Interface
# =========================================================================

class Retriever:
    """Combined retriever supporting keyword, vector, and hybrid search."""

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        embedding_provider: EmbeddingProvider
    ):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.embedding_provider = embedding_provider

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        query_vector = self.embedding_provider.embed_query(query)
        results = self.vector_store.search(query_vector, top_k, filter_fn)
        return [{"metadata": meta, "score": score, "method": "vector"} for meta, score in results]

    def search_keyword(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Keyword search."""
        results = self.keyword_index.search(query, top_k)
        output = []
        for chunk_id, score in results:
            meta = self.vector_store.get_by_chunk_id(chunk_id)
            if meta:
                output.append({
                    "metadata": meta,
                    "score": score / 10.0,  # Normalize
                    "method": "keyword"
                })
        return output

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and keyword."""
        # Get more candidates than needed
        vector_results = self.search_vector(query, top_k * 2, filter_fn)
        keyword_results = self.search_keyword(query, top_k * 2)

        # Combine scores
        combined: Dict[str, Dict[str, Any]] = {}

        for r in vector_results:
            chunk_id = r["metadata"]["chunk_id"]
            combined[chunk_id] = {
                "metadata": r["metadata"],
                "vector_score": r["score"],
                "keyword_score": 0,
                "combined_score": r["score"] * vector_weight
            }

        for r in keyword_results:
            chunk_id = r["metadata"]["chunk_id"]
            if chunk_id in combined:
                combined[chunk_id]["keyword_score"] = r["score"]
                combined[chunk_id]["combined_score"] += r["score"] * (1 - vector_weight)
            else:
                combined[chunk_id] = {
                    "metadata": r["metadata"],
                    "vector_score": 0,
                    "keyword_score": r["score"],
                    "combined_score": r["score"] * (1 - vector_weight)
                }

        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        return [
            {
                "metadata": r["metadata"],
                "score": r["combined_score"],
                "method": "hybrid",
                "vector_score": r["vector_score"],
                "keyword_score": r["keyword_score"]
            }
            for r in sorted_results
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Unified search interface."""
        if mode == "vector":
            return self.search_vector(query, top_k, filter_fn)
        elif mode == "keyword":
            return self.search_keyword(query, top_k)
        else:
            return self.search_hybrid(query, top_k, filter_fn=filter_fn)


# =========================================================================
# Index Builder
# =========================================================================

def run_index(week: str, pipeline_config=None, full_reindex: bool = False) -> Dict[str, Any]:
    """
    Run the index stage for a given week.
    Builds/updates vector and keyword indexes.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    index_cfg = pipeline_config.index
    embed_cfg = index_cfg.get("embeddings", {})

    print(f"Starting index build for week {week}")

    # Initialize embedding provider
    provider = get_embedding_provider(embed_cfg)
    print(f"Using embedding provider with dimension {provider.dimension}")

    # Initialize stores
    persist_dir = INDEXES_DIR / "chroma"
    vector_store = VectorStore(persist_dir, provider.dimension)
    keyword_index = KeywordIndex(persist_dir)

    # Get chunks to index
    if full_reindex:
        # Get all chunks (would need to query all weeks)
        chunks = get_chunks_by_week(week)
    else:
        # Get only non-embedded chunks from current week
        chunks = [c for c in get_chunks_by_week(week) if not c.get("embedded")]

    print(f"Found {len(chunks)} chunks to index")

    if not chunks:
        print("No new chunks to index")
        return {
            "week": week,
            "chunks_indexed": 0,
            "total_vectors": vector_store.count(),
            "completed_at": datetime.utcnow().isoformat()
        }

    # Prepare texts and metadata
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "artifact_id": c["artifact_id"],
            "canonical_url": c["canonical_url"],
            "title": c.get("title", ""),
            "source_kind": c.get("source_kind", ""),
            "source_name": c.get("source_name", ""),
            "week": c["week"],
            "published_at": c.get("published_at"),
            "retrieved_at": c.get("retrieved_at"),
            "text": c["text"][:500]  # Store snippet
        }
        for c in chunks
    ]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = provider.embed(texts)

    # Add to vector store
    print("Adding to vector store...")
    vector_store.add(embeddings, metadatas)

    # Add to keyword index
    print("Building keyword index...")
    for chunk in chunks:
        keyword_index.add(chunk["chunk_id"], chunk["text"])
    keyword_index.save()

    # Mark chunks as embedded
    chunk_ids = [c["chunk_id"] for c in chunks]
    mark_chunks_embedded(chunk_ids)

    summary = {
        "week": week,
        "chunks_indexed": len(chunks),
        "total_vectors": vector_store.count(),
        "embedding_dimension": provider.dimension,
        "completed_at": datetime.utcnow().isoformat()
    }

    # Save summary
    summary_path = dirs["runs"] / "index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nIndex complete: {len(chunks)} chunks indexed, {vector_store.count()} total vectors")
    return summary


def get_retriever(pipeline_config=None) -> Retriever:
    """Get a configured retriever instance."""
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    index_cfg = pipeline_config.index
    embed_cfg = index_cfg.get("embeddings", {})

    provider = get_embedding_provider(embed_cfg)
    persist_dir = INDEXES_DIR / "chroma"
    vector_store = VectorStore(persist_dir, provider.dimension)
    keyword_index = KeywordIndex(persist_dir)

    return Retriever(vector_store, keyword_index, provider)


def main(week: str):
    """Entry point for index stage."""
    return run_index(week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(week)
