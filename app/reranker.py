"""
Reranker
========
Document reranking strategies for improved retrieval quality.

Strategies:
    - cross_encoder: Semantic reranking using cross-encoder model
    - keyword_boost: Boost scores based on keyword matches
    - reciprocal_rank_fusion: Combine multiple ranking signals
"""

from typing import Any, Callable, Dict, List, Optional

# Optional imports for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    CrossEncoder = None


class Reranker:
    """
    Document reranking with multiple strategies.

    Reranking improves retrieval quality by re-scoring
    retrieved documents using more sophisticated methods
    than the initial retrieval.

    Strategies:
        - cross_encoder: Best quality, uses cross-encoder neural model
        - keyword_boost: Fast, boosts by query term overlap
        - reciprocal_rank_fusion: Combines multiple ranking signals
    """

    # Cross-encoder model for semantic reranking
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, strategy: str = "keyword_boost"):
        """
        Initialize reranker with strategy.

        Args:
            strategy: Reranking strategy (cross_encoder, keyword_boost, rrf)
        """
        self.strategy = strategy
        self._cross_encoder = None
        self._initialize_strategy()

    def _initialize_strategy(self) -> None:
        """Initialize resources for the selected strategy."""
        if self.strategy == "cross_encoder":
            if not HAS_CROSS_ENCODER:
                print(
                    "Warning: sentence-transformers not installed. "
                    "Falling back to keyword_boost. "
                    "Install with: pip install sentence-transformers"
                )
                self.strategy = "keyword_boost"
            else:
                try:
                    self._cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
                except Exception as e:
                    print(f"Warning: Could not load cross-encoder: {e}")
                    self.strategy = "keyword_boost"

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on selected strategy.

        Args:
            query: User query
            documents: Retrieved documents with metadata and scores
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents with rerank_score added
        """
        if not documents:
            return []

        if self.strategy == "cross_encoder" and self._cross_encoder:
            return self._rerank_cross_encoder(query, documents, top_k)
        elif self.strategy == "reciprocal_rank_fusion" or self.strategy == "rrf":
            return self._rerank_rrf(query, documents, top_k)
        else:
            return self._rerank_keyword_boost(query, documents, top_k)

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using cross-encoder model for semantic scoring.

        Cross-encoders process query and document together,
        providing more accurate relevance scores than bi-encoders.
        """
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get("metadata", {}).get("text", "")[:512]  # Limit length
            pairs.append((query, text))

        # Get cross-encoder scores
        try:
            scores = self._cross_encoder.predict(pairs)
        except Exception as e:
            print(f"Cross-encoder failed: {e}, falling back to keyword_boost")
            return self._rerank_keyword_boost(query, documents, top_k)

        # Add rerank scores and preserve original scores
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            doc["original_score"] = doc.get("score", 0)
            doc["rerank_method"] = "cross_encoder"

        # Sort by rerank score and return top-k
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _rerank_keyword_boost(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank by boosting documents with query keyword matches.

        Fast and simple strategy that boosts scores based on
        overlap between query terms and document content.
        """
        # Tokenize query
        query_terms = set(query.lower().split())
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        query_terms = query_terms - stopwords

        for doc in documents:
            meta = doc.get("metadata", {})
            text = meta.get("text", "").lower()
            title = meta.get("title", "").lower()

            # Count matches
            text_matches = sum(1 for term in query_terms if term in text)
            title_matches = sum(1 for term in query_terms if term in title) * 2  # Title boost

            # Calculate boost factor
            total_matches = text_matches + title_matches
            boost = total_matches / (len(query_terms) + 1) if query_terms else 0

            # Apply boost to original score
            original_score = doc.get("score", 0)
            doc["rerank_score"] = original_score * (1 + boost * 0.5)
            doc["original_score"] = original_score
            doc["rerank_method"] = "keyword_boost"
            doc["keyword_matches"] = total_matches

        # Sort and return top-k
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _rerank_rrf(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion for combining multiple ranking signals.

        RRF combines rankings from different methods (vector, keyword)
        using the formula: score = sum(1 / (k + rank_i)) for each method.

        Args:
            k: RRF parameter (default 60, per original paper)
        """
        # Separate documents by retrieval method
        rrf_scores = {}

        # Get vector rankings
        vector_docs = sorted(
            [d for d in documents if d.get("vector_score") is not None],
            key=lambda x: x.get("vector_score", 0),
            reverse=True
        )

        # Get keyword rankings
        keyword_docs = sorted(
            [d for d in documents if d.get("keyword_score") is not None],
            key=lambda x: x.get("keyword_score", 0),
            reverse=True
        )

        # If no method-specific scores, use original score for both
        if not vector_docs and not keyword_docs:
            # Treat all as single ranking
            sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
            for rank, doc in enumerate(sorted_docs):
                chunk_id = doc.get("metadata", {}).get("chunk_id", id(doc))
                rrf_scores[chunk_id] = 1 / (k + rank + 1)
        else:
            # Compute RRF from vector rankings
            for rank, doc in enumerate(vector_docs):
                chunk_id = doc.get("metadata", {}).get("chunk_id", id(doc))
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

            # Add RRF from keyword rankings
            for rank, doc in enumerate(keyword_docs):
                chunk_id = doc.get("metadata", {}).get("chunk_id", id(doc))
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # Deduplicate and assign RRF scores
        seen = set()
        results = []

        for doc in documents:
            chunk_id = doc.get("metadata", {}).get("chunk_id", id(doc))
            if chunk_id not in seen:
                seen.add(chunk_id)
                doc["rerank_score"] = rrf_scores.get(chunk_id, 0)
                doc["original_score"] = doc.get("score", 0)
                doc["rerank_method"] = "rrf"
                results.append(doc)

        # Sort by RRF score and return top-k
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def get_status(self) -> Dict[str, Any]:
        """Get reranker status information."""
        return {
            "strategy": self.strategy,
            "cross_encoder_loaded": self._cross_encoder is not None,
            "cross_encoder_available": HAS_CROSS_ENCODER,
            "model": self.CROSS_ENCODER_MODEL if self._cross_encoder else None
        }


def get_reranker(strategy: Optional[str] = None) -> Reranker:
    """
    Factory function to create reranker instance.

    Args:
        strategy: Reranking strategy (default: keyword_boost)

    Returns:
        Configured Reranker instance
    """
    # Default to keyword_boost as it's fast and has no dependencies
    strategy = strategy or "keyword_boost"
    return Reranker(strategy=strategy)
