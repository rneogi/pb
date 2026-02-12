"""
Runtime Agent
=============
Query-time retrieval, reranking, and LLM generation.

Orchestrates the PARGV pipeline at runtime:
    1. Parse: Classify query intent
    2. Abstract: Determine retrieval filters
    3. Retrieve: Hybrid search (vector + keyword)
    4. Rerank: Cross-encoder or keyword boost
    5. Augment: Load memory context
    6. Generate: Claude LLM response
    7. Validate: Check grounding

Triggered by: HTTP request to /chat endpoint
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add app directory to path for imports
APP_DIR = Path(__file__).parent.parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from .agent_base import AgentBase, RESPONSES_DIR
from .memory_agent import MemoryAgent


class RuntimeAgent(AgentBase):
    """
    Handles query-time retrieval, reranking, and LLM generation.

    This is the main agent that processes user queries at runtime.
    It orchestrates the full RAG pipeline including:
        - Hybrid retrieval (vector + keyword)
        - Document reranking
        - Memory context augmentation
        - Claude LLM response generation
        - Response validation

    Events emitted:
        - query_processed: After each query (for analytics)
    """

    def __init__(
        self,
        rerank_strategy: str = "keyword_boost",
        use_llm: bool = True,
        use_memory: bool = True
    ):
        """
        Initialize Runtime Agent.

        Args:
            rerank_strategy: Reranking strategy (cross_encoder, keyword_boost, rrf)
            use_llm: If False, use template-based generation (no API calls)
            use_memory: If True, augment with previous session context
        """
        super().__init__("runtime_agent")
        self.rerank_strategy = rerank_strategy
        self.use_llm = use_llm
        self.use_memory = use_memory

        # Lazy-loaded components
        self._retriever = None
        self._reranker = None
        self._llm_client = None
        self._memory_agent = None

    @property
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None:
            from pipeline.index import get_retriever
            self._retriever = get_retriever(self.config)
        return self._retriever

    @property
    def reranker(self):
        """Lazy-load reranker."""
        if self._reranker is None:
            from app.reranker import get_reranker
            self._reranker = get_reranker(self.rerank_strategy)
        return self._reranker

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None and self.use_llm:
            from app.llm_client import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def memory_agent(self):
        """Lazy-load memory agent."""
        if self._memory_agent is None and self.use_memory:
            self._memory_agent = MemoryAgent()
        return self._memory_agent

    def run(
        self,
        query: str,
        top_k: int = 8,
        rerank_top_k: int = 5,
        mode: str = "hybrid",
        filter_source_kinds: Optional[List[str]] = None,
        filter_weeks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            rerank_top_k: Number of documents after reranking
            mode: Retrieval mode (hybrid, vector, keyword)
            filter_source_kinds: Optional filter by source types
            filter_weeks: Optional filter by weeks

        Returns:
            Complete response with answer, citations, confidence, etc.
        """
        start_time = datetime.utcnow()
        timings = {}

        # Import intent classification from app.main
        from app.main import classify_intent, get_retrieval_filters, validate_response

        # 1. PARSE: Classify intent
        t0 = datetime.utcnow()
        intent = classify_intent(query)
        timings["intent_ms"] = (datetime.utcnow() - t0).total_seconds() * 1000

        # 2. ABSTRACT: Determine retrieval filters
        filter_fn = get_retrieval_filters(intent, filter_source_kinds, filter_weeks)

        # 3. RETRIEVE: Get candidates from hybrid search
        t0 = datetime.utcnow()
        results = self.retriever.search(
            query=query,
            top_k=top_k * 2,  # Over-fetch for reranking
            mode=mode,
            filter_fn=filter_fn
        )
        timings["retrieval_ms"] = (datetime.utcnow() - t0).total_seconds() * 1000

        # 4. RERANK: Apply reranking strategy
        t0 = datetime.utcnow()
        reranked_results = self.reranker.rerank(query, results, top_k=rerank_top_k)
        timings["reranking_ms"] = (datetime.utcnow() - t0).total_seconds() * 1000

        # 5. AUGMENT: Load memory context (with context change detection)
        memory_context = None
        memory_decision = {"use_memory": False, "reason": "disabled"}

        if self.use_memory and self.memory_agent:
            t0 = datetime.utcnow()

            # Get augmentation decision based on context coherence
            memory_decision = self.memory_agent.get_augmentation_decision(query)

            if memory_decision.get("use_memory", False):
                memory_context = memory_decision.get("context", "")
                self.logger.info(f"Memory augmentation: ENABLED (reason: {memory_decision.get('reason')})")
            else:
                # Context changed - skip memory to prevent hallucination
                self.logger.info(
                    f"Memory augmentation: DISABLED (reason: {memory_decision.get('reason')}) - "
                    "using retrieval-only augmentation"
                )
                memory_context = None

            timings["memory_ms"] = (datetime.utcnow() - t0).total_seconds() * 1000

        # 6. GENERATE: Create response
        t0 = datetime.utcnow()
        if self.use_llm and self.llm_client:
            llm_response = self.llm_client.generate_response(
                query=query,
                context_chunks=reranked_results,
                memory_context=memory_context
            )
            answer = llm_response.get("text", "")
            usage = llm_response.get("usage", {})
            model = llm_response.get("model", "unknown")
        else:
            # Template-based fallback
            answer = self._generate_template_answer(query, intent, reranked_results)
            usage = {}
            model = "template"

        timings["generation_ms"] = (datetime.utcnow() - t0).total_seconds() * 1000

        # Build citations
        citations = self._build_citations(reranked_results)

        # 7. VALIDATE: Check grounding
        t0 = datetime.utcnow()
        notes = validate_response(citations, answer)
        timings["validation_ms"] = (datetime.utcnow() - t0).total_seconds() * 1000

        # Compute confidence
        confidence = self._compute_confidence(reranked_results)

        # Total time
        total_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        timings["total_ms"] = total_ms

        # Build response
        response = {
            "query": query,
            "answer": answer,
            "confidence_label": confidence,
            "citations": citations,
            "notes": notes,
            "query_info": {
                "intent": intent.value,
                "mode": mode,
                "results_retrieved": len(results),
                "results_after_rerank": len(reranked_results),
                "rerank_strategy": self.rerank_strategy,
                "model": model,
                "usage": usage,
                "memory_augmentation": {
                    "used": memory_decision.get("use_memory", False),
                    "reason": memory_decision.get("reason", "unknown")
                }
            },
            "timings": timings,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update memory with this response
        if self.use_memory and self.memory_agent:
            self.memory_agent.run(response)

        # Save response for visualization
        self._save_response(response)

        # Auto-generate visualization data
        self._generate_visualization(response)

        # Emit event for analytics
        self.emit_event("query_processed", {
            "query": query[:100],
            "intent": intent.value,
            "confidence": confidence,
            "results_count": len(reranked_results),
            "total_ms": total_ms
        })

        self.logger.info(
            f"Query processed: intent={intent.value}, "
            f"results={len(reranked_results)}, "
            f"confidence={confidence}, "
            f"memory_used={memory_decision.get('use_memory', False)}, "
            f"time={total_ms:.0f}ms"
        )

        return response

    def _generate_template_answer(
        self,
        query: str,
        intent,
        results: List[Dict[str, Any]]
    ) -> str:
        """Generate template-based answer (no LLM)."""
        if not results:
            return (
                "I couldn't find relevant information to answer your query. "
                "This could mean the topic isn't covered in the indexed public sources."
            )

        # Intent-specific headers
        headers = {
            "deals": "Based on recent public filings and announcements:\n",
            "investor": "From public investor and portfolio information:\n",
            "company": "Based on available public information:\n",
            "trend": "Here's what I found regarding trends:\n",
            "general": "Here's what I found from public sources:\n"
        }

        parts = [headers.get(intent.value, headers["general"])]

        # Add top results
        seen_urls = set()
        for result in results[:5]:
            meta = result.get("metadata", {})
            url = meta.get("canonical_url", "")

            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = meta.get("title", "Untitled")
            snippet = meta.get("text", "")[:200].replace("\n", " ").strip()

            parts.append(f"\n**{title}**")
            if snippet:
                parts.append(f"\n> {snippet}...")

        return "\n".join(parts)

    def _build_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build citation list from results."""
        citations = []
        seen_urls = set()

        for result in results:
            meta = result.get("metadata", {})
            url = meta.get("canonical_url", "")

            if url in seen_urls:
                continue
            seen_urls.add(url)

            citations.append({
                "url": url,
                "title": meta.get("title", "Untitled"),
                "snippet": meta.get("text", "")[:300],
                "source_name": meta.get("source_name"),
                "source_kind": meta.get("source_kind"),
                "week": meta.get("week"),
                "score": result.get("rerank_score", result.get("score"))
            })

        return citations

    def _compute_confidence(self, results: List[Dict[str, Any]]) -> str:
        """Compute confidence label based on retrieval scores."""
        if not results:
            return "low"

        top_score = results[0].get("rerank_score", results[0].get("score", 0))

        # Get thresholds from config
        chat_cfg = self.config.chat if hasattr(self.config, 'chat') else {}
        conf_cfg = chat_cfg.get('confidence', {}) if isinstance(chat_cfg, dict) else {}

        high_threshold = conf_cfg.get('high_min_score', 0.8) if isinstance(conf_cfg, dict) else 0.8
        medium_threshold = conf_cfg.get('medium_min_score', 0.5) if isinstance(conf_cfg, dict) else 0.5

        if top_score >= high_threshold:
            return "high"
        elif top_score >= medium_threshold:
            return "medium"
        return "low"

    def _save_response(self, response: Dict[str, Any]) -> Path:
        """Save response for visualization agent."""
        import json
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        response_file = RESPONSES_DIR / f"response_{timestamp}.json"
        response_file.write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
        return response_file

    def _generate_visualization(self, response: Dict[str, Any]) -> None:
        """Auto-generate visualization data and launch dashboard on first response."""
        try:
            from pipeline.agents.presentation_agent import PresentationAgent

            # Check if dashboard should auto-launch (first response in session)
            dashboard_flag = RESPONSES_DIR / ".dashboard_launched"
            should_launch = not dashboard_flag.exists()

            presentation = PresentationAgent()
            result = presentation.run(response=response, launch_dashboard=should_launch)

            if should_launch:
                # Mark dashboard as launched for this session
                dashboard_flag.write_text(datetime.utcnow().isoformat())
                self.logger.info("Dashboard auto-launched on first response")

            self.logger.info(
                f"Generated visualization: {result.get('kpis_extracted', 0)} KPIs, "
                f"charts: {result.get('chart_types', [])}"
            )
        except Exception as e:
            self.logger.warning(f"Could not generate visualization: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = super().get_status()
        status["rerank_strategy"] = self.rerank_strategy
        status["use_llm"] = self.use_llm
        status["use_memory"] = self.use_memory
        status["llm_available"] = self.llm_client.is_available() if self.llm_client else False
        return status


# CLI entry point
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Runtime Agent")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--top-k", type=int, default=8, help="Results to retrieve")
    parser.add_argument("--rerank-k", type=int, default=5, help="Results after rerank")
    parser.add_argument("--mode", default="hybrid", help="Retrieval mode")
    parser.add_argument("--no-llm", action="store_true", help="Use template generation")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory context")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    agent = RuntimeAgent(
        use_llm=not args.no_llm,
        use_memory=not args.no_memory
    )

    if args.status:
        print(json.dumps(agent.get_status(), indent=2))
    elif args.query:
        result = agent.run(
            query=args.query,
            top_k=args.top_k,
            rerank_top_k=args.rerank_k,
            mode=args.mode
        )
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])
        print("\n" + "="*60)
        print(f"Confidence: {result['confidence_label']}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Time: {result['timings']['total_ms']:.0f}ms")
    else:
        parser.print_help()
