"""
FastAPI Chat Application
========================
RAG-first chat interface for querying the indexed corpus.
Implements PARGV routing with grounded answers and citations.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
from enum import Enum

# Load .env file if present (for ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(__file__).rsplit("app", 1)[0])

from pipeline.config import load_pipeline_config
from pipeline.index import get_retriever
from pipeline.database import init_database


# =========================================================================
# Pydantic Models
# =========================================================================

class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of results to retrieve")
    mode: Literal["hybrid", "keyword", "vector"] = Field(
        default="hybrid",
        description="Retrieval mode"
    )
    filter_source_kinds: Optional[List[str]] = Field(
        default=None,
        description="Filter by source kinds (filing, pr_wire, news, etc.)"
    )
    filter_weeks: Optional[List[str]] = Field(
        default=None,
        description="Filter by weeks (e.g., ['2026-W05', '2026-W04'])"
    )


class Citation(BaseModel):
    """Citation for a retrieved chunk."""
    url: str
    title: str
    snippet: str
    source_name: Optional[str] = None
    source_kind: Optional[str] = None
    week: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    answer: str
    confidence_label: Literal["low", "medium", "high"]
    citations: List[Citation]
    notes: List[str]
    query_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    index_status: Optional[Dict[str, Any]] = None


# =========================================================================
# PARGV Router
# =========================================================================

class QueryIntent(Enum):
    """Classified query intent."""
    COMPANY = "company"
    INVESTOR = "investor"
    DEALS = "deals"
    TREND = "trend"
    GENERAL = "general"


def classify_intent(query: str) -> QueryIntent:
    """
    Parse: Classify query intent using rules.
    """
    query_lower = query.lower()

    # Check for company-related queries
    company_patterns = [
        r'\b(company|startup|firm)\b',
        r'\bwhat.*(?:does|is)\s+\w+\s+(?:do|working)',
        r'\btell me about\b',
        r'\bprofile\b'
    ]
    for pattern in company_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.COMPANY

    # Check for investor-related queries
    investor_patterns = [
        r'\b(investor|vc|venture capital|pe|private equity)\b',
        r'\bportfolio\b',
        r'\bwho invested\b',
        r'\bfund\b'
    ]
    for pattern in investor_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.INVESTOR

    # Check for deal-related queries
    deal_patterns = [
        r'\b(funding|raised|series [a-d]|seed|investment)\b',
        r'\b(acquisition|acquired|merger|ipo)\b',
        r'\brecent deals\b',
        r'\bhow much\b.*\braised\b'
    ]
    for pattern in deal_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.DEALS

    # Check for trend-related queries
    trend_patterns = [
        r'\b(trend|trending|popular|hot)\b',
        r'\bwhat.*(?:happening|going on)\b',
        r'\bmarket\b',
        r'\bsector\b'
    ]
    for pattern in trend_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.TREND

    return QueryIntent.GENERAL


def get_retrieval_filters(
    intent: QueryIntent,
    filter_source_kinds: Optional[List[str]],
    filter_weeks: Optional[List[str]]
) -> Optional[callable]:
    """
    Abstract: Determine retrieval filters based on intent.
    """
    def filter_fn(metadata: Dict[str, Any]) -> bool:
        # Apply source kind filter
        if filter_source_kinds:
            if metadata.get("source_kind") not in filter_source_kinds:
                return False

        # Apply week filter
        if filter_weeks:
            if metadata.get("week") not in filter_weeks:
                return False

        # Intent-based filtering
        source_kind = metadata.get("source_kind", "")

        if intent == QueryIntent.DEALS:
            # Prefer filings, PR wires, news for deal queries
            preferred = ["filing", "pr_wire", "news"]
            if source_kind and source_kind not in preferred:
                return False

        elif intent == QueryIntent.INVESTOR:
            # Prefer investor portfolio pages
            preferred = ["investor_portfolio", "news", "pr_wire"]
            if source_kind and source_kind not in preferred:
                return False

        return True

    # Only return filter if we have constraints
    if filter_source_kinds or filter_weeks or intent in [QueryIntent.DEALS, QueryIntent.INVESTOR]:
        return filter_fn

    return None


def generate_answer(
    query: str,
    intent: QueryIntent,
    results: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> tuple:
    """
    Generate: Create grounded answer from retrieved chunks.
    Returns (answer, confidence_label).
    """
    if not results:
        return (
            "I couldn't find relevant information to answer your query. "
            "This could mean the topic isn't covered in the indexed public sources, "
            "or the information may not be publicly available.",
            "low"
        )

    # Calculate confidence based on scores
    confidence_cfg = config.get("confidence", {})
    high_threshold = confidence_cfg.get("high_min_score", 0.8)
    medium_threshold = confidence_cfg.get("medium_min_score", 0.5)

    top_score = results[0].get("score", 0) if results else 0

    if top_score >= high_threshold:
        confidence = "high"
    elif top_score >= medium_threshold:
        confidence = "medium"
    else:
        confidence = "low"

    # Build answer from top results
    answer_parts = []

    if intent == QueryIntent.DEALS:
        answer_parts.append("Based on recent public filings and announcements:\n")
    elif intent == QueryIntent.INVESTOR:
        answer_parts.append("From public investor and portfolio information:\n")
    elif intent == QueryIntent.COMPANY:
        answer_parts.append("Based on available public information:\n")
    else:
        answer_parts.append("Here's what I found from public sources:\n")

    # Add summarized findings
    seen_urls = set()
    for i, result in enumerate(results[:5]):
        meta = result.get("metadata", {})
        url = meta.get("canonical_url", "")

        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = meta.get("title", "Untitled")
        snippet = meta.get("text", "")[:200]

        answer_parts.append(f"\n**{title}**")
        if snippet:
            # Clean up snippet
            snippet = snippet.replace("\n", " ").strip()
            answer_parts.append(f"\n> {snippet}...")

    # Add confidence caveat
    if confidence == "low":
        answer_parts.append(
            "\n\n_Note: The relevance of these results to your specific query is uncertain. "
            "Please verify information directly from the source links._"
        )
    elif confidence == "medium":
        answer_parts.append(
            "\n\n_Note: These results appear relevant but may not fully answer your query. "
            "See citations for original sources._"
        )

    return "\n".join(answer_parts), confidence


def validate_response(citations: List[Citation], answer: str) -> List[str]:
    """
    Validate: Ensure response is grounded with citations.
    Returns list of notes/warnings.
    """
    notes = [
        "Epistemic humility: This is a public-source observer; information may be incomplete.",
        "All findings should be verified against original sources."
    ]

    if not citations:
        notes.append("WARNING: No citations found - response may not be grounded.")

    # Check if answer makes claims without corresponding citations
    claim_keywords = ["raised", "acquired", "announced", "reported", "according to"]
    has_claims = any(kw in answer.lower() for kw in claim_keywords)

    if has_claims and len(citations) < 2:
        notes.append("Note: Limited citations for claims made. Verify with additional sources.")

    return notes


# =========================================================================
# FastAPI Application
# =========================================================================

app = FastAPI(
    title="Public PitchBook Observer",
    description="RAG-first chat interface for public funding/investment data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_retriever = None
_config = None


def get_retriever_instance():
    """Get or create retriever instance."""
    global _retriever, _config
    if _retriever is None:
        _config = load_pipeline_config()
        _retriever = get_retriever(_config)
    return _retriever, _config


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("Initializing database...")
    init_database()
    print("Loading retriever...")
    get_retriever_instance()
    print("Ready!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        retriever, _ = get_retriever_instance()
        index_status = {
            "vector_count": retriever.vector_store.count(),
            "has_keyword_index": bool(retriever.keyword_index.inverted_index)
        }
    except Exception as e:
        index_status = {"error": str(e)}

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        index_status=index_status
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint implementing PARGV routing.

    PARGV Steps:
    1. Parse: Classify query intent
    2. Abstract: Determine retrieval strategy
    3. Retrieve: Search indexed corpus
    4. Generate: Create grounded answer
    5. Validate: Ensure citations exist
    """
    try:
        retriever, config = get_retriever_instance()
        chat_cfg = config.chat
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize retriever: {e}")

    # 1. PARSE: Classify intent
    intent = classify_intent(request.query)

    # 2. ABSTRACT: Determine retrieval filters
    filter_fn = get_retrieval_filters(
        intent,
        request.filter_source_kinds,
        request.filter_weeks
    )

    # 3. RETRIEVE: Search corpus
    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
            filter_fn=filter_fn
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    # Build citations
    citations = []
    for result in results:
        meta = result.get("metadata", {})
        citations.append(Citation(
            url=meta.get("canonical_url", ""),
            title=meta.get("title", "Untitled"),
            snippet=meta.get("text", "")[:300],
            source_name=meta.get("source_name"),
            source_kind=meta.get("source_kind"),
            week=meta.get("week"),
            score=result.get("score")
        ))

    # 4. GENERATE: Create answer
    answer, confidence = generate_answer(request.query, intent, results, chat_cfg)

    # 5. VALIDATE: Check grounding
    notes = validate_response(citations, answer)

    return ChatResponse(
        answer=answer,
        confidence_label=confidence,
        citations=citations,
        notes=notes,
        query_info={
            "intent": intent.value,
            "mode": request.mode,
            "results_count": len(results)
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Public PitchBook Observer API",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/chat": "POST - Chat with template generation (legacy)",
            "/chat/v2": "POST - Chat with LLM generation (Runtime Agent)",
            "/visualize": "POST - Launch KPI dashboard",
            "/agents/status": "GET - All agent statuses",
            "/agents/ingest": "POST - Trigger manual ingest",
            "/docs": "OpenAPI documentation"
        },
        "note": "This is a public-source observer. Information may be incomplete."
    }


# =========================================================================
# Agent Endpoints (v2)
# =========================================================================

class ChatV2Request(BaseModel):
    """Request schema for v2 chat endpoint with LLM."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of results to retrieve")
    rerank_top_k: int = Field(default=5, ge=1, le=20, description="Results after reranking")
    mode: Literal["hybrid", "keyword", "vector"] = Field(default="hybrid")
    use_llm: bool = Field(default=True, description="Use Claude LLM for generation")
    use_memory: bool = Field(default=True, description="Include session memory context")
    filter_source_kinds: Optional[List[str]] = Field(default=None)
    filter_weeks: Optional[List[str]] = Field(default=None)


class ChatV2Response(BaseModel):
    """Response schema for v2 chat endpoint."""
    query: str
    answer: str
    confidence_label: Literal["low", "medium", "high"]
    citations: List[Citation]
    notes: List[str]
    query_info: Dict[str, Any]
    timings: Dict[str, float]
    timestamp: str


class AgentStatusResponse(BaseModel):
    """Status of all agents."""
    agents: Dict[str, Dict[str, Any]]
    timestamp: str


class IngestRequest(BaseModel):
    """Request for manual ingest."""
    week: Optional[str] = Field(default=None, description="Target week (YYYY-WNN)")
    stages: Optional[List[str]] = Field(default=None, description="Specific stages to run")


class IngestResponse(BaseModel):
    """Response from ingest operation."""
    status: str
    week: str
    stages_succeeded: List[str]
    stages_failed: List[str]
    duration_seconds: float


class VisualizeResponse(BaseModel):
    """Response from visualization launch."""
    status: str
    dashboard_url: Optional[str]
    kpis_extracted: int
    message: str


@app.post("/chat/v2", response_model=ChatV2Response)
async def chat_v2(request: ChatV2Request):
    """
    Enhanced chat endpoint using Runtime Agent.

    Features:
    - LLM-powered generation (Claude)
    - Document reranking
    - Session memory context
    - Detailed timing breakdown
    """
    try:
        from pipeline.agents.runtime_agent import RuntimeAgent

        agent = RuntimeAgent(
            use_llm=request.use_llm,
            use_memory=request.use_memory
        )

        result = agent.run(
            query=request.query,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            mode=request.mode,
            filter_source_kinds=request.filter_source_kinds,
            filter_weeks=request.filter_weeks
        )

        # Convert citations to Pydantic models
        citations = [
            Citation(
                url=c.get("url", ""),
                title=c.get("title", "Untitled"),
                snippet=c.get("snippet", "")[:300],
                source_name=c.get("source_name"),
                source_kind=c.get("source_kind"),
                week=c.get("week"),
                score=c.get("score")
            )
            for c in result.get("citations", [])
        ]

        return ChatV2Response(
            query=result.get("query", request.query),
            answer=result.get("answer", ""),
            confidence_label=result.get("confidence_label", "low"),
            citations=citations,
            notes=result.get("notes", []),
            query_info=result.get("query_info", {}),
            timings=result.get("timings", {}),
            timestamp=result.get("timestamp", datetime.utcnow().isoformat())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Runtime agent error: {e}")


@app.get("/agents/status", response_model=AgentStatusResponse)
async def agents_status():
    """Get status of all agents."""
    statuses = {}

    try:
        from pipeline.agents.ingest_agent import IngestAgent
        statuses["ingest"] = IngestAgent().get_status()
    except Exception as e:
        statuses["ingest"] = {"error": str(e)}

    try:
        from pipeline.agents.compilation_agent import CompilationAgent
        statuses["compilation"] = CompilationAgent().get_status()
    except Exception as e:
        statuses["compilation"] = {"error": str(e)}

    try:
        from pipeline.agents.runtime_agent import RuntimeAgent
        statuses["runtime"] = RuntimeAgent().get_status()
    except Exception as e:
        statuses["runtime"] = {"error": str(e)}

    try:
        from pipeline.agents.presentation_agent import PresentationAgent
        statuses["presentation"] = PresentationAgent().get_status()
    except Exception as e:
        statuses["presentation"] = {"error": str(e)}

    try:
        from pipeline.agents.memory_agent import MemoryAgent
        statuses["memory"] = MemoryAgent().get_status()
    except Exception as e:
        statuses["memory"] = {"error": str(e)}

    return AgentStatusResponse(
        agents=statuses,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/agents/ingest", response_model=IngestResponse)
async def trigger_ingest(request: IngestRequest):
    """
    Trigger manual data ingest.

    This runs the Ingest Agent immediately with specified parameters.
    """
    try:
        from pipeline.agents.ingest_agent import IngestAgent

        agent = IngestAgent()
        result = agent.run(week=request.week, stages=request.stages)

        return IngestResponse(
            status="success" if not result.get("stages_failed") else "partial",
            week=result.get("week", ""),
            stages_succeeded=result.get("stages_succeeded", []),
            stages_failed=result.get("stages_failed", []),
            duration_seconds=result.get("duration_seconds", 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")


@app.post("/visualize", response_model=VisualizeResponse)
async def launch_visualization():
    """
    Launch the KPI visualization dashboard.

    Extracts KPIs from the latest response and opens Streamlit dashboard.
    """
    try:
        from pipeline.agents.presentation_agent import PresentationAgent

        agent = PresentationAgent()
        result = agent.run(launch_dashboard=True)

        return VisualizeResponse(
            status=result.get("status", "unknown"),
            dashboard_url=result.get("dashboard_url"),
            kpis_extracted=result.get("kpis_extracted", 0),
            message=f"Dashboard launched with {result.get('kpis_extracted', 0)} KPIs"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
