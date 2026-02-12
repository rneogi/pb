# Agent Architecture

## Overview

PitchBook Observer uses a hierarchical multi-agent architecture. Each agent has a single responsibility and communicates through file-system events. All agents work fully offline -- the only optional external dependency is Claude API for LLM-enhanced answers.

## Agent Hierarchy

```
                        +-----------------+
                        |   run_chat.bat  |   <-- Entry point (CLI)
                        |   runit.bat     |   <-- Zero-config demo
                        +--------+--------+
                                 |
                                 v
                      +----------+----------+
                      |    RuntimeAgent     |   <-- Query orchestrator
                      | (PARGV pipeline)    |       Optional: Claude API
                      +---+------+------+---+
                          |      |      |
               +----------+   +--+--+   +----------+
               v              v     v              v
        +-----------+   +--------+  +-----------+
        |  Memory   |   |Retriever| |Presentation|
        |  Agent    |   |(local)  | |  Agent     |
        +-----------+   +--------+  +-----------+
        Session ctx     Vector +     Streamlit +
        SmartCard       Keyword      Plotly KPIs


         +---------------+       +------------------+
         |  IngestAgent  | ----> | CompilationAgent |
         | (scheduled)   | event | (event-driven)   |
         +---------------+       +------------------+
         crawl/clean/             Index build
         delta/evaluate           (vectors + keywords)
```

## Agent Details

### RuntimeAgent
**File:** `pipeline/agents/runtime_agent.py`

The top-level query orchestrator. Processes user queries through the full PARGV pipeline:

1. **Parse** -- Classify intent (deals, investor, company, trend, general)
2. **Abstract** -- Determine retrieval filters
3. **Retrieve** -- Hybrid search (vector + keyword)
4. **Rerank** -- Cross-encoder or keyword-boost reranking
5. **Augment** -- Load session memory context (SmartCard)
6. **Generate** -- LLM response (Claude) OR template-based fallback
7. **Validate** -- Check grounding, add epistemic notes

| Property | Value |
|----------|-------|
| API Required | Claude API (optional -- template fallback) |
| Delegates to | MemoryAgent, PresentationAgent |
| Emits | `query_processed` event |

### IngestAgent
**File:** `pipeline/agents/ingest_agent.py`

Weekly data collection orchestrator. Runs the ingest pipeline:

1. **Crawl** -- Fetch content from configured public sources
2. **Clean** -- Extract text, chunk for retrieval
3. **Delta** -- Identify new/changed/stale content
4. **Evaluate** -- Classify into buckets (deal_signal, noise, etc.)

| Property | Value |
|----------|-------|
| API Required | None (fully local) |
| Schedule | Every Sunday at midnight (configurable) |
| Emits | `ingest_complete` event |

### CompilationAgent
**File:** `pipeline/agents/compilation_agent.py`

Event-driven knowledge base builder. Watches for ingest completion and rebuilds the search index.

| Property | Value |
|----------|-------|
| API Required | None (fully local) |
| Consumes | `ingest_complete` event |
| Emits | `compilation_complete` event |
| Action | Build vector + keyword indexes |

### MemoryAgent
**File:** `pipeline/agents/memory_agent.py`

Session context manager using a "SmartCard" -- a persistent JSON structure that accumulates knowledge across queries within a session.

Features:
- Entity tracking (primary entity being discussed)
- Knowledge claims with confidence scores
- Context coherence detection (detects topic shifts)
- Diff-merge updates (only adds new information)
- Visualization data (relationships, time series, comparisons)

| Property | Value |
|----------|-------|
| API Required | None (fully local) |
| Storage | `data/memory/smartcard.json` |
| Called by | RuntimeAgent (augment + post-query update) |

### PresentationAgent
**File:** `pipeline/agents/presentation_agent.py`

Visualization and KPI extraction. Processes query responses and generates interactive dashboards.

Features:
- KPI extraction (funding, acquisition, leadership, product, market)
- Streamlit dashboard (port 8501)
- Plotly charts (network graphs, waterfall, gauges, time series)
- Export to JSON/CSV

| Property | Value |
|----------|-------|
| API Required | None (fully local) |
| Called by | RuntimeAgent (after each query) |
| Emits | `visualization_generated` event |

### AgentBase
**File:** `pipeline/agents/agent_base.py`

Abstract base class providing shared infrastructure:
- Event emission and consumption (file-system based)
- Configuration loading
- Logging with timestamps
- Event archiving

## Event Flow

```
User query via CLI or HTTP
         |
         v
   RuntimeAgent.run()
     |-- MemoryAgent.get_augmentation_decision()
     |-- Retrieve (hybrid search)
     |-- Rerank results
     |-- Generate response (LLM or template)
     |-- MemoryAgent.run(response)       # update SmartCard
     |-- PresentationAgent.run(response) # extract KPIs
     |-- emit "query_processed" event
     v
   Return response with citations


Weekly schedule (or manual trigger)
         |
         v
   IngestAgent.run()
     |-- crawl -> clean -> delta -> evaluate
     |-- emit "ingest_complete" event
         |
         v
   CompilationAgent (watches events)
     |-- Rebuild vector + keyword index
     |-- emit "compilation_complete" event
         |
         v
   Knowledge base updated
```

## Running Without API Key

All agents function without an Anthropic API key:

- **RuntimeAgent**: Falls back to template-based generation from retrieved context
- **IngestAgent**: Fully local (HTTP crawling of public sources)
- **CompilationAgent**: Fully local (sentence-transformers or TF-IDF)
- **MemoryAgent**: Fully local (file-based SmartCard)
- **PresentationAgent**: Fully local (Streamlit + Plotly)

To enable LLM-enhanced answers, set `ANTHROPIC_API_KEY` in your `.env` file.
