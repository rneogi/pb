# PitchBook Observer

Public-source RAG pipeline for tracking startup funding, investment, and M&A activity.

> Works fully offline -- no API key required for the demo.
> Optionally bring your own Anthropic API key for LLM-enhanced answers.

## What This Does

PitchBook Observer crawls publicly accessible sources (SEC EDGAR, PR Newswire, GlobeNewswire, TechCrunch) on a weekly schedule, extracts and classifies content into deal signals, and provides a RAG-powered chat interface for querying the knowledge base with citations.

## 60-Second Demo

```bash
# Windows
runit.bat

# Mac / Linux
chmod +x runit.sh && ./runit.sh
```

This generates 50 synthetic demo articles and starts an interactive chat. Try asking:

- "What funding rounds were announced?"
- "Tell me about recent acquisitions"
- "Which investors are most active?"
- "Show me SEC filings from this week"

No API key needed. No internet required. Everything runs locally.

## Features

- **7-stage ingest pipeline**: crawl -> clean -> delta -> evaluate -> schedule -> index -> digest
- **Hybrid retrieval**: Vector (sentence-transformers) + keyword search with reranking
- **5 specialized agents**: Runtime, Ingest, Compilation, Presentation, Memory
- **Streamlit + Plotly dashboard**: Interactive KPI visualization
- **Session memory**: SmartCard tracks context across queries
- **105 demo questions** for testing and benchmarking
- **Local embeddings** with TF-IDF fallback (no GPU needed)
- **Optional Claude LLM** for enhanced natural language answers

## Two Modes

### Local Mode (Default -- No API Key)

Template-based answers from retrieved context. Full retrieval, reranking, citations, and visualization. Works completely offline.

### LLM Mode (Bring Your Own Key)

Natural language answers powered by Claude. Set `ANTHROPIC_API_KEY` in a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your key
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `runit.bat` | Zero-config demo (Windows) |
| `./runit.sh` | Zero-config demo (Mac/Linux) |
| `run_chat.bat` | Interactive chat (auto-detects API key) |
| `run_pipeline.bat` | Run full ingest pipeline |
| `run_batch.bat` | Batch-run 105 demo questions with profiling |
| `run_api.bat` | Start FastAPI server (port 8000) |

## Pipeline Stages

```
Sources -> CRAWL -> CLEAN -> DELTA -> EVALUATE -> SCHEDULE -> INDEX -> DIGEST
           fetch    extract   compare   classify    job queue   vector    weekly
           HTML     text,     vs prior  deal_signal            + keyword  report
                    chunk     week      noise                  indexes   with cites
```

## Data Sources (Public Only)

| Source | Type | Content |
|--------|------|---------|
| SEC EDGAR | RSS/Atom | Form D filings, regulatory submissions |
| PR Newswire | RSS | Press releases (financial, technology) |
| GlobeNewswire | RSS | Business and technology press releases |
| TechCrunch | RSS | Startup and venture capital news |
| Hacker News | RSS | Technology community top stories |
| VC Portfolios | HTTP | a16z, Sequoia, Y Combinator pages |

All sources are publicly accessible. No logins, paywalls, or API keys required.

## Documentation

All docs live in the **agent hub**: [`pipeline/agents/`](pipeline/agents/)

| Document | Contents |
|----------|----------|
| [pipeline/agents/README.md](pipeline/agents/README.md) | Agent hub entry point |
| [pipeline/agents/SETUP.md](pipeline/agents/SETUP.md) | Installation and configuration guide |
| [pipeline/agents/ARCHITECTURE.md](pipeline/agents/ARCHITECTURE.md) | System design, pipeline stages, data flow |
| [pipeline/agents/AGENTS.md](pipeline/agents/AGENTS.md) | Agent hierarchy and event flow |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Index status |
| `POST` | `/chat` | Template-based query (no API key) |
| `POST` | `/chat/v2` | LLM-enhanced query (optional API key) |
| `GET` | `/agents/status` | Agent system status |
| `POST` | `/agents/ingest` | Trigger manual ingest |
| `POST` | `/visualize` | Launch KPI dashboard |

## Configuration

- `manifests/sources.yaml` -- Data sources (add your own watchlists)
- `manifests/pipeline.yaml` -- Pipeline settings, agent config, embedding options

## Technology Stack

Python 3.12+ | FastAPI | SQLite | sentence-transformers | Streamlit | Plotly | httpx | trafilatura

## Phase 2 (Planned)

- Claim extraction (structured data from unstructured text)
- Entity resolution (normalize company/investor names)
- Record assembly (build deal records with confidence scores)
- State machine (draft -> confirmed -> stale lifecycle)

Database schema and module stubs are already in place.

## Constraints

1. **Public sources only** -- No logins, paywalls, or bypassing protections
2. **Rate limiting** -- Respects robots.txt and applies throttling
3. **Audit trail** -- Stores raw + cleaned + metadata for every fetch
4. **Citations required** -- Every claim links to a source URL
5. **Epistemic humility** -- Acknowledges limitations in responses

## License

MIT License. Respects all source terms of service.
