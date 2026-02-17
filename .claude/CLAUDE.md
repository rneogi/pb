# PitchBook Observer Agent

Public deal intelligence agent. Monitors funding rounds, acquisitions, and investor activity from public sources (SEC EDGAR, PR Newswire, TechCrunch, etc.) and answers natural language queries with cited sources and episodic memory.

## Architecture

Single agent (`/pb`) wrapping a multi-stage pipeline:

```
User Query
    │
    ▼
RuntimeAgent          ← PARGV loop: Parse → Abstract → Retrieve → Generate → Validate
    ├── IngestAgent   ← Crawl → Clean → Delta → Evaluate → Schedule → Index
    ├── MemoryAgent   ← SmartCard episodic memory (session-scoped)
    └── PresentationAgent ← KPI extraction, visualization data
```

## Entry Points

| Interface     | Command / File                  | Audience          |
|---------------|---------------------------------|-------------------|
| Claude Code   | `/pb [query]`                   | Developers        |
| Web UI        | `streamlit run app/streamlit_chat.py` | All users    |
| CLI           | `run_chat.bat`                  | Developers        |
| Cloud demo    | https://pbobserver.streamlit.app | Teammates        |

## Key Files

| Path | Purpose |
|------|---------|
| `pipeline/agents/runtime_agent.py` | Main agent orchestrator |
| `pipeline/agents/memory_agent.py` | SmartCard episodic memory |
| `pipeline/agents/ingest_agent.py` | Data ingestion pipeline |
| `pipeline/agents/presentation_agent.py` | KPI + visualization |
| `app/streamlit_chat.py` | Web chat UI |
| `app/chat_interface.py` | CLI chat loop |
| `scripts/inject_demo_data.py` | 50-article demo dataset |
| `manifests/sources.yaml` | Crawl sources config |

## Specs

- [Runtime Agent](.claude/specs/runtime.md)
- [Memory Agent](.claude/specs/memory.md)
- [Ingest Agent](.claude/specs/ingest.md)
- [Presentation Agent](.claude/specs/presentation.md)
- [Pipeline](.claude/specs/pipeline.md)

## Quick Start

```bash
# Web UI (recommended)
runit.bat          # Windows
bash runit.sh      # Mac/Linux

# CLI
run_chat.bat

# Claude Code
/pb what funding rounds were announced this week?
```

## Environment

- `ANTHROPIC_API_KEY` — required for Claude Opus; optional for template mode
- Python 3.12+
- See `.env.example` for full config
