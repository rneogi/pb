# PitchBook Observer -- Agent Hub

Drop this project into any repo. Run `runit.bat` (Windows) or `./runit.sh` (Mac/Linux).
No API key required. Everything runs locally.

## Quick Start

```bash
# From project root (two levels up from this folder):
cd ../..

# Windows
runit.bat

# Mac/Linux
./runit.sh
```

Or from this folder directly:

```bash
# Windows
runit.bat

# Mac/Linux
./runit.sh
```

## Bring Your Own API Key (Optional)

For LLM-enhanced answers, create a `.env` file in the project root:

```bash
cd ../..
cp .env.example .env
# Add: ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Then `run_chat.bat` auto-detects the key and enables Claude.

## Agent Files

| File | Agent | Role |
|------|-------|------|
| `runtime_agent.py` | RuntimeAgent | Query orchestrator (PARGV pipeline) |
| `ingest_agent.py` | IngestAgent | Weekly data collection |
| `compilation_agent.py` | CompilationAgent | Index building (event-driven) |
| `memory_agent.py` | MemoryAgent | Session context (SmartCard) |
| `presentation_agent.py` | PresentationAgent | Visualization + KPIs |
| `agent_base.py` | AgentBase | Shared infrastructure |

## Dependencies (sibling modules)

The agents import from the parent project:

```
pipeline/agents/   <-- you are here
  needs:
    pipeline/index.py       (vector + keyword retrieval)
    pipeline/config.py      (YAML config loader)
    pipeline/database.py    (SQLite schema)
    pipeline/crawl.py       (source fetching)
    pipeline/clean.py       (text extraction)
    pipeline/delta.py       (weekly diff)
    pipeline/evaluate.py    (bucket classification)
    app/llm_client.py       (Claude client, optional)
    app/reranker.py         (result reranking)
    app/main.py             (intent classification)
    app/chat_interface.py   (CLI entry point)
    app/kpi_extractor.py    (KPI extraction)
    app/streamlit_dashboard.py  (Plotly dashboards)
    manifests/sources.yaml  (source config)
    manifests/pipeline.yaml (pipeline config)
    scripts/inject_demo_data.py (demo data generator)
```

**Copy the entire project folder** (not just `pipeline/agents/`) to preserve these dependencies.

## Documentation

| Document | Contents |
|----------|----------|
| [AGENTS.md](AGENTS.md) | Agent hierarchy, event flow, interaction diagram |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Pipeline stages, PARGV, embedding strategy |
| [SETUP.md](SETUP.md) | Full installation guide, troubleshooting |

## How It Works

```
User runs runit.bat
    |
    v
Install deps --> Generate 50 demo articles --> Launch chat
    |
    v
User types query: "What funding rounds were announced?"
    |
    v
RuntimeAgent.run()
  |-- Parse intent (deals)
  |-- Retrieve (hybrid: vector + keyword)
  |-- Rerank results
  |-- Augment with session memory
  |-- Generate answer (template or Claude)
  |-- Validate citations
  |-- Update SmartCard
  |-- Extract KPIs for dashboard
    |
    v
Response with citations + confidence label
```
