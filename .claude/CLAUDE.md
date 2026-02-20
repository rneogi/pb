# PitchBook Observer Agent

Public deal intelligence agent. Monitors funding rounds, acquisitions, and investor activity from public sources (SEC EDGAR, PR Newswire, TechCrunch, etc.) and answers natural language queries with cited sources and episodic memory.

## Architecture

```
/pipeline                          /pb [query]
├── /ingest                        ├── /runtime  ← PARGV loop
│   └── IngestAgent                ├── /memory   ← SmartCard (auto)
└── /compile                       └── /visualize ← KPI + charts (auto)
    └── CompilationAgent
```

Full agent hierarchy:

```
User Query
    │
    ▼
RuntimeAgent          ← PARGV loop: Parse → Abstract → Retrieve → Generate → Validate
    ├── IngestAgent   ← Crawl → Clean → Delta → Evaluate → Schedule → Index
    ├── MemoryAgent   ← SmartCard episodic memory (session-scoped)
    └── PresentationAgent ← KPI extraction, visualization data
```

## Commands

| Command | File | Purpose |
|---------|------|---------|
| `/pb [query]` | `.claude/commands/pb.md` | Query the agent |
| `/pipeline` | `.claude/commands/pipeline.md` | Run ingest + compile (admin) |
| `/ingest` | `.claude/commands/ingest.md` | Crawl live sources only (admin) |

## Entry Points

| Interface     | Command / File                  | Audience          |
|---------------|---------------------------------|-------------------|
| Claude Code   | `/pb [query]`                   | Developers / SAs  |
| Web UI        | `runit.bat` / `bash runit.sh`   | All users         |
| CLI           | `run_chat.bat`                  | Developers        |
| Cloud demo    | https://pbobserver.streamlit.app | External demo    |

## Key Files

| Path | Purpose |
|------|---------|
| `pipeline/agents/runtime_agent.py` | Main agent orchestrator |
| `pipeline/agents/memory_agent.py` | SmartCard episodic memory |
| `pipeline/agents/ingest_agent.py` | Data ingestion pipeline |
| `pipeline/agents/presentation_agent.py` | KPI + visualization |
| `app/streamlit_chat.py` | Web chat UI |
| `app/chat_interface.py` | CLI chat loop |
| `pipeline/extract_relationships.py` | Accenture relationship extractor |
| `manifests/sources.yaml` | Crawl sources config |
| `indexes/chroma/` | Vector + keyword index (committed — refreshed by admin) |
| `data/private/relationships.csv` | Auto-generated relationship CSV (committed) |

## Specs

- [Runtime Agent](.claude/specs/runtime.md)
- [Memory Agent](.claude/specs/memory.md)
- [Ingest Agent](.claude/specs/ingest.md)
- [Presentation Agent](.claude/specs/presentation.md)
- [Pipeline](.claude/specs/pipeline.md)
- [Compilation Agent](.claude/specs/compilation.md)

## SA Onboarding (First Time Setup)

```bash
# 1. Clone the internal repo
git clone https://dev.azure.com/genlite-azure/genlite_saagents/_git/pb

# 2. Run setup (installs deps, prompts for your API key)
setup.bat          # Windows
bash setup.sh      # Mac/Linux

# 3. Launch
runit.bat          # Web UI (recommended)
run_chat.bat       # CLI

# 4. In Claude Code
/pb what funding rounds were announced this week?
```

## Admin Data Refresh

```bash
# Run weekly to pull fresh data, then push so SAs get it on git pull
/pipeline
git add indexes/ data/private/relationships.csv
git commit -m "Refresh index YYYY-WNN"
git push
```

## Environment

- `ANTHROPIC_API_KEY` — required for LLM synthesis; optional for template mode
- `CLAUDE_MODEL` — defaults to `claude-opus-4-6`
- Python 3.12+
- See `.env.example` for full config
- `.env` is gitignored — each SA sets their own key via `setup.bat`

## Repository

Internal: https://dev.azure.com/genlite-azure/genlite_saagents/_git/pb
