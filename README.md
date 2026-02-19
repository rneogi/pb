# PitchBook Observer

Public deal intelligence agent. Monitors funding rounds, acquisitions, and investor activity from public sources (SEC EDGAR, PR Newswire, TechCrunch, GlobeNewswire) and answers natural language queries with cited sources and session memory.

## Quick Start (SAs)

```bash
# 1. Clone
git clone https://dev.azure.com/genlite-azure/genlite_saagents/_git/pb
cd pb

# 2. Setup (installs deps + prompts for your API key)
setup.bat          # Windows
bash setup.sh      # Mac / Linux

# 3. Launch
runit.bat          # Windows  → opens http://localhost:8501
bash runit.sh      # Mac/Linux
```

Get your API key at **https://console.anthropic.com/** — you'll be prompted during setup.

---

## What It Does

Ask questions in plain English about public deal activity:

- *"What funding rounds were announced this week?"*
- *"Which investors are most active in AI?"*
- *"Tell me about recent acquisitions in fintech"*
- *"Which of these startups is in drug discovery?"*

The agent retrieves relevant articles from its indexed knowledge base, reranks results, augments with session memory, and generates a grounded answer with citations — powered by **Claude Opus 4.6**.

---

## How It Works

```
/pipeline  (admin, weekly)          /pb [query]  (SA, anytime)
├── IngestAgent                     ├── RuntimeAgent  ← PARGV retrieval loop
│   └── Crawl → Clean → Evaluate   ├── MemoryAgent   ← session SmartCard
└── CompilationAgent                └── PresentationAgent ← KPI + charts
    └── Embed → Index
```

**Data sources (public only):**

| Source | Content |
|--------|---------|
| SEC EDGAR Form D | Real funding disclosures |
| PR Newswire | Financial + tech press releases |
| GlobeNewswire | Business + tech news |
| TechCrunch | Startup and VC news |
| Hacker News | Top tech stories |
| a16z, Sequoia, YC | Investor portfolio pages |

The index is pre-built and committed to this repo. SAs get working data on first clone. The admin refreshes it weekly by running `/pipeline` and pushing.

---

## Interfaces

| Interface | How to launch | Best for |
|-----------|--------------|---------|
| **Web UI** | `runit.bat` / `bash runit.sh` | Everyone |
| **CLI** | `run_chat.bat` | Developers |
| **Claude Code** | `/pb [query]` | Developers with Claude Code |
| **Cloud demo** | https://pbobserver.streamlit.app | External demos |

---

## Environment

Your API key goes in `.env` (created automatically by `setup.bat`):

```
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-6
```

`.env` is gitignored — it never leaves your machine.

---

## Refreshing Data (Admin Only)

```bash
/pipeline                          # crawl + compile
git add indexes/
git commit -m "Refresh index YYYY-WNN"
git push azure main
```

SAs run `git pull` to get the updated index.

---

## Configuration

- `manifests/sources.yaml` — add/remove crawl sources and company watchlists
- `.env` — API key and model selection
- `.claude/specs/` — agent specs and pipeline documentation

---

## Tech Stack

Python 3.12 · Streamlit · Plotly · sentence-transformers · Claude Opus 4.6 · httpx · trafilatura

## Constraints

- Public sources only — no logins, paywalls, or bypassing protections
- Rate-limited crawling — respects robots.txt
- Every answer includes source citations
- Acknowledges data gaps and inconsistencies explicitly
