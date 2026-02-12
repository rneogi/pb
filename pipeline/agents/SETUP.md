# Setup Guide

## Quick Start (No API Key)

The fastest way to try PitchBook Observer:

### Windows
```
runit.bat
```

### Mac / Linux
```bash
chmod +x runit.sh
./runit.sh
```

This installs dependencies, generates 50 synthetic demo articles, and starts the interactive chat in local mode. No API key needed.

## Manual Setup

### 1. Prerequisites

- Python 3.12+
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The `sentence-transformers` package downloads a ~100 MB model on first use. If this fails, the system automatically falls back to TF-IDF embeddings (lighter but less accurate).

### 3. (Optional) Configure LLM API Key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and uncomment + fill in your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Get your key at [console.anthropic.com](https://console.anthropic.com/).

### 4. Generate Demo Data

```bash
python scripts/inject_demo_data.py
```

This creates 50 synthetic articles (funding rounds, M&A, investor news, SEC filings) and builds the search index.

### 5. Start Chatting

```bash
# Local mode (no API key)
python -m app.chat_interface

# LLM mode (requires ANTHROPIC_API_KEY in .env)
python -m app.chat_interface --llm

# Windows batch file (auto-detects mode)
run_chat.bat
```

### 6. (Optional) Start API Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 7. (Optional) Launch Visualization Dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

## Running the Full Pipeline

To crawl real public sources (SEC EDGAR, PR Newswire, GlobeNewswire, TechCrunch):

```bash
# Run all stages for current week
python -m pipeline.run

# Run for specific week
python -m pipeline.run --week 2026-W05

# Run only a specific stage
python -m pipeline.run --week 2026-W05 --only crawl
```

## Batch Demo Runner

Run all 105 demo questions with profiling:

```bash
# All questions
python -m app.batch_runner

# Specific category
python -m app.batch_runner --category deals_funding --limit 20

# Windows
run_batch.bat
```

## Two Operating Modes

### Local Mode (Default)

- No API key required
- Uses template-based answer generation from retrieved context
- Full hybrid retrieval (vector + keyword), reranking, citations
- Session memory (SmartCard) and visualization work fully
- Suitable for exploring data and demonstrating the architecture

### LLM Mode (Optional)

- Requires `ANTHROPIC_API_KEY` in `.env`
- Claude generates natural language answers with reasoning
- Same retrieval pipeline underneath
- Enhanced synthesis and follow-up capabilities

## Chat Commands

| Command | Description |
|---------|-------------|
| `help` | Show all commands |
| `quit` / `exit` | Exit chat |
| `mode hybrid\|vector\|keyword` | Change retrieval mode |
| `topk <n>` | Set number of results (1-50) |
| `timing on\|off` | Toggle timing display |
| `stats` | Show session statistics |
| `memory` | Show SmartCard contents |
| `viz` | Launch visualization dashboard |
| `export <file>` | Export history to JSON |

## Query Modifiers

| Modifier | Description |
|----------|-------------|
| `--deals` | Filter to deal-related sources |
| `--news` | Filter to news sources only |
| `--filings` | Filter to SEC filings only |

## Troubleshooting

### "sentence-transformers" fails to install

The system automatically falls back to TF-IDF embeddings. TF-IDF is lighter but less accurate for semantic search. Everything still works.

### No results returned

Make sure demo data is generated:
```bash
python scripts/inject_demo_data.py
```

Or run the full pipeline to crawl real sources:
```bash
python -m pipeline.run
```

### Streamlit dashboard not launching

```bash
pip install streamlit plotly pandas
```

### "ModuleNotFoundError: No module named 'yaml'"

```bash
pip install pyyaml
```

### HTTP/2 errors during crawl

```bash
pip install "httpx[http2]"
```

## Configuring Sources

Edit `manifests/sources.yaml` to add or modify data sources:

```yaml
company_watchlist:
  enabled: true
  source_kind: company_press
  entries:
    - name: "Company Blog"
      url: "https://company.com/blog"
      format: html
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.
