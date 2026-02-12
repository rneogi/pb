# System Architecture

## Pipeline Stages

The ingest pipeline runs weekly (or on-demand) through 7 stages:

```
 Sources          CRAWL        CLEAN         DELTA        EVALUATE      SCHEDULE       INDEX         DIGEST
(SEC EDGAR,   +---------+  +---------+  +-----------+  +----------+  +----------+  +---------+  +---------+
 PR Newswire, | Fetch   |->| Extract |->| Compare   |->| Classify |->| Generate |->| Build   |->| Weekly  |
 GlobeNews,   | content |  | text,   |  | vs prior  |  | buckets  |  | job      |  | vector  |  | digest  |
 TechCrunch,  | via RSS |  | chunk   |  | week      |  |          |  | queue    |  | + kw    |  | with    |
 HackerNews)  | + HTTP  |  |         |  |           |  |          |  |          |  | indexes |  | cites   |
              +---------+  +---------+  +-----------+  +----------+  +----------+  +---------+  +---------+
                  |             |            |              |                            |
                  v             v            v              v                            v
              data/raw/    data/clean/   runs/          runs/                     indexes/chroma/
                           data/meta/    ingest_        eval_                     vectors.npy
                                         delta.json     report.json              metadata.json
```

### Stage Details

| Stage | Module | Input | Output | Notes |
|-------|--------|-------|--------|-------|
| **crawl** | `pipeline/crawl.py` | RSS feeds, URLs | `data/raw/YYYY-WW/*.html` | Rate-limited, async httpx |
| **clean** | `pipeline/clean.py` | Raw HTML | `data/clean/YYYY-WW/*.md` + chunks | trafilatura extraction |
| **delta** | `pipeline/delta.py` | Current + previous week | `runs/YYYY-WW/ingest_delta.json` | Content hash comparison |
| **evaluate** | `pipeline/evaluate.py` | Cleaned text | `runs/YYYY-WW/eval_report.json` | Keyword-based classification |
| **schedule** | `pipeline/schedule.py` | Eval report | `runs/YYYY-WW/job_queue.json` | Phase 1 + Phase 2 stubs |
| **index** | `pipeline/index.py` | Chunks | `indexes/chroma/` | Sentence-transformers or TF-IDF |
| **pargv_batch** | `pipeline/pargv_batch.py` | Eval + chunks | `products/weekly_digest_*.md` | Markdown with citations |

### Bucket Classification

The evaluate stage classifies each artifact:

| Bucket | Description | Keywords |
|--------|-------------|----------|
| `deal_signal` | Funding, M&A, IPO activity | funding, acquisition, series A/B/C, merger |
| `investor_graph_change` | Portfolio or team changes | portfolio, partner, venture |
| `company_profile_change` | Company updates | rebrand, pivot, expansion |
| `telemetry_change` | Hiring/career signals | hiring, careers, headcount |
| `noise` | No significant signals | (default) |

## RAG Query Pipeline (PARGV)

User queries are processed through the RuntimeAgent:

```
Query: "What companies raised Series A recently?"
  |
  v
1. PARSE         Intent = "deals"
  |
2. ABSTRACT      Filters = [source_kind: filing, pr_wire]
  |
3. RETRIEVE      Hybrid search: 70% vector + 30% keyword
  |
4. RERANK        Cross-encoder or keyword-boost scoring
  |
5. AUGMENT       Load session memory (SmartCard context)
  |
6. GENERATE      LLM response (Claude) OR template fallback
  |
7. VALIDATE      Check citations exist, add epistemic notes
  |
  v
Response with citations + confidence label
```

## Embedding Strategy

Three-tier approach with automatic fallback:

| Tier | Provider | Model | Dimension | API Key |
|------|----------|-------|-----------|---------|
| 1 (default) | sentence-transformers | all-MiniLM-L6-v2 | 384 | None |
| 2 (fallback) | scikit-learn TF-IDF | N/A | Variable | None |
| 3 (Phase 2) | OpenAI / Vertex | text-embedding-3-small | 1536 | Required |

The system automatically falls back from Tier 1 to Tier 2 if sentence-transformers is not installed.

## Retrieval Modes

| Mode | Description |
|------|-------------|
| `hybrid` (default) | 70% vector similarity + 30% keyword match |
| `vector` | Pure cosine similarity on embeddings |
| `keyword` | Inverted index with token matching |

## Directory Layout

```
PB/
+-- manifests/
|   +-- sources.yaml             # Crawl source configuration
|   +-- pipeline.yaml            # Pipeline + agent settings
+-- data/
|   +-- raw/YYYY-WW/             # Raw HTML from sources
|   +-- clean/YYYY-WW/           # Extracted text (markdown)
|   +-- meta/YYYY-WW/            # Artifact metadata (JSON)
|   +-- events/                  # Agent event files
|   +-- memory/                  # SmartCard session state
|   +-- responses/               # Cached query responses
+-- indexes/chroma/
|   +-- vectors.npy              # Embedding vectors
|   +-- metadata.json            # Chunk metadata
|   +-- keyword_index.json       # Inverted keyword index
+-- db/
|   +-- pitchbook.db             # SQLite (artifacts, chunks, evals)
+-- products/
|   +-- weekly_digest_*.md       # Generated weekly digests
+-- runs/YYYY-WW/
|   +-- ingest_delta.json        # Delta report
|   +-- eval_report.json         # Evaluation results
|   +-- job_queue.json           # Scheduled jobs
|   +-- pipeline_summary.json    # Run metadata
+-- app/
|   +-- main.py                  # FastAPI application
|   +-- chat_interface.py        # Interactive CLI
|   +-- llm_client.py            # Claude client (optional)
|   +-- reranker.py              # Result reranking
|   +-- kpi_extractor.py         # KPI extraction
|   +-- streamlit_dashboard.py   # Visualization dashboard
|   +-- batch_runner.py          # Demo batch tester
|   +-- demo_questions.py        # 105 synthetic questions
+-- pipeline/
|   +-- run.py                   # Pipeline orchestrator
|   +-- crawl.py                 # Source fetching
|   +-- clean.py                 # Text extraction + chunking
|   +-- delta.py                 # Weekly comparison
|   +-- evaluate.py              # Bucket classification
|   +-- schedule.py              # Job queue generation
|   +-- index.py                 # Vector + keyword indexing
|   +-- pargv_batch.py           # Weekly digest generation
|   +-- config.py                # Configuration loader
|   +-- database.py              # SQLite schema + queries
|   +-- agents/                  # Multi-agent system
|   +-- phase2_stubs/            # Phase 2 placeholders
+-- scripts/
|   +-- inject_demo_data.py      # Synthetic data generator
|   +-- check_index.py           # Index diagnostic tool
+-- tests/
    +-- test_delta.py
    +-- test_evaluate.py
    +-- test_clean.py
```

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.12+ | |
| HTTP API | FastAPI + Uvicorn | REST endpoints |
| Database | SQLite | Metadata, artifacts, chunks |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 (local) |
| Fallback | scikit-learn TF-IDF | No model download needed |
| Vector Store | NumPy | Cosine similarity search |
| Text Extraction | trafilatura | HTML to text |
| Visualization | Streamlit + Plotly | Interactive dashboards |
| LLM (optional) | Anthropic Claude | Enhanced answer generation |
| HTTP Client | httpx | Async with HTTP/2 |
| Config | YAML | sources.yaml, pipeline.yaml |

## Two Operating Modes

### Local Mode (Default)
- No API key required
- Template-based answer generation from retrieved context
- Full retrieval, reranking, citations, and visualization
- Suitable for exploring data and testing the architecture

### LLM Mode (Optional)
- Requires `ANTHROPIC_API_KEY` in `.env` file
- Claude generates natural language answers
- Same retrieval pipeline underneath
- Enhanced reasoning and synthesis
