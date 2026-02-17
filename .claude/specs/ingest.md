# IngestAgent Spec

**File:** `pipeline/agents/ingest_agent.py`

## Role

Orchestrates the data ingestion pipeline. Crawls configured public sources, cleans and normalizes content, detects new/changed artifacts, evaluates quality, schedules for indexing, and updates the vector + keyword indexes.

## Pipeline Stages

```
crawl → clean → delta → evaluate → schedule → index → pargv_batch
```

| Stage | File | Action |
|-------|------|--------|
| **Crawl** | `pipeline/crawl.py` | Fetch HTML/JSON from sources, extract text |
| **Clean** | `pipeline/clean.py` | Normalize, deduplicate, language detect |
| **Delta** | `pipeline/delta.py` | Diff against previous week, detect new/changed |
| **Evaluate** | `pipeline/evaluate.py` | Score relevance, flag low-quality artifacts |
| **Schedule** | `pipeline/schedule.py` | Prioritize artifacts for indexing |
| **Index** | `pipeline/index.py` | Embed chunks → VectorStore + KeywordIndex |
| **Batch** | `pipeline/pargv_batch.py` | Run PARGV at scale over new artifacts |

## Sources Config

`manifests/sources.yaml` — list of crawl targets with URL, kind, and schedule.

Source kinds: `news`, `filing`, `pr_wire`, `investor_db`

## Interface

```python
agent = IngestAgent()
result = agent.run(week="2026-W07")
# → {"artifacts_crawled": int, "artifacts_cleaned": int, "chunks_indexed": int, "errors": []}
```

## Trigger

Currently: manual (human-triggered).
Future: scheduled weekly via cron or Streamlit "Ingest" button.

## Output

- `data/raw/{week}/` — raw crawled HTML/text
- `data/clean/{week}/` — normalized JSON artifacts
- `data/meta/{week}/` — metadata per artifact
- `indexes/chroma/` — updated vector + keyword indexes
