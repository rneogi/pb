Run the full PitchBook Observer data pipeline: Ingest → Compile → ready for /pb queries.

## What this does

Runs two agents sequentially:

1. **IngestAgent** — Crawls all enabled public sources (SEC EDGAR Form D, PR Newswire, GlobeNewswire, TechCrunch, Hacker News, company blogs, investor portfolio pages), cleans and deduplicates content, evaluates for deal signals, and saves to the weekly data directory.

2. **CompilationAgent** — Reads the ingested articles, generates embeddings, builds the hybrid vector + keyword index, and writes the updated index to `indexes/chroma/`.

After this completes, `/pb` queries will use the freshly indexed real-world data.

## How to invoke

Use the Bash tool to run both stages sequentially:

```bash
python -m pipeline.agents.ingest_agent && python -m pipeline.agents.compilation_agent
```

If $ARGUMENTS contains `--week YYYY-WNN`, pass it to the ingest stage:

```bash
python -m pipeline.agents.ingest_agent --week $ARGUMENTS && python -m pipeline.agents.compilation_agent --week $ARGUMENTS
```

## Output to report

After completion, summarize:
- Sources crawled and article counts per source
- Deal signals found vs noise filtered
- Articles indexed into the vector store
- Accenture relationships extracted (count by type)
- CSV saved to `data/private/relationships.csv`
- Claims extracted (count by type: funding_round, acquisition, valuation)
- Claims stored in SQLite `claims` table
- Week label processed (e.g. 2026-W07)
- Any sources that failed (404, timeout, etc.)

## Examples

- `/pipeline` — run ingest + compile for current week
- `/pipeline --week 2026-W06` — reprocess a specific week
