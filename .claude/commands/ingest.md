Run the IngestAgent to crawl and index fresh data from public sources.

## What this does

Crawls all enabled public sources defined in `manifests/sources.yaml`:
- SEC EDGAR Form D filings (real funding disclosures)
- PR Newswire RSS feeds (tech + financial services)
- GlobeNewswire RSS feeds (tech + business)
- TechCrunch RSS
- Hacker News Best
- Company blogs (Stripe, OpenAI, and watchlist entries)
- Investor portfolio pages (a16z, Sequoia, YC, and watchlist entries)

Stages: Crawl → Clean → Delta (dedup) → Evaluate (deal signal scoring)

Output is saved to `data/raw/YYYY-WNN/` and an `ingest_complete` event is emitted.

## How to invoke

```bash
python -m pipeline.agents.ingest_agent
```

With a specific week:

```bash
python -m pipeline.agents.ingest_agent --week $ARGUMENTS
```

## Output to report

- Articles fetched per source
- Deal signals vs noise counts
- Any failed sources
- Output directory path

## Note

After ingest, run `/pipeline` or manually trigger `/compile` to rebuild the search index before querying with `/pb`.
