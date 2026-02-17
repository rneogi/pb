# Pipeline Spec

## Overview

End-to-end flow from raw public data to grounded LLM answer.

```
Public Web
    │
    ▼
[IngestAgent]─────────────────────────────────────────────
    crawl → clean → delta → evaluate → schedule → index
                                                    │
                                            indexes/chroma/
                                            (vectors.npy, keyword_index.json)
                                                    │
    ▼
[RuntimeAgent]────────────────────────────────────────────
    query → PARGV → rerank → memory augment → LLM → validate
                │                   │
          [MemoryAgent]      [PresentationAgent]
          SmartCard               KPIs + charts
                                                    │
                                            User Answer
```

## Data Contracts Between Stages

### Crawl → Clean
- Input: URL, source kind, week string
- Output: `data/raw/{week}/{artifact_id}.json` — raw text + metadata

### Clean → Index
- Input: `data/clean/{week}/{artifact_id}.json`
- Schema: `{artifact_id, title, text, source_name, source_kind, canonical_url, week, published_at}`

### Index → RuntimeAgent
- Chunks: 512-token sliding window with 64-token overlap
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), TF-IDF fallback
- Retrieval: cosine similarity (vector) + BM25 (keyword), merged 70/30

### RuntimeAgent → MemoryAgent
Full response dict passed to `MemoryAgent.run()` after each query.

### RuntimeAgent → PresentationAgent
Full response dict passed to `PresentationAgent.run()` after each query.

## Configuration

`manifests/pipeline.yaml` — pipeline-level settings (chunk size, overlap, embedding model)
`manifests/sources.yaml` — crawl sources (URL, kind, enabled, schedule)

## Embedding Strategy

1. **Primary:** `sentence-transformers/all-MiniLM-L6-v2` — local, no API calls
2. **Fallback:** TF-IDF with SVD (scikit-learn) — when sentence-transformers unavailable

## Week Format

`YYYY-WNN` (e.g. `2026-W07`) — ISO week number used as partition key across all data directories.
