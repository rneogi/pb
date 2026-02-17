# CompilationAgent Spec

**File:** `pipeline/agents/compilation_agent.py`

## Role

Event-driven knowledge base synthesizer. Watches for `ingest_complete` filesystem events and rebuilds the vector + keyword indexes. Sits between IngestAgent and RuntimeAgent — it's what makes fresh crawl data queryable.

## Position in Pipeline

```
IngestAgent
    └── emits: ingest_complete event (data/events/)
                    │
                    ▼
           CompilationAgent          ← watches data/events/ via watchdog
                    │
              runs: index stage
              (embed chunks → VectorStore + KeywordIndex)
                    │
              emits: compilation_complete event
                    │
                    ▼
           RuntimeAgent can now query fresh data
```

## Interface

```python
agent = CompilationAgent()

# Direct invocation (manual)
result = agent.run(week="2026-W07", full_reindex=False)
# → {"week", "status", "duration_seconds", "index_result", "vectors_added"}

# Event-driven (automatic, requires watchdog)
agent.start_watching()   # starts filesystem observer
agent.stop_watching()
```

## Event Contract

**Consumes:** `ingest_complete`
```json
{"week": "2026-W07", "artifacts_indexed": 116, "timestamp": "..."}
```

**Emits:** `compilation_complete`
```json
{"week": "2026-W07", "success": true, "duration_seconds": 42.3, "vectors_added": 439}
```

Events stored as JSON files in `data/events/`.

## Index Output

- `indexes/chroma/vectors.npy` — dense embeddings (384-dim)
- `indexes/chroma/metadata.json` — chunk metadata
- `indexes/chroma/keyword_index.json` — TF-IDF keyword index

## Dependencies

- `pipeline/index.py` — runs `run_index()` to embed and store chunks
- `watchdog` (optional) — filesystem event watching; falls back to manual invocation
- `AgentBase.emit_event()` — writes event JSON to `data/events/`

## Note

On Streamlit Cloud (stateless), the CompilationAgent is not used — demo data is injected directly into the index at startup via `inject_demo_data.py`. CompilationAgent is relevant for local deployments running the full ingest pipeline.
