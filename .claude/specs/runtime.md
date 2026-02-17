# RuntimeAgent Spec

**File:** `pipeline/agents/runtime_agent.py`

## Role

Orchestrates the full query lifecycle. Takes a natural language query and returns a grounded answer with citations, confidence, and timing.

## Pattern: PARGV

| Step | Action | Output |
|------|--------|--------|
| **P**arse | Classify intent (funding / acquisition / investor / general) | `intent` |
| **A**bstract | Build retrieval filters from intent | `filter_fn` |
| **R**etrieve | Hybrid search (70% vector + 30% keyword), top-k | `results[]` |
| **G**enerate | LLM (Claude Opus) or template answer with memory context | `answer` |
| **V**alidate | Ground-check answer against citations | `notes[]` |

## Interface

```python
agent = RuntimeAgent(
    rerank_strategy="cross_encoder",
    use_llm=True,           # requires ANTHROPIC_API_KEY
    use_memory=True,        # episodic SmartCard augmentation
)

response = agent.run(
    query="what funding rounds were announced?",
    top_k=8,                # candidates retrieved
    rerank_top_k=5,         # kept after reranking
    mode="hybrid",          # hybrid | vector | keyword
)
```

## Response Contract

```python
{
    "query": str,
    "answer": str,
    "confidence_label": "high" | "medium" | "low",
    "citations": [{"title", "source_name", "score", "snippet", "url"}],
    "query_info": {
        "intent": str,
        "model": str,
        "results_retrieved": int,
        "results_after_rerank": int,
        "memory_augmentation": {"used": bool, "reason": str}
    },
    "timings": {"retrieval_ms", "reranking_ms", "memory_ms", "generation_ms", "total_ms"},
    "notes": [str],         # validation warnings
}
```

## Dependencies

- `MemoryAgent` — provides session context, records claims
- `PresentationAgent` — auto-generates KPI + visualization data
- `app/reranker.py` — cross-encoder or BM25 reranking
- `pipeline/index.py` — vector + keyword retrieval
- `app/llm_client.py` — Claude API wrapper
