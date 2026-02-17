# MemoryAgent Spec

**File:** `pipeline/agents/memory_agent.py`

## Role

Episodic session memory. Maintains a SmartCard that accumulates knowledge claims, entity profiles, and relationships across queries within a session. Cleared on session start/end (not persistent across sessions).

## SmartCard Schema

```python
@dataclass
class SmartCard:
    card_id: str                    # session identifier
    created_at: str
    updated_at: str
    entity_context: Optional[Dict]  # primary entity focus {legal_name, aliases}
    current_query_state: Dict       # {intent, query_text, timestamp, context_changed}
    query_history: List[Dict]       # last N queries
    knowledge_claims: List[Dict]    # [{subject, predicate, object_value, confidence, source_ids}]
    epistemic_metadata: Dict        # {coherence_score, discrepancies, missing_data}
    relationship_graph: List[Dict]  # [{source_entity, target_entity, relation_type, strength}]
    time_series: List[Dict]         # [{entity, metric, value, timestamp}]
    entity_profiles: Dict           # {entity_name: {mentions_count, funding_history}}
    statistical_summary: Dict
    total_queries: int
```

## Interface

```python
agent = MemoryAgent()

# Get augmentation decision for a query
decision = agent.get_augmentation_decision(query)
# → {"use_memory": bool, "reason": str, "context": str}

# Update SmartCard after a response
agent.run(response_dict)

# Get status for display
status = agent.get_status()
# → {"total_queries", "claims_count", "entity_profiles_count", "coherence", ...}

# Clear for new session
agent.clear()
```

## Context Coherence

Memory is only injected when `coherence_score >= 0.2`. Below that threshold, memory is skipped to prevent hallucination from stale context. Coherence is computed by comparing current query intent/entities to the stored card.

## Storage

Single file: `data/memory/smartcard.json`
Cleared at session start. Not committed to git (in `.gitignore`).
