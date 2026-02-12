"""
Entity Resolution - Phase 2 Stub
================================
Will resolve entity mentions to canonical entities:
- Company name normalization (e.g., "OpenAI", "Open AI" -> canonical)
- Investor identification
- Person disambiguation
- Cross-reference with external identifiers (CIK, LEI, etc.)

Phase 1: No-op stub that returns empty results.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for comparison.

    Phase 2 will implement:
    - Lowercase normalization
    - Punctuation removal
    - Common suffix handling (Inc, LLC, Corp, etc.)
    - Whitespace normalization
    """
    # Basic normalization for Phase 1
    return name.lower().strip()


def find_similar_entities(
    name: str,
    entity_type: str,
    threshold: float = 0.85
) -> List[Tuple[str, float]]:
    """
    Find similar existing entities.

    Phase 2 will implement:
    - Fuzzy string matching
    - Embedding-based similarity
    - Domain/website matching
    - Known aliases lookup

    Args:
        name: Entity name to search
        entity_type: Type of entity (company, investor, person)
        threshold: Minimum similarity threshold

    Returns:
        List of (entity_id, similarity_score) tuples
    """
    # Phase 1: No-op stub
    return []


def resolve_entity(
    mention_text: str,
    context: str,
    entity_type: str
) -> Optional[Dict[str, Any]]:
    """
    Resolve an entity mention to a canonical entity.

    Phase 2 will implement:
    1. Check for exact match in existing entities
    2. Check for fuzzy match above threshold
    3. Use context for disambiguation
    4. Create new entity if no match found

    Args:
        mention_text: The text mention of the entity
        context: Surrounding context text
        entity_type: Type of entity

    Returns:
        Entity dictionary or None if unresolvable
    """
    # Phase 1: No-op stub
    return None


def run_entity_resolution(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Run entity resolution for a given week.

    Phase 2 will:
    1. Load extracted claims
    2. Resolve subject/object entities in claims
    3. Create/update entity records
    4. Link entity mentions to artifacts/chunks

    Args:
        week: Week identifier (e.g., "2026-W05")
        pipeline_config: Optional pipeline configuration

    Returns:
        Summary of resolution results
    """
    print(f"[STUB] entity_resolution for week {week} - not implemented in Phase 1")

    return {
        "week": week,
        "status": "stub",
        "message": "Entity resolution not implemented in Phase 1",
        "entities_resolved": 0,
        "new_entities_created": 0,
        "executed_at": datetime.utcnow().isoformat()
    }


def main(week: str):
    """Entry point for entity resolution."""
    return run_entity_resolution(week)


if __name__ == "__main__":
    import sys
    week = sys.argv[1] if len(sys.argv) > 1 else "2026-W01"
    main(week)
