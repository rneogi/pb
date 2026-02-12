"""
Memory Agent (SmartCard)
========================
Context-aware session memory with diff-merge semantics.

Features:
    - Entity-centric context tracking
    - Knowledge claims with confidence scores
    - Epistemic metadata (discrepancies, missing data)
    - Diff-merge updates (only adds new information)
    - Context change detection for hallucination prevention

Schema follows structured SmartCard format for query chains.
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .agent_base import AgentBase, MEMORY_DIR

# Memory file path
SMARTCARD_FILE = MEMORY_DIR / "smartcard.json"


@dataclass
class EntityContext:
    """Primary entity being discussed in the query chain."""
    canonical_id: str
    legal_name: str
    primary_domain: str = ""
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "company"  # company, investor, person, fund


@dataclass
class DealInfo:
    """Last confirmed deal information."""
    amount: str = ""
    date: str = ""
    round_type: str = ""
    source_url: str = ""
    confidence: float = 0.0


@dataclass
class QueryState:
    """Current query state in the chain."""
    intent: str  # funding_inquiry, competitor_analysis, deal_tracking, etc.
    query_text: str
    timestamp: str
    last_confirmed_deal: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeClaim:
    """A factual claim extracted from responses."""
    claim_id: str
    subject: str
    predicate: str  # raised, acquired, valued_at, competitor, partner, etc.
    object_value: str
    confidence: float
    source_ids: List[str] = field(default_factory=list)
    timestamp: str = ""
    verified: bool = False


@dataclass
class EpistemicMetadata:
    """Metadata about knowledge quality and gaps."""
    unresolved_discrepancies: List[str] = field(default_factory=list)
    missing_data_points: List[str] = field(default_factory=list)
    ranker_threshold_status: str = "UNKNOWN"  # PASS, FAIL, UNKNOWN
    last_high_confidence_query: str = ""
    context_coherence_score: float = 1.0


@dataclass
class EntityRelationship:
    """Relationship between two entities for network graph."""
    source_entity: str
    target_entity: str
    relation_type: str  # investor_of, acquired, competitor, partner, subsidiary
    strength: float = 0.5  # 0-1 relationship strength
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class TimeSeriesPoint:
    """Single point in time series data."""
    entity: str
    metric: str  # funding_total, valuation, headcount, etc.
    value: float
    unit: str
    timestamp: str
    source: str = ""


@dataclass
class ComparisonSnapshot:
    """Snapshot comparing multiple entities on same metrics."""
    snapshot_id: str
    timestamp: str
    entities: List[str]
    metrics: Dict[str, Dict[str, Any]]  # {metric_name: {entity: value}}
    query_context: str = ""


@dataclass
class SmartCard:
    """
    Enhanced memory card for query chain context.

    Captures entity-centric knowledge with epistemic awareness.
    Supports diff-merge updates to accumulate knowledge.

    Enhanced features for visualization:
    - relationship_graph: Entity relationships for network visualization
    - time_series: Metrics over time for trend analysis
    - comparison_snapshots: Entity comparisons for matrix visualization
    - statistical_summary: Aggregate statistics for instruments
    """
    card_id: str
    created_at: str
    updated_at: str
    entity_context: Optional[Dict[str, Any]] = None
    current_query_state: Optional[Dict[str, Any]] = None
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_claims: List[Dict[str, Any]] = field(default_factory=list)
    epistemic_metadata: Optional[Dict[str, Any]] = None
    sources_used: List[str] = field(default_factory=list)
    total_queries: int = 0

    # Enhanced fields for visualization support
    relationship_graph: List[Dict[str, Any]] = field(default_factory=list)
    time_series: List[Dict[str, Any]] = field(default_factory=list)
    comparison_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    entity_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SmartCard':
        # Filter to only known fields to handle schema changes gracefully
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        # Ensure required fields exist
        if not all(k in filtered for k in ('card_id', 'created_at', 'updated_at')):
            return cls.create_new()
        return cls(**filtered)

    @classmethod
    def create_new(cls) -> 'SmartCard':
        """Create a fresh SmartCard."""
        now = datetime.utcnow().isoformat()
        return cls(
            card_id=str(uuid.uuid4())[:8],
            created_at=now,
            updated_at=now,
            epistemic_metadata={
                "unresolved_discrepancies": [],
                "missing_data_points": [],
                "ranker_threshold_status": "UNKNOWN",
                "last_high_confidence_query": "",
                "context_coherence_score": 1.0
            },
            relationship_graph=[],
            time_series=[],
            comparison_snapshots=[],
            statistical_summary={
                "total_funding_observed": 0,
                "entities_tracked": 0,
                "avg_confidence": 0,
                "funding_by_round": {},
                "top_investors": [],
                "sector_distribution": {}
            },
            entity_profiles={}
        )


class MemoryAgent(AgentBase):
    """
    SmartCard-based memory with context change detection.

    Key behaviors:
    1. Detects context changes in query trajectory
    2. Diff-merges new information (doesn't overwrite)
    3. Tracks knowledge claims with confidence
    4. Signals when memory should NOT be used (context shift)

    Memory is stored in: data/memory/smartcard.json
    """

    # Intent categories for context detection
    INTENT_KEYWORDS = {
        "funding_inquiry": ["funding", "raised", "investment", "series", "round", "capital"],
        "competitor_analysis": ["competitor", "versus", "compare", "alternative", "rival"],
        "deal_tracking": ["deal", "acquisition", "merger", "acquired", "bought"],
        "company_profile": ["about", "who is", "what does", "founded", "headquarters"],
        "market_analysis": ["market", "industry", "sector", "trend", "growth"],
        "leadership": ["ceo", "founder", "executive", "board", "appointed"],
        "valuation": ["valuation", "worth", "valued", "unicorn", "cap"]
    }

    # Entity extraction patterns
    COMPANY_PATTERNS = [
        r'\b(Stripe|OpenAI|Anthropic|Google|Microsoft|Meta|Amazon|Apple|Nvidia|Tesla)\b',
        r'\b(Sequoia|Andreessen Horowitz|a16z|Y Combinator|Benchmark|Accel)\b',
        r'\b([A-Z][a-z]+(?:\.(?:ai|io|com|co))?)\s+(?:Inc|Corp|LLC|Ltd|Technologies|Labs)\b',
    ]

    def __init__(self):
        super().__init__("memory_agent")
        self._card: Optional[SmartCard] = None
        self._previous_entities: Set[str] = set()
        self._previous_intent: str = ""

    def run(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update SmartCard with response using diff-merge.

        Args:
            response: Chat response with query, answer, citations

        Returns:
            Updated SmartCard as dictionary
        """
        # Load existing card or create new
        self._card = self._load_or_create()

        query = response.get("query", "")
        answer = response.get("answer", "")
        citations = response.get("citations", [])
        confidence = response.get("confidence_label", "unknown")

        # Extract current context
        current_entities = self._extract_entities(query + " " + answer)
        current_intent = self._classify_intent(query)

        # Detect context change
        context_changed, coherence_score = self._detect_context_change(
            current_entities, current_intent
        )

        # Update query state
        now = datetime.utcnow().isoformat()
        query_state = {
            "intent": current_intent,
            "query_text": query[:200],
            "timestamp": now,
            "context_changed": context_changed
        }

        # If context changed significantly, update entity context
        if context_changed or self._card.entity_context is None:
            primary_entity = self._get_primary_entity(current_entities, query)
            if primary_entity:
                self._card.entity_context = {
                    "canonical_id": f"uuid-{primary_entity.lower().replace(' ', '-')}-{str(uuid.uuid4())[:4]}",
                    "legal_name": primary_entity,
                    "primary_domain": self._infer_domain(primary_entity),
                    "aliases": list(current_entities - {primary_entity})[:5],
                    "entity_type": "company"
                }

        # Extract and merge knowledge claims
        new_claims = self._extract_claims(answer, citations, current_entities)
        self._merge_claims(new_claims)

        # Update epistemic metadata
        discrepancies = self._detect_discrepancies(answer, citations)
        missing = self._detect_missing_data(query, answer)

        self._card.epistemic_metadata = {
            "unresolved_discrepancies": self._merge_list(
                self._card.epistemic_metadata.get("unresolved_discrepancies", []) if self._card.epistemic_metadata else [],
                discrepancies,
                max_items=5
            ),
            "missing_data_points": self._merge_list(
                self._card.epistemic_metadata.get("missing_data_points", []) if self._card.epistemic_metadata else [],
                missing,
                max_items=5
            ),
            "ranker_threshold_status": "PASS" if confidence == "high" else "FAIL" if confidence == "low" else "UNKNOWN",
            "last_high_confidence_query": query if confidence == "high" else (
                self._card.epistemic_metadata.get("last_high_confidence_query", "") if self._card.epistemic_metadata else ""
            ),
            "context_coherence_score": coherence_score
        }

        # Update query state and history
        self._card.current_query_state = query_state
        self._card.query_history = self._merge_list(
            self._card.query_history,
            [query_state],
            max_items=10
        )

        # Update sources
        new_sources = [c.get("source_name", "") for c in citations if c.get("source_name")]
        self._card.sources_used = list(set(self._card.sources_used + new_sources))[-15:]

        # =====================================================================
        # Enhanced: Extract visualization data
        # =====================================================================

        # Extract relationships for network graph
        new_relationships = self.extract_relationships(answer, citations)
        if new_relationships:
            existing_rels = self._card.relationship_graph or []
            for rel in new_relationships:
                # Check for duplicate
                is_dup = any(
                    r.get("source_entity") == rel.get("source_entity") and
                    r.get("target_entity") == rel.get("target_entity") and
                    r.get("relation_type") == rel.get("relation_type")
                    for r in existing_rels
                )
                if not is_dup:
                    existing_rels.append(rel)
            self._card.relationship_graph = existing_rels[-50:]  # Keep max 50

        # Extract time series points
        new_ts_points = self.extract_time_series_points(answer, current_entities)
        if new_ts_points:
            existing_ts = self._card.time_series or []
            existing_ts.extend(new_ts_points)
            self._card.time_series = existing_ts[-100:]  # Keep max 100

        # Update entity profiles
        for entity in current_entities:
            entity_claims = [c for c in new_claims if c.get("subject", "").lower() == entity.lower()]
            profile_data = {}
            for claim in entity_claims:
                if claim.get("predicate") == "raised":
                    profile_data["funding"] = {
                        "value": claim.get("object_value"),
                        "timestamp": now
                    }
                elif claim.get("predicate") == "valued_at":
                    profile_data["valuation"] = {
                        "value": claim.get("object_value"),
                        "timestamp": now
                    }
            self.update_entity_profile(entity, profile_data)

        # Build comparison snapshot if multiple entities
        if len(current_entities) >= 2:
            snapshot = self.build_comparison_snapshot(current_entities, new_claims, query)
            if snapshot:
                existing_snaps = self._card.comparison_snapshots or []
                existing_snaps.append(snapshot)
                self._card.comparison_snapshots = existing_snaps[-10:]  # Keep max 10

        # Update statistical summary
        self.update_statistical_summary()

        # Update metadata
        self._card.updated_at = now
        self._card.total_queries += 1

        # Save state for next comparison
        self._previous_entities = current_entities
        self._previous_intent = current_intent

        # Persist
        self._save(self._card)

        self.logger.info(
            f"SmartCard updated: entities={len(current_entities)}, "
            f"claims={len(self._card.knowledge_claims)}, "
            f"context_changed={context_changed}, "
            f"coherence={coherence_score:.2f}"
        )

        return self._card.to_dict()

    def load(self) -> Optional[Dict[str, Any]]:
        """Load existing SmartCard."""
        if not SMARTCARD_FILE.exists():
            return None
        try:
            data = json.loads(SMARTCARD_FILE.read_text(encoding="utf-8"))
            return data
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to load SmartCard: {e}")
            return None

    def should_use_memory(self, query: str) -> Tuple[bool, str]:
        """
        Determine if memory should be used for augmentation.

        Memory is ENABLED by default and only disabled when there is
        strong evidence of context divergence. This ensures the SmartCard
        accumulates knowledge across the query chain.

        Returns:
            Tuple of (should_use: bool, reason: str)
        """
        card_data = self.load()
        if not card_data:
            # First query - enable memory to bootstrap the SmartCard
            return True, "initializing_memory"

        # Check if card has any claims yet
        claims = card_data.get("knowledge_claims") or []
        if not claims:
            # Card exists but empty - keep building
            return True, "building_memory"

        # Check context coherence - only reject on very low coherence
        epistemic = card_data.get("epistemic_metadata") or {}
        coherence = epistemic.get("context_coherence_score", 0.5)

        if coherence < 0.2:
            # Severe divergence - but still allow with warning
            self.logger.warning(f"Low coherence ({coherence:.2f}) but keeping memory enabled")
            return True, "low_coherence_warning"

        # Check entity overlap (informational, not blocking)
        current_entities = self._extract_entities(query)
        entity_ctx = card_data.get("entity_context") or {}
        stored_entity = entity_ctx.get("legal_name", "")
        stored_aliases = set(entity_ctx.get("aliases", []))

        if stored_entity and current_entities:
            all_stored = {stored_entity.lower()} | {a.lower() for a in stored_aliases}
            current_lower = {e.lower() for e in current_entities}
            entity_match = bool(all_stored & current_lower)

            if not entity_match:
                # New entity context - still use memory but note the shift
                self.logger.info(f"Entity shift detected: {stored_entity} -> {current_entities}")
                return True, "entity_shift_noted"

        # Check intent continuity (informational, not blocking)
        current_intent = self._classify_intent(query)
        query_state = card_data.get("current_query_state") or {}
        previous_intent = query_state.get("intent", "")

        if previous_intent and current_intent != previous_intent:
            self.logger.info(f"Intent shift: {previous_intent} -> {current_intent}")
            return True, "intent_shift_noted"

        return True, "context_coherent"

    def get_context_for_prompt(self) -> str:
        """
        Get SmartCard context formatted for LLM augmentation.

        Only returns context if it should be used.
        """
        should_use, reason = self.should_use_memory("")

        # Need to check with actual query, but for backward compatibility
        # we'll return the formatted context
        card_data = self.load()
        if not card_data:
            return ""

        sections = []

        # Entity context
        entity = card_data.get("entity_context") or {}
        if entity.get("legal_name"):
            sections.append(f"Primary entity: {entity['legal_name']}")
            if entity.get("aliases"):
                sections.append(f"Also known as: {', '.join(entity['aliases'][:3])}")

        # Last query state
        query_state = card_data.get("current_query_state") or {}
        if query_state.get("query_text"):
            sections.append(f"Previous query: {query_state['query_text'][:100]}...")
            sections.append(f"Previous intent: {query_state.get('intent', 'unknown')}")

        # High-confidence claims
        claims = card_data.get("knowledge_claims") or []
        high_conf_claims = [c for c in claims if c.get("confidence", 0) >= 0.8]
        if high_conf_claims:
            sections.append("Verified facts from previous queries:")
            for claim in high_conf_claims[:3]:
                sections.append(
                    f"  - {claim.get('subject', '?')} {claim.get('predicate', '?')} {claim.get('object_value', '?')}"
                )

        # Epistemic warnings
        epistemic = card_data.get("epistemic_metadata") or {}
        discrepancies = epistemic.get("unresolved_discrepancies") or []
        if discrepancies:
            sections.append(f"Note: Unresolved discrepancy - {discrepancies[0]}")

        if sections:
            return "PREVIOUS SESSION CONTEXT (use if relevant):\n" + "\n".join(sections)

        return ""

    def get_augmentation_decision(self, query: str) -> Dict[str, Any]:
        """
        Get decision about whether to use memory augmentation.

        Memory augmentation is always enabled to ensure the SmartCard
        continuously accumulates knowledge. Context coherence notes
        are logged but do not block augmentation.

        Args:
            query: Current user query

        Returns:
            Dict with use_memory, reason, and optional context
        """
        should_use, reason = self.should_use_memory(query)

        # Always provide context when memory is enabled
        context = self.get_context_for_prompt() if should_use else ""

        result = {
            "use_memory": should_use,
            "reason": reason,
            "context": context,
            "has_prior_data": bool(self.load())
        }

        self.logger.info(f"Memory decision: use={should_use}, reason={reason}, context_len={len(context)}")

        return result

    def clear(self) -> bool:
        """Clear the SmartCard file."""
        if SMARTCARD_FILE.exists():
            SMARTCARD_FILE.unlink()
            self._card = None
            self._previous_entities = set()
            self._previous_intent = ""
            self.logger.info("SmartCard cleared")
            return True
        return False

    def _load_or_create(self) -> SmartCard:
        """Load existing SmartCard or create new."""
        data = self.load()
        if data:
            return SmartCard.from_dict(data)
        return SmartCard.create_new()

    def _save(self, card: SmartCard) -> None:
        """Save SmartCard to file."""
        SMARTCARD_FILE.parent.mkdir(parents=True, exist_ok=True)
        SMARTCARD_FILE.write_text(
            json.dumps(card.to_dict(), indent=2, default=str),
            encoding="utf-8"
        )

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entity names from text."""
        entities = set()

        for pattern in self.COMPANY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                entities.add(match)

        return entities

    def _classify_intent(self, query: str) -> str:
        """Classify query intent."""
        query_lower = query.lower()

        best_intent = "general"
        best_score = 0

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    def _detect_context_change(
        self,
        current_entities: Set[str],
        current_intent: str
    ) -> Tuple[bool, float]:
        """
        Detect if query context has changed significantly.

        Returns:
            Tuple of (context_changed: bool, coherence_score: float)
        """
        if not self._previous_entities and not self._previous_intent:
            # First query - no change
            return False, 1.0

        # Calculate entity overlap
        if self._previous_entities and current_entities:
            prev_lower = {e.lower() for e in self._previous_entities}
            curr_lower = {e.lower() for e in current_entities}

            intersection = prev_lower & curr_lower
            union = prev_lower | curr_lower

            entity_overlap = len(intersection) / len(union) if union else 0
        else:
            entity_overlap = 0.5  # No entities to compare

        # Calculate intent similarity
        intent_match = 1.0 if current_intent == self._previous_intent else 0.3

        # Combined coherence score
        coherence_score = (entity_overlap * 0.7) + (intent_match * 0.3)

        # Context changed if coherence drops below threshold
        context_changed = coherence_score < 0.4

        return context_changed, coherence_score

    def _get_primary_entity(self, entities: Set[str], query: str) -> Optional[str]:
        """Get the primary entity from a set."""
        if not entities:
            return None

        # Prioritize entity that appears first in query
        query_lower = query.lower()
        for entity in entities:
            if entity.lower() in query_lower:
                return entity

        return list(entities)[0]

    def _infer_domain(self, entity: str) -> str:
        """Infer domain from entity name."""
        clean = entity.lower().replace(" ", "").replace(",", "").replace("inc", "").replace("corp", "")
        return f"{clean}.com"

    def _extract_claims(
        self,
        answer: str,
        citations: List[Dict],
        entities: Set[str]
    ) -> List[Dict[str, Any]]:
        """Extract knowledge claims from answer."""
        claims = []

        # Funding claims
        funding_pattern = r'(\w+(?:\s+\w+)?)\s+(?:raised|secured|received)\s+\$?([\d.]+)\s*(million|billion|M|B)'
        for match in re.finditer(funding_pattern, answer, re.IGNORECASE):
            subject = match.group(1)
            amount = match.group(2)
            unit = match.group(3)

            # Normalize unit
            if unit.lower() in ('billion', 'b'):
                amount_str = f"${amount}B"
            else:
                amount_str = f"${amount}M"

            claims.append({
                "claim_id": f"c_{str(uuid.uuid4())[:6]}",
                "subject": subject,
                "predicate": "raised",
                "object_value": amount_str,
                "confidence": 0.8 if any(subject.lower() in str(c).lower() for c in citations) else 0.5,
                "source_ids": [c.get("url", "")[:50] for c in citations[:2]],
                "timestamp": datetime.utcnow().isoformat(),
                "verified": False
            })

        # Valuation claims
        val_pattern = r'(\w+(?:\s+\w+)?)\s+(?:valued at|valuation of)\s+\$?([\d.]+)\s*(million|billion|M|B)'
        for match in re.finditer(val_pattern, answer, re.IGNORECASE):
            subject = match.group(1)
            amount = match.group(2)
            unit = match.group(3)

            if unit.lower() in ('billion', 'b'):
                amount_str = f"${amount}B"
            else:
                amount_str = f"${amount}M"

            claims.append({
                "claim_id": f"c_{str(uuid.uuid4())[:6]}",
                "subject": subject,
                "predicate": "valued_at",
                "object_value": amount_str,
                "confidence": 0.7,
                "source_ids": [c.get("url", "")[:50] for c in citations[:2]],
                "timestamp": datetime.utcnow().isoformat(),
                "verified": False
            })

        # Acquisition claims
        acq_pattern = r'(\w+(?:\s+\w+)?)\s+(?:acquired|bought|purchased)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(acq_pattern, answer, re.IGNORECASE):
            acquirer = match.group(1)
            target = match.group(2)

            claims.append({
                "claim_id": f"c_{str(uuid.uuid4())[:6]}",
                "subject": acquirer,
                "predicate": "acquired",
                "object_value": target,
                "confidence": 0.75,
                "source_ids": [c.get("url", "")[:50] for c in citations[:2]],
                "timestamp": datetime.utcnow().isoformat(),
                "verified": False
            })

        return claims

    def _merge_claims(self, new_claims: List[Dict[str, Any]]) -> None:
        """Merge new claims into existing, avoiding duplicates."""
        existing = self._card.knowledge_claims or []

        for new_claim in new_claims:
            # Check for duplicate (same subject + predicate + object)
            is_duplicate = False
            for existing_claim in existing:
                if (existing_claim.get("subject", "").lower() == new_claim.get("subject", "").lower() and
                    existing_claim.get("predicate") == new_claim.get("predicate") and
                    existing_claim.get("object_value") == new_claim.get("object_value")):
                    # Update confidence if new is higher
                    if new_claim.get("confidence", 0) > existing_claim.get("confidence", 0):
                        existing_claim["confidence"] = new_claim["confidence"]
                        existing_claim["source_ids"] = list(set(
                            existing_claim.get("source_ids", []) + new_claim.get("source_ids", [])
                        ))[:5]
                    is_duplicate = True
                    break

            if not is_duplicate:
                existing.append(new_claim)

        # Keep only most recent/confident claims
        existing.sort(key=lambda x: (x.get("confidence", 0), x.get("timestamp", "")), reverse=True)
        self._card.knowledge_claims = existing[:20]

    def _merge_list(
        self,
        existing: List[Any],
        new_items: List[Any],
        max_items: int = 10
    ) -> List[Any]:
        """Merge lists, avoiding exact duplicates."""
        result = list(existing)

        for item in new_items:
            # Check for duplicates (works for strings and simple dicts)
            if isinstance(item, str):
                if item not in result:
                    result.append(item)
            elif isinstance(item, dict):
                # For dicts, check if similar dict exists
                is_dup = False
                for r in result:
                    if isinstance(r, dict):
                        # Simple similarity check
                        if r.get("query_text") == item.get("query_text"):
                            is_dup = True
                            break
                if not is_dup:
                    result.append(item)
            else:
                result.append(item)

        return result[-max_items:]

    def _detect_discrepancies(
        self,
        answer: str,
        citations: List[Dict]
    ) -> List[str]:
        """Detect potential discrepancies in the data."""
        discrepancies = []

        # Look for conflicting series mentions
        series_mentions = re.findall(r'series\s+([A-H])', answer, re.IGNORECASE)
        if len(set(series_mentions)) > 1:
            discrepancies.append(
                f"Multiple funding series mentioned: {', '.join(set(series_mentions))}"
            )

        # Look for different amounts for same entity
        amounts = re.findall(r'(\w+)\s+raised\s+\$([\d.]+)\s*(M|B|million|billion)', answer, re.IGNORECASE)
        by_entity = {}
        for entity, amount, unit in amounts:
            key = entity.lower()
            if key not in by_entity:
                by_entity[key] = []
            by_entity[key].append(f"${amount}{unit[0].upper()}")

        for entity, entity_amounts in by_entity.items():
            if len(set(entity_amounts)) > 1:
                discrepancies.append(
                    f"Conflicting amounts for {entity}: {', '.join(set(entity_amounts))}"
                )

        return discrepancies[:3]

    def _detect_missing_data(self, query: str, answer: str) -> List[str]:
        """Detect data points that might be missing."""
        missing = []
        query_lower = query.lower()
        answer_lower = answer.lower()

        # Check for common missing patterns
        if "valuation" in query_lower and "valuation" not in answer_lower and "$" not in answer_lower:
            missing.append("Valuation not found in sources")

        if "board" in query_lower and "board" not in answer_lower:
            missing.append("Board member information not available")

        if "when" in query_lower and not re.search(r'\d{4}', answer):
            missing.append("Specific date not found")

        if any(w in query_lower for w in ["investor", "who invested"]) and "invested" not in answer_lower:
            missing.append("Investor details not confirmed")

        return missing[:3]

    # =========================================================================
    # Visualization Support Methods
    # =========================================================================

    def extract_relationships(self, answer: str, citations: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract entity relationships for network graph visualization.

        Returns list of relationships: {source, target, relation_type, strength}
        """
        relationships = []

        # Investor relationships
        investor_pattern = r'(\w+(?:\s+\w+)?)\s+(?:invested in|led|participated in|backed)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(investor_pattern, answer, re.IGNORECASE):
            relationships.append({
                "source_entity": match.group(1),
                "target_entity": match.group(2),
                "relation_type": "investor_of",
                "strength": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Acquisition relationships
        acq_pattern = r'(\w+(?:\s+\w+)?)\s+(?:acquired|bought|purchased)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(acq_pattern, answer, re.IGNORECASE):
            relationships.append({
                "source_entity": match.group(1),
                "target_entity": match.group(2),
                "relation_type": "acquired",
                "strength": 1.0,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Competitor relationships
        comp_pattern = r'(\w+(?:\s+\w+)?)\s+(?:competes with|rival of|competitor to)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(comp_pattern, answer, re.IGNORECASE):
            relationships.append({
                "source_entity": match.group(1),
                "target_entity": match.group(2),
                "relation_type": "competitor",
                "strength": 0.6,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Partnership relationships
        partner_pattern = r'(\w+(?:\s+\w+)?)\s+(?:partnered with|partnership with|collaborat\w+ with)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(partner_pattern, answer, re.IGNORECASE):
            relationships.append({
                "source_entity": match.group(1),
                "target_entity": match.group(2),
                "relation_type": "partner",
                "strength": 0.7,
                "timestamp": datetime.utcnow().isoformat()
            })

        return relationships

    def extract_time_series_points(self, answer: str, entities: Set[str]) -> List[Dict[str, Any]]:
        """
        Extract time series data points for trend visualization.

        Returns list of points: {entity, metric, value, unit, timestamp}
        """
        points = []
        now = datetime.utcnow().isoformat()

        # Funding amounts with dates
        funding_pattern = r'(\w+(?:\s+\w+)?)\s+raised\s+\$?([\d.]+)\s*(million|billion|M|B)(?:\s+in\s+(\d{4}|\w+\s+\d{4}))?'
        for match in re.finditer(funding_pattern, answer, re.IGNORECASE):
            entity = match.group(1)
            amount = float(match.group(2))
            unit = match.group(3)
            date = match.group(4) if match.group(4) else "Recent"

            # Normalize to millions
            if unit.lower() in ('billion', 'b'):
                amount *= 1000

            points.append({
                "entity": entity,
                "metric": "funding",
                "value": amount,
                "unit": "$M",
                "timestamp": date,
                "source": "llm_answer"
            })

        # Valuation data
        val_pattern = r'(\w+(?:\s+\w+)?)\s+(?:valued at|valuation of)\s+\$?([\d.]+)\s*(million|billion|M|B)'
        for match in re.finditer(val_pattern, answer, re.IGNORECASE):
            entity = match.group(1)
            amount = float(match.group(2))
            unit = match.group(3)

            if unit.lower() in ('billion', 'b'):
                amount *= 1000

            points.append({
                "entity": entity,
                "metric": "valuation",
                "value": amount,
                "unit": "$M",
                "timestamp": now,
                "source": "llm_answer"
            })

        # Headcount/employee data
        headcount_pattern = r'(\w+(?:\s+\w+)?)\s+(?:has|employs|with)\s+(\d+(?:,\d+)?)\s+(?:employees|workers|staff)'
        for match in re.finditer(headcount_pattern, answer, re.IGNORECASE):
            entity = match.group(1)
            count = int(match.group(2).replace(',', ''))

            points.append({
                "entity": entity,
                "metric": "headcount",
                "value": count,
                "unit": "employees",
                "timestamp": now,
                "source": "llm_answer"
            })

        return points

    def build_comparison_snapshot(
        self,
        entities: Set[str],
        claims: List[Dict[str, Any]],
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Build a comparison snapshot for matrix visualization.

        Returns: {snapshot_id, timestamp, entities, metrics}
        """
        if len(entities) < 2:
            return None

        metrics = {
            "funding": {},
            "valuation": {},
            "round_stage": {},
            "sector": {}
        }

        for claim in claims:
            subject = claim.get("subject", "")
            predicate = claim.get("predicate", "")
            value = claim.get("object_value", "")

            if subject in entities:
                if predicate == "raised":
                    metrics["funding"][subject] = value
                elif predicate == "valued_at":
                    metrics["valuation"][subject] = value

        # Only create snapshot if we have data
        if any(metrics[m] for m in metrics):
            return {
                "snapshot_id": f"snap_{str(uuid.uuid4())[:6]}",
                "timestamp": datetime.utcnow().isoformat(),
                "entities": list(entities),
                "metrics": metrics,
                "query_context": query[:100]
            }

        return None

    def update_entity_profile(self, entity: str, data: Dict[str, Any]) -> None:
        """Update or create entity profile for comparison."""
        if not self._card:
            return

        if entity not in self._card.entity_profiles:
            self._card.entity_profiles[entity] = {
                "name": entity,
                "first_seen": datetime.utcnow().isoformat(),
                "funding_history": [],
                "valuations": [],
                "relationships": [],
                "sectors": [],
                "mentions_count": 0
            }

        profile = self._card.entity_profiles[entity]
        profile["last_seen"] = datetime.utcnow().isoformat()
        profile["mentions_count"] += 1

        # Merge new data
        if data.get("funding"):
            profile["funding_history"].append(data["funding"])
        if data.get("valuation"):
            profile["valuations"].append(data["valuation"])
        if data.get("sector"):
            if data["sector"] not in profile["sectors"]:
                profile["sectors"].append(data["sector"])

    def update_statistical_summary(self) -> None:
        """Update aggregate statistics for visualization."""
        if not self._card:
            return

        claims = self._card.knowledge_claims or []
        profiles = self._card.entity_profiles or {}

        # Calculate totals
        total_funding = 0
        funding_by_round = {}
        confidences = []

        for claim in claims:
            confidences.append(claim.get("confidence", 0))

            if claim.get("predicate") == "raised":
                value_str = claim.get("object_value", "")
                # Parse value
                match = re.search(r'\$?([\d.]+)\s*(M|B)', value_str)
                if match:
                    amount = float(match.group(1))
                    if match.group(2) == 'B':
                        amount *= 1000
                    total_funding += amount

                    # Track by round
                    subject = claim.get("subject", "Unknown")
                    if subject not in funding_by_round:
                        funding_by_round[subject] = 0
                    funding_by_round[subject] += amount

        self._card.statistical_summary = {
            "total_funding_observed": total_funding,
            "entities_tracked": len(profiles),
            "claims_count": len(claims),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "funding_by_entity": funding_by_round,
            "relationship_count": len(self._card.relationship_graph or []),
            "time_series_points": len(self._card.time_series or [])
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get all data needed for visualization agent.

        Returns comprehensive data structure for charts.
        """
        card_data = self.load()
        if not card_data:
            return {"status": "no_data"}

        return {
            "status": "ok",
            "entity_context": card_data.get("entity_context", {}),
            "knowledge_claims": card_data.get("knowledge_claims", []),
            "relationship_graph": card_data.get("relationship_graph", []),
            "time_series": card_data.get("time_series", []),
            "comparison_snapshots": card_data.get("comparison_snapshots", []),
            "statistical_summary": card_data.get("statistical_summary", {}),
            "entity_profiles": card_data.get("entity_profiles", {}),
            "query_history": card_data.get("query_history", []),
            "epistemic_metadata": card_data.get("epistemic_metadata", {})
        }

    def get_network_graph_data(self) -> Dict[str, Any]:
        """Get data formatted for network graph visualization."""
        card_data = self.load()
        if not card_data:
            return {"nodes": [], "edges": []}

        relationships = card_data.get("relationship_graph", [])
        profiles = card_data.get("entity_profiles", {})

        # Build nodes
        nodes = []
        seen_entities = set()

        for rel in relationships:
            for entity in [rel.get("source_entity"), rel.get("target_entity")]:
                if entity and entity not in seen_entities:
                    seen_entities.add(entity)
                    profile = profiles.get(entity, {})
                    nodes.append({
                        "id": entity,
                        "label": entity,
                        "type": profile.get("type", "company"),
                        "size": profile.get("mentions_count", 1),
                        "funding": sum(f.get("value", 0) for f in profile.get("funding_history", []) if isinstance(f, dict))
                    })

        # Build edges
        edges = []
        for rel in relationships:
            edges.append({
                "source": rel.get("source_entity"),
                "target": rel.get("target_entity"),
                "type": rel.get("relation_type"),
                "weight": rel.get("strength", 0.5)
            })

        return {"nodes": nodes, "edges": edges}

    def get_comparison_matrix_data(self) -> Dict[str, Any]:
        """Get data formatted for comparison matrix visualization."""
        card_data = self.load()
        if not card_data:
            return {"entities": [], "metrics": {}, "values": {}}

        profiles = card_data.get("entity_profiles", {})
        claims = card_data.get("knowledge_claims", [])

        entities = list(profiles.keys())
        metrics = ["funding", "valuation", "mentions", "relationships"]

        values = {}
        for entity in entities:
            profile = profiles.get(entity, {})
            values[entity] = {
                "funding": sum(f.get("value", 0) for f in profile.get("funding_history", []) if isinstance(f, dict)),
                "valuation": profile.get("valuations", [{}])[-1].get("value", 0) if profile.get("valuations") else 0,
                "mentions": profile.get("mentions_count", 0),
                "relationships": len([r for r in card_data.get("relationship_graph", [])
                                     if entity in [r.get("source_entity"), r.get("target_entity")]])
            }

        return {
            "entities": entities,
            "metrics": metrics,
            "values": values
        }

    def get_waterfall_data(self, entity: Optional[str] = None) -> Dict[str, Any]:
        """Get data formatted for waterfall chart (cumulative funding)."""
        card_data = self.load()
        if not card_data:
            return {"steps": [], "total": 0}

        time_series = card_data.get("time_series", [])
        funding_points = [p for p in time_series if p.get("metric") == "funding"]

        if entity:
            funding_points = [p for p in funding_points if p.get("entity") == entity]

        # Sort by timestamp
        funding_points.sort(key=lambda x: x.get("timestamp", ""))

        steps = []
        cumulative = 0

        for point in funding_points:
            value = point.get("value", 0)
            steps.append({
                "entity": point.get("entity"),
                "round": point.get("timestamp"),
                "amount": value,
                "cumulative": cumulative + value
            })
            cumulative += value

        return {
            "steps": steps,
            "total": cumulative,
            "entity_filter": entity
        }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = super().get_status()

        card_data = self.load()
        if card_data:
            status["card_id"] = card_data.get("card_id", "")
            status["total_queries"] = card_data.get("total_queries", 0)
            status["claims_count"] = len(card_data.get("knowledge_claims", []))
            status["entity"] = (card_data.get("entity_context") or {}).get("legal_name", "")
            status["last_intent"] = (card_data.get("current_query_state") or {}).get("intent", "")
            status["coherence"] = (card_data.get("epistemic_metadata") or {}).get("context_coherence_score", 0)
            status["relationships_count"] = len(card_data.get("relationship_graph", []))
            status["time_series_points"] = len(card_data.get("time_series", []))
            status["entity_profiles_count"] = len(card_data.get("entity_profiles", {}))
        else:
            status["has_memory"] = False

        return status


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Agent (SmartCard)")
    parser.add_argument("--load", action="store_true", help="Load and display SmartCard")
    parser.add_argument("--context", action="store_true", help="Show prompt context")
    parser.add_argument("--check", type=str, help="Check if memory should be used for query")
    parser.add_argument("--clear", action="store_true", help="Clear SmartCard")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    agent = MemoryAgent()

    if args.load:
        memory = agent.load()
        if memory:
            print(json.dumps(memory, indent=2))
        else:
            print("No SmartCard found")
    elif args.context:
        context = agent.get_context_for_prompt()
        if context:
            print(context)
        else:
            print("No context available")
    elif args.check:
        decision = agent.get_augmentation_decision(args.check)
        print(json.dumps(decision, indent=2))
    elif args.clear:
        if agent.clear():
            print("SmartCard cleared")
        else:
            print("No SmartCard to clear")
    elif args.status:
        print(json.dumps(agent.get_status(), indent=2))
    else:
        parser.print_help()
