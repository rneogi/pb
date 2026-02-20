"""
Extract Claims — Phase 2
=========================
Scans deal-signal chunks for structured claims (funding rounds,
acquisitions, valuations) using regex/keyword matching.  No LLM calls.

Stores claims in the SQLite ``claims`` table and returns a summary.
"""

import hashlib
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from ..config import load_pipeline_config, ensure_week_dirs
from ..database import (
    get_chunks_by_week,
    get_artifacts_by_week,
    get_evaluations_by_week,
    insert_claims,
    get_claims_by_week,
    init_database,
)
from ..extract_relationships import _extract_entities, _sentence_split

# ---------------------------------------------------------------------------
# Regex patterns — detect event + amount (case-insensitive)
# Entity names are extracted separately via _extract_entities()
# ---------------------------------------------------------------------------

# Funding keyword + amount: "raised $150 million"
FUNDING_KW_RE = re.compile(
    r'(?:raised|secured|received|closed|announced)\s+'
    r'\$\s*([\d.,]+)\s*(million|billion|M|B)\b',
    re.IGNORECASE,
)

# Round type: "Series A", "Seed round", "pre-seed"
ROUND_RE = re.compile(
    r'[Ss]eries\s+([A-Z](?:-\d)?)'
    r'|(?:(?:pre-)?[Ss]eed)\s*(?:round|funding)?'
    r'|[Aa]ngel\s*(?:round)?'
    r'|(?:growth|bridge|extension)\s*(?:round|funding)?',
    re.IGNORECASE,
)

# Acquisition keyword: "acquired", "to acquire"
ACQUISITION_KW_RE = re.compile(
    r'\b(?:acquired|acquires|to\s+acquire|bought|purchased|is\s+acquiring)\b',
    re.IGNORECASE,
)

# Valuation keyword + amount: "valued at $10 billion"
VALUATION_KW_RE = re.compile(
    r'(?:valued\s+at|valuation\s+of|valued\s+around|worth)\s+'
    r'\$\s*([\d.,]+)\s*(million|billion|M|B)\b',
    re.IGNORECASE,
)

# Dollar amount (standalone): "$150 million", "$2.5B"
AMOUNT_RE = re.compile(
    r'\$\s*([\d.,]+)\s*(million|billion|M|B)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Headline verbs and noise words to strip/exclude from entity names
_HEADLINE_VERBS = {
    "announces", "announced", "announcing", "announce",
    "acquires", "acquired", "acquiring", "acquire",
    "raises", "raised", "raising", "raise",
    "closes", "closed", "closing", "close",
    "secures", "secured", "securing", "secure",
    "receives", "received", "receiving", "receive",
    "leads", "led", "leading", "lead",
    "launches", "launched", "launching", "launch",
    "reports", "reported", "reporting", "report",
    "invests", "invested", "investing", "invest",
    "reveals", "revealed", "revealing", "reveal",
    "deal", "round", "million", "billion",
}

# Single-word entities that are too generic for claims
_GENERIC_ENTITIES = {
    "labs", "computing", "solutions", "infrastructure", "platform",
    "satellites", "systems", "health", "holdings", "units",
    "consortium", "introduction", "warrant", "share", "common",
    "francisco", "diego", "paris", "london", "headquartered",
}


def _clean_entity(name: str) -> Optional[str]:
    """Strip headline verbs from entity name, return None if nothing left."""
    words = name.split()
    # Strip trailing headline verbs
    while words and words[-1].lower() in _HEADLINE_VERBS:
        words.pop()
    # Strip leading noise
    while words and words[0].lower() in _HEADLINE_VERBS:
        words.pop(0)
    cleaned = " ".join(words).strip()
    if not cleaned or len(cleaned) < 3:
        return None
    # Reject single-word generic entities
    if len(words) == 1 and cleaned.lower() in _GENERIC_ENTITIES:
        return None
    return cleaned


def _get_entities(sentence: str, extra_exclude: List[str] = None) -> List[str]:
    """Extract and clean proper-noun entities from a sentence."""
    exclude = list(extra_exclude or [])
    raw = _extract_entities(sentence, exclude)
    cleaned = []
    for ent in raw:
        c = _clean_entity(ent)
        if c:
            cleaned.append(c)
    return list(dict.fromkeys(cleaned))  # dedupe, preserve order


def _claim_id(claim_type: str, subject: str, predicate: str, value: str, week: str) -> str:
    """Deterministic claim ID from key fields."""
    raw = f"{claim_type}|{subject.lower()}|{predicate}|{value}|{week}"
    return "clm_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _normalize_unit(unit: str) -> str:
    """Normalize 'million'→'M', 'billion'→'B'."""
    u = unit.strip().lower()
    if u in ("million", "m"):
        return "M"
    if u in ("billion", "b"):
        return "B"
    return unit


def _snippet(text: str, max_len: int = 200) -> str:
    s = re.sub(r'\s+', ' ', text).strip()
    return s[:max_len] + "..." if len(s) > max_len else s


def _detect_round_type(sentence: str) -> Optional[str]:
    """Return round type string if found in sentence."""
    m = ROUND_RE.search(sentence)
    if not m:
        return None
    full = m.group(0).lower().strip()
    if m.group(1):
        return f"series_{m.group(1).lower()}"
    if "pre-seed" in full or "preseed" in full:
        return "pre_seed"
    if "seed" in full:
        return "seed"
    if "angel" in full:
        return "angel"
    if "growth" in full:
        return "growth"
    if "bridge" in full:
        return "bridge"
    if "extension" in full:
        return "extension"
    return None


# ---------------------------------------------------------------------------
# Pass 1 — Funding round extraction
# ---------------------------------------------------------------------------

def _pass_funding(
    chunks: List[Dict], artifacts_by_id: Dict[str, Dict], config: Dict
) -> List[Dict]:
    """Extract funding claims by detecting amount pattern + proper-noun entities."""
    min_conf = config.get("confidence_threshold", 0.7)
    max_snip = config.get("max_snippet_length", 200)
    rows: List[Dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        week = chunk.get("week", "")

        for sentence in _sentence_split(text):
            match = FUNDING_KW_RE.search(sentence)
            if not match:
                continue

            amount = match.group(1).replace(",", "")
            unit = _normalize_unit(match.group(2))

            # Extract proper-noun entities from this sentence
            entities = _get_entities(sentence)
            if not entities:
                continue

            subject = entities[0]  # First proper noun is likely the company

            # Detect round type
            round_type = _detect_round_type(sentence)
            predicate = f"raised_{round_type}" if round_type else "raised"

            confidence = 0.75
            if round_type:
                confidence = 0.85
            if "." in amount:
                confidence = min(1.0, confidence + 0.05)

            if confidence < min_conf:
                continue

            # Look for investor entity (second proper noun after "led by" / "from")
            object_entity = None
            led_match = re.search(
                r'(?:led\s+by|from)\s+',
                sentence, re.IGNORECASE,
            )
            if led_match and len(entities) > 1:
                # Find an entity that appears after "led by"
                led_pos = led_match.end()
                for ent in entities[1:]:
                    if sentence.find(ent) >= led_pos:
                        object_entity = ent
                        break

            rows.append({
                "claim_id": _claim_id("funding_round", subject, predicate, amount + unit, week),
                "claim_type": "funding_round",
                "subject_entity_id": subject,
                "object_entity_id": object_entity,
                "predicate": predicate,
                "value": amount,
                "unit": unit,
                "confidence": round(confidence, 2),
                "source_artifact_id": chunk.get("artifact_id", ""),
                "source_chunk_id": chunk.get("chunk_id", ""),
                "evidence_text": _snippet(sentence, max_snip),
                "extracted_at": datetime.utcnow().isoformat(),
                "validated": False,
                "week": week,
            })
    return rows


# ---------------------------------------------------------------------------
# Pass 2 — Acquisition extraction
# ---------------------------------------------------------------------------

def _pass_acquisition(
    chunks: List[Dict], artifacts_by_id: Dict[str, Dict], config: Dict
) -> List[Dict]:
    """Extract acquisition claims by detecting keyword + proper-noun entities."""
    min_conf = config.get("confidence_threshold", 0.7)
    max_snip = config.get("max_snippet_length", 200)
    rows: List[Dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        week = chunk.get("week", "")

        for sentence in _sentence_split(text):
            kw_match = ACQUISITION_KW_RE.search(sentence)
            if not kw_match:
                continue

            entities = _get_entities(sentence)
            if len(entities) < 2:
                continue

            # Acquirer = entity before keyword, Target = entity after keyword
            kw_pos = kw_match.start()
            acquirer = None
            target = None

            for ent in entities:
                ent_pos = sentence.find(ent)
                if ent_pos < kw_pos and acquirer is None:
                    acquirer = ent
                elif ent_pos > kw_pos and target is None:
                    target = ent

            if not acquirer or not target:
                # Fallback: first two entities
                acquirer = entities[0]
                target = entities[1]

            if acquirer.lower() == target.lower():
                continue

            # Try to find amount
            amount_match = AMOUNT_RE.search(sentence)
            value = None
            unit = None
            if amount_match:
                value = amount_match.group(1).replace(",", "")
                unit = _normalize_unit(amount_match.group(2))

            confidence = 0.8
            if value:
                confidence = 0.85

            if confidence < min_conf:
                continue

            rows.append({
                "claim_id": _claim_id("acquisition", acquirer, "acquired", target, week),
                "claim_type": "acquisition",
                "subject_entity_id": acquirer,
                "object_entity_id": target,
                "predicate": "acquired",
                "value": value,
                "unit": unit,
                "confidence": round(confidence, 2),
                "source_artifact_id": chunk.get("artifact_id", ""),
                "source_chunk_id": chunk.get("chunk_id", ""),
                "evidence_text": _snippet(sentence, max_snip),
                "extracted_at": datetime.utcnow().isoformat(),
                "validated": False,
                "week": week,
            })
    return rows


# ---------------------------------------------------------------------------
# Pass 3 — Valuation extraction
# ---------------------------------------------------------------------------

def _pass_valuation(
    chunks: List[Dict], artifacts_by_id: Dict[str, Dict], config: Dict
) -> List[Dict]:
    """Extract valuation claims by detecting keyword + proper-noun entities."""
    min_conf = config.get("confidence_threshold", 0.7)
    max_snip = config.get("max_snippet_length", 200)
    rows: List[Dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        week = chunk.get("week", "")

        for sentence in _sentence_split(text):
            match = VALUATION_KW_RE.search(sentence)
            if not match:
                continue

            amount = match.group(1).replace(",", "")
            unit = _normalize_unit(match.group(2))

            entities = _get_entities(sentence)
            if not entities:
                continue

            subject = entities[0]

            confidence = 0.7
            if "." in amount:
                confidence = 0.75

            if confidence < min_conf:
                continue

            rows.append({
                "claim_id": _claim_id("valuation", subject, "valued_at", amount + unit, week),
                "claim_type": "valuation",
                "subject_entity_id": subject,
                "object_entity_id": None,
                "predicate": "valued_at",
                "value": amount,
                "unit": unit,
                "confidence": round(confidence, 2),
                "source_artifact_id": chunk.get("artifact_id", ""),
                "source_chunk_id": chunk.get("chunk_id", ""),
                "evidence_text": _snippet(sentence, max_snip),
                "extracted_at": datetime.utcnow().isoformat(),
                "validated": False,
                "week": week,
            })
    return rows


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_claims(claims: List[Dict]) -> List[Dict]:
    """Deduplicate on claim_id, keep highest confidence."""
    best: Dict[str, Dict] = {}
    for c in claims:
        cid = c["claim_id"]
        if cid not in best or c.get("confidence", 0) > best[cid].get("confidence", 0):
            best[cid] = c
    return list(best.values())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extract_claims(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Extract structured claims from deal-signal chunks.

    Returns summary dict with counts by type.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    config = pipeline_config.raw.get("phase2", {}).get("claims_extraction", {})
    if not config.get("enabled", False):
        print("[extract_claims] Claims extraction is disabled, skipping")
        return {"status": "skipped", "claims_extracted": 0}

    dirs = ensure_week_dirs(week)

    # Load chunks and filter to deal_signal bucket
    all_chunks = get_chunks_by_week(week)
    evaluations = get_evaluations_by_week(week)
    deal_signal_ids = {
        e["artifact_id"] for e in evaluations
        if e.get("bucket") == "deal_signal"
    }

    chunks = [c for c in all_chunks if c.get("artifact_id") in deal_signal_ids]
    artifacts = get_artifacts_by_week(week)
    artifacts_by_id = {a["artifact_id"]: a for a in artifacts}

    print(f"Starting claims extraction for week {week}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Deal-signal chunks: {len(chunks)} (from {len(deal_signal_ids)} artifacts)")

    # Three extraction passes
    all_claims: List[Dict] = []

    pass1 = _pass_funding(chunks, artifacts_by_id, config)
    print(f"  Pass 1 (Funding rounds):  {len(pass1)} raw claims")
    all_claims.extend(pass1)

    pass2 = _pass_acquisition(chunks, artifacts_by_id, config)
    print(f"  Pass 2 (Acquisitions):    {len(pass2)} raw claims")
    all_claims.extend(pass2)

    pass3 = _pass_valuation(chunks, artifacts_by_id, config)
    print(f"  Pass 3 (Valuations):      {len(pass3)} raw claims")
    all_claims.extend(pass3)

    # Merge with existing claims and deduplicate
    existing = get_claims_by_week(week)
    merged = _dedup_claims(existing + all_claims)

    # Store in database
    inserted = insert_claims(merged)

    # Count by type
    by_type: Dict[str, int] = {}
    for c in merged:
        by_type[c["claim_type"]] = by_type.get(c["claim_type"], 0) + 1

    summary = {
        "week": week,
        "status": "success",
        "claims_extracted": len(all_claims),
        "claims_after_dedup": len(merged),
        "claims_inserted": inserted,
        "by_type": by_type,
        "executed_at": datetime.utcnow().isoformat(),
    }

    print(f"\nClaims extraction complete:")
    print(f"  New claims: {len(all_claims)}")
    print(f"  After dedup: {len(merged)}")
    print(f"  Inserted to DB: {inserted}")
    for ct, cnt in sorted(by_type.items()):
        print(f"    {ct}: {cnt}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(week: str = None):
    """Entry point for claims extraction stage."""
    if week is None:
        from ..config import get_current_week
        week = get_current_week()
    init_database()
    return run_extract_claims(week)


if __name__ == "__main__":
    import sys
    week = sys.argv[1] if len(sys.argv) > 1 else "2026-W01"
    main(week)
