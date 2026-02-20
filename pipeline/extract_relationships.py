"""
Extract Relationships Pipeline Stage
=====================================
Scans cleaned chunks for Accenture partnerships, alliances, client
relationships, and startup portfolio mentions.  Outputs a cumulative CSV
at data/private/relationships.csv.

No LLM calls -- uses regex / keyword matching only.
"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .config import load_pipeline_config, ensure_week_dirs, DATA_DIR
from .database import get_chunks_by_week, get_artifacts_by_week, init_database

# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------
RELATIONSHIPS_DIR = DATA_DIR / "private"
RELATIONSHIPS_CSV = RELATIONSHIPS_DIR / "relationships.csv"

CSV_COLUMNS = [
    "company", "relationship_type", "partner",
    "evidence_url", "confidence", "evidence_snippet",
    "week", "source_name",
]

# ---------------------------------------------------------------------------
# Relationship keyword maps
# ---------------------------------------------------------------------------
RELATIONSHIP_KEYWORDS: Dict[str, List[str]] = {
    "partnership": [
        "partner", "partnership", "partnered", "partnering",
        "teaming", "collaborate", "collaboration", "collaborative",
        "joint", "jointly", "joint venture", "co-develop",
    ],
    "alliance": [
        "alliance", "allied", "ecosystem partner", "alliances",
        "technology partner", "strategic alliance",
    ],
    "investment": [
        "invested", "investment", "funding", "backed", "venture",
        "portfolio company", "incubat",
    ],
    "client": [
        "client", "customer", "deployed", "implemented",
        "selected", "chosen", "engagement", "project for",
        "working with", "contracted",
    ],
    "acquisition": [
        "acquired", "acquisition", "acquires", "merged", "merger",
    ],
}

# Words that look like proper nouns but aren't entity names
ENTITY_STOPWORDS = {
    # Determiners / pronouns / prepositions
    "The", "This", "That", "These", "Those", "Our", "Your", "Its",
    "Their", "His", "Her", "Who", "What", "Which", "Where", "When",
    "Whether", "How", "Why", "Each", "Every", "All", "Any", "Some",
    "of", "for", "and", "or", "the", "in", "on", "at", "to", "with",
    # Corporate suffixes
    "Inc", "Ltd", "Corp", "LLC", "Company", "Group", "Holdings",
    # Sentence-starting words (common false positives from sentence boundaries)
    "According", "Additionally", "Although", "Among", "Another",
    "Based", "Because", "Before", "Between", "Both", "Breaking",
    "Comprehensive", "During", "However", "Including", "Instead",
    "Meanwhile", "Moreover", "Nevertheless", "Overall", "Rather",
    "Similarly", "Since", "Through", "Together", "Under", "While",
    # Generic descriptors often capitalised at sentence start
    "Agentic", "Advanced", "Automated", "Connected", "Digital",
    "Fully", "Integrated", "International", "Leading", "Mobile",
    "Network", "Personalisation", "Programmable", "Verticalised",
    # Generic nouns
    "About", "Read", "More", "View", "See", "Learn", "Click", "Here",
    "Press", "Release", "News", "Today", "Business", "Wire", "Global",
    "Annual", "Report", "Financial", "Results", "Quarter", "Year", "Half",
    "Total", "Key", "Top", "Recent", "Notable", "Follow", "Update",
    "Legal", "License", "Terms", "Service", "Free", "Pro", "Max",
    "Team", "Enterprise", "Commercial", "Consumer", "Healthcare",
    "Compliance", "Zero", "Data", "Security", "Computing", "Energy",
    "Infrastructure", "Solutions", "Systems", "Analytics", "Series",
    "Seed", "Portfolio", "Investments", "Highlights", "For",
    "Agreement", "Terms", "Retention", "Service", "Agreements",
    "Policy", "Privacy", "Clause", "Section", "Appendix",
    "Bedrock", "Vertex", "Claude", "Code", "Consumer",
    "Associate", "Innovation", "Services", "Americas", "Ecosystem",
    "Delivery", "Labs", "Growth", "Opportunities",
    # Accenture internal terms (not entities)
    "Centers of Excellence", "Think Tank",
    # Geographic / directional (not company names)
    "North", "South", "East", "West", "Sea", "Anglia",
    # Sector words (too generic as standalone entities)
    "Robotics", "Renewables", "Automotive", "Aerospace",
    # Time
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
}

# Proper-noun pattern: 1-5 capitalized words
ENTITY_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|&|of|for)){0,4})\b')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> List[str]:
    """Simple sentence splitter."""
    return re.split(r'(?<=[.!?])\s+', text)


def _classify_relationship(sentence: str) -> Tuple[Optional[str], float]:
    """Return (relationship_type, confidence) from keyword scan."""
    s_lower = sentence.lower()
    best_type = None
    best_count = 0

    for rel_type, keywords in RELATIONSHIP_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', s_lower):
                count += 1
        if count > best_count:
            best_count = count
            best_type = rel_type

    if best_type is None:
        return None, 0.0
    confidence = min(1.0, 0.5 + 0.1 * best_count)
    return best_type, confidence


def _extract_entities(sentence: str, exclude: List[str]) -> List[str]:
    """Extract proper-noun candidates, filtering stopwords and exclusions."""
    # Normalise whitespace (collapse newlines, tabs, etc.)
    sentence = re.sub(r'\s+', ' ', sentence)
    exclude_lower = {e.lower() for e in exclude}
    # Build case-insensitive stopword set
    stopwords_lower = {w.lower() for w in ENTITY_STOPWORDS}
    entities = []
    for match in ENTITY_RE.finditer(sentence):
        name = match.group(1).strip()
        # Skip single stopwords (case-insensitive)
        if name.lower() in stopwords_lower:
            continue
        # Skip if majority of words are stopwords
        words = name.split()
        stop_count = sum(1 for w in words if w.lower() in stopwords_lower)
        if stop_count > len(words) / 2:
            continue
        if name.lower() in exclude_lower:
            continue
        if len(name) < 3:
            continue
        # Require at least 2 words OR at least 5 chars for single-word entities
        if len(words) == 1 and len(name) < 5:
            continue
        # Skip common title patterns (person names with titles)
        if re.search(r'\b(Lead|Director|Officer|Manager|Partner|Head|Chief|VP|SVP)\b', name):
            continue
        entities.append(name)
    return list(dict.fromkeys(entities))  # dedupe, preserve order


def _snippet(sentence: str, max_len: int = 200) -> str:
    """Truncate sentence to max_len chars."""
    s = sentence.strip().replace("\n", " ")
    return s[:max_len] + "..." if len(s) > max_len else s


# ---------------------------------------------------------------------------
# Pass 1 — Accenture co-mention scan
# ---------------------------------------------------------------------------

def _pass_accenture_comention(
    chunks: List[Dict], artifacts_by_id: Dict[str, Dict], config: Dict
) -> List[Dict]:
    """Find sentences with Accenture + another entity + relationship keyword."""
    primary = config.get("primary_entity", "Accenture")
    min_conf = config.get("min_confidence", 0.4)
    max_snip = config.get("max_snippet_length", 200)
    rows: List[Dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        if not re.search(r'\b' + re.escape(primary) + r'\b', text, re.IGNORECASE):
            continue

        artifact = artifacts_by_id.get(chunk.get("artifact_id", ""), {})
        url = artifact.get("canonical_url", chunk.get("canonical_url", ""))
        week = chunk.get("week", "")
        source_name = chunk.get("source_name", "")

        for sentence in _sentence_split(text):
            if not re.search(r'\b' + re.escape(primary) + r'\b', sentence, re.IGNORECASE):
                continue

            rel_type, confidence = _classify_relationship(sentence)
            if rel_type is None or confidence < min_conf:
                continue

            entities = _extract_entities(sentence, [primary])
            for entity in entities:
                rows.append({
                    "company": entity,
                    "relationship_type": rel_type,
                    "partner": primary,
                    "evidence_url": url,
                    "confidence": round(confidence, 2),
                    "evidence_snippet": _snippet(sentence, max_snip),
                    "week": week,
                    "source_name": source_name,
                })
    return rows


# ---------------------------------------------------------------------------
# Pass 2 — Accenture Ventures portfolio extraction
# ---------------------------------------------------------------------------

def _pass_ventures_portfolio(
    chunks: List[Dict], artifacts_by_id: Dict[str, Dict], config: Dict
) -> List[Dict]:
    """Extract company names from Accenture Ventures portfolio chunks."""
    max_snip = config.get("max_snippet_length", 200)
    rows: List[Dict] = []

    for chunk in chunks:
        src = (chunk.get("source_name") or "").lower()
        if "accenture" not in src or "ventures" not in src:
            continue

        text = chunk.get("text", "")
        artifact = artifacts_by_id.get(chunk.get("artifact_id", ""), {})
        url = artifact.get("canonical_url", chunk.get("canonical_url", ""))
        week = chunk.get("week", "")
        source_name = chunk.get("source_name", "")

        entities = _extract_entities(text, ["Accenture", "Accenture Ventures"])
        for entity in entities:
            rows.append({
                "company": entity,
                "relationship_type": "venture_portfolio",
                "partner": "Accenture Ventures",
                "evidence_url": url,
                "confidence": 0.85,
                "evidence_snippet": _snippet(text, max_snip),
                "week": week,
                "source_name": source_name,
            })
    return rows


# ---------------------------------------------------------------------------
# Pass 3 — Client cross-reference scan
# ---------------------------------------------------------------------------

def _pass_client_crossref(
    chunks: List[Dict], artifacts_by_id: Dict[str, Dict], config: Dict
) -> List[Dict]:
    """Find client/customer co-mentions between known enterprises."""
    watchlist = config.get("enterprise_watchlist", [])
    if not watchlist:
        return []

    min_conf = config.get("min_confidence", 0.4)
    max_snip = config.get("max_snippet_length", 200)
    client_kw = RELATIONSHIP_KEYWORDS["client"]
    rows: List[Dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        text_lower = text.lower()

        # Quick check: does any watchlist name appear?
        present = [name for name in watchlist
                   if re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE)]
        if not present:
            continue

        # Do any client keywords appear?
        has_client_kw = any(
            re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
            for kw in client_kw
        )
        if not has_client_kw:
            continue

        artifact = artifacts_by_id.get(chunk.get("artifact_id", ""), {})
        url = artifact.get("canonical_url", chunk.get("canonical_url", ""))
        week = chunk.get("week", "")
        source_name = chunk.get("source_name", "")

        for sentence in _sentence_split(text):
            s_lower = sentence.lower()
            has_kw = any(
                re.search(r'\b' + re.escape(kw) + r'\b', s_lower) for kw in client_kw
            )
            if not has_kw:
                continue

            # Which watchlist names appear in this sentence?
            names_in_sent = [n for n in present
                            if re.search(r'\b' + re.escape(n) + r'\b', sentence, re.IGNORECASE)]
            other_entities = _extract_entities(sentence, names_in_sent)

            for enterprise in names_in_sent:
                for entity in other_entities:
                    # Skip self-references (e.g., "Google Vertex" ↔ "Google")
                    if entity.lower().startswith(enterprise.lower()):
                        continue
                    if enterprise.lower().startswith(entity.lower()):
                        continue
                    conf = 0.5
                    if conf < min_conf:
                        continue
                    rows.append({
                        "company": entity,
                        "relationship_type": "client",
                        "partner": enterprise,
                        "evidence_url": url,
                        "confidence": round(conf, 2),
                        "evidence_snippet": _snippet(sentence, max_snip),
                        "week": week,
                        "source_name": source_name,
                    })
    return rows


# ---------------------------------------------------------------------------
# Deduplication & CSV I/O
# ---------------------------------------------------------------------------

def _dedup(rows: List[Dict]) -> List[Dict]:
    """Deduplicate on (company, relationship_type, partner), keep highest confidence."""
    best: Dict[tuple, Dict] = {}
    for r in rows:
        key = (r["company"].lower(), r["relationship_type"], r["partner"].lower())
        if key not in best or r["confidence"] > best[key]["confidence"]:
            best[key] = r
    return list(best.values())


def _load_csv(path: Path) -> List[Dict]:
    """Load existing relationships CSV."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: List[Dict], path: Path):
    """Write relationships CSV with BOM for Excel compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (-r.get("confidence", 0), r.get("company", ""))):
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extract_relationships(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Extract relationships from indexed chunks.

    Returns summary dict with counts.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    config = pipeline_config.raw.get("extract_relationships", {})
    if not config.get("enabled", True):
        print("extract_relationships stage is disabled, skipping")
        return {"status": "skipped"}

    dirs = ensure_week_dirs(week)

    # Load chunks and artifacts
    chunks = get_chunks_by_week(week)
    artifacts = get_artifacts_by_week(week)
    artifacts_by_id = {a["artifact_id"]: a for a in artifacts}

    print(f"Starting relationship extraction for week {week}")
    print(f"  Chunks to scan: {len(chunks)}")

    # Three extraction passes
    rows: List[Dict] = []

    pass1 = _pass_accenture_comention(chunks, artifacts_by_id, config)
    print(f"  Pass 1 (Accenture co-mentions): {len(pass1)} raw matches")
    rows.extend(pass1)

    pass2 = _pass_ventures_portfolio(chunks, artifacts_by_id, config)
    print(f"  Pass 2 (Ventures portfolio):     {len(pass2)} raw matches")
    rows.extend(pass2)

    pass3 = _pass_client_crossref(chunks, artifacts_by_id, config)
    print(f"  Pass 3 (Client cross-ref):       {len(pass3)} raw matches")
    rows.extend(pass3)

    # Merge with existing CSV and deduplicate
    existing = _load_csv(RELATIONSHIPS_CSV)
    # normalise confidence in existing rows
    for r in existing:
        try:
            r["confidence"] = float(r.get("confidence", 0))
        except (ValueError, TypeError):
            r["confidence"] = 0.0

    merged = _dedup(existing + rows)
    _write_csv(merged, RELATIONSHIPS_CSV)

    # Count by type
    by_type: Dict[str, int] = {}
    for r in merged:
        by_type[r["relationship_type"]] = by_type.get(r["relationship_type"], 0) + 1

    summary = {
        "week": week,
        "new_relationships": len(rows),
        "total_relationships": len(merged),
        "by_type": by_type,
        "csv_path": str(RELATIONSHIPS_CSV),
        "completed_at": datetime.utcnow().isoformat(),
    }

    # Save summary alongside other run artifacts
    summary_path = dirs["runs"] / "relationships_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nRelationship extraction complete:")
    print(f"  New this week: {len(rows)}")
    print(f"  Total (cumulative): {len(merged)}")
    for rt, cnt in sorted(by_type.items()):
        print(f"    {rt}: {cnt}")
    print(f"  CSV: {RELATIONSHIPS_CSV}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(week: str = None):
    """Entry point for extract_relationships stage."""
    if week is None:
        from .config import get_current_week
        week = get_current_week()
    init_database()
    return run_extract_relationships(week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    w = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(w)
