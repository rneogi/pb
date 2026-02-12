"""
Clean Pipeline Stage
====================
Converts raw HTML/XML to cleaned markdown with boilerplate removed.
Chunks content for embedding and retrieval.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from html import unescape

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    print("WARNING: trafilatura not installed, using basic HTML cleaning")

from bs4 import BeautifulSoup

from .config import (
    load_pipeline_config, get_week_dirs, ensure_week_dirs,
    DATA_DIR
)
from .database import (
    get_artifacts_by_week, upsert_artifact, insert_chunks,
    db_session
)


def extract_text_trafilatura(html: str, config: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract main text using trafilatura."""
    traf_config = config.get("trafilatura", {})

    result = trafilatura.extract(
        html,
        include_comments=traf_config.get("include_comments", False),
        include_tables=traf_config.get("include_tables", True),
        favor_precision=traf_config.get("favor_precision", True),
        output_format="markdown"
    )

    # Try to extract title
    title = None
    try:
        metadata = trafilatura.extract_metadata(html)
        if metadata:
            title = metadata.title
    except Exception:
        pass

    return result or "", title


def extract_text_basic(html: str) -> Tuple[str, Optional[str]]:
    """Basic HTML to text extraction fallback."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    # Extract title
    title = None
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Get text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n\n".join(lines)

    return text, title


def extract_text(html: str, config: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract main text from HTML."""
    if HAS_TRAFILATURA:
        return extract_text_trafilatura(html, config)
    return extract_text_basic(html)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from character count."""
    return len(text) // chars_per_token


def chunk_text(
    text: str,
    chunk_size_tokens: int = 450,
    chunk_overlap_tokens: int = 80,
    chars_per_token: int = 4
) -> List[Dict[str, Any]]:
    """
    Chunk text into overlapping segments.
    Returns list of chunk dicts with text and position info.
    """
    if not text:
        return []

    chunk_size_chars = chunk_size_tokens * chars_per_token
    overlap_chars = chunk_overlap_tokens * chars_per_token

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size_chars

        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start + chunk_size_chars // 2, end + 100)
            if para_break > start:
                end = para_break

            # Or sentence break
            elif (sentence_break := text.rfind(". ", start + chunk_size_chars // 2, end + 50)) > start:
                end = sentence_break + 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "token_count_approx": estimate_tokens(chunk_text, chars_per_token)
            })
            chunk_index += 1

        # Move start with overlap
        start = end - overlap_chars
        if start >= len(text) - overlap_chars:
            break

    return chunks


def clean_artifact(
    artifact: Dict[str, Any],
    dirs: Dict[str, Path],
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Clean a single artifact: extract text, chunk, save.
    Returns updated artifact metadata or None on failure.
    """
    artifact_id = artifact["artifact_id"]
    source_name = artifact["source_name"]

    # Load raw content
    raw_path = DATA_DIR / artifact["raw_path"]
    if not raw_path.exists():
        print(f"  WARNING: Raw file not found: {raw_path}")
        return None

    html = raw_path.read_text(encoding="utf-8", errors="replace")

    # Extract text
    text, extracted_title = extract_text(html, config)

    if not text or len(text) < config.get("min_content_length", 100):
        print(f"  Skipping {artifact_id}: insufficient content ({len(text) if text else 0} chars)")
        return None

    # Use extracted title if not already set
    title = artifact.get("title") or extracted_title or "Untitled"

    # Create clean directory structure
    source_dir_name = re.sub(r'[^\w\-]', '_', source_name)[:50]
    clean_dir = dirs["clean"] / source_dir_name
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned markdown
    clean_path = clean_dir / f"{artifact_id}.md"
    clean_content = f"# {title}\n\n{text}"
    clean_path.write_text(clean_content, encoding="utf-8")

    # Chunk the text
    chunk_config = {
        "chunk_size_tokens": config.get("chunk_size_tokens", 450),
        "chunk_overlap_tokens": config.get("chunk_overlap_tokens", 80),
        "chars_per_token": config.get("chars_per_token", 4)
    }
    chunks = chunk_text(text, **chunk_config)

    # Prepare chunk records
    chunk_records = []
    chunk_boundaries = []

    for chunk in chunks:
        chunk_id = f"{artifact_id}_{chunk['chunk_index']:04d}"
        chunk_records.append({
            "chunk_id": chunk_id,
            "artifact_id": artifact_id,
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
            "token_count_approx": chunk["token_count_approx"],
            "week": artifact["week"],
            "source_kind": artifact["source_kind"],
            "source_name": source_name,
            "canonical_url": artifact["canonical_url"],
            "title": title,
            "published_at": artifact.get("published_at"),
            "retrieved_at": artifact["retrieved_at"]
        })
        chunk_boundaries.append({
            "index": chunk["chunk_index"],
            "start": chunk["start_char"],
            "end": chunk["end_char"]
        })

    # Save chunks to database
    if chunk_records:
        insert_chunks(chunk_records)

    # Update artifact metadata
    artifact_update = {
        **artifact,
        "title": title,
        "clean_path": str(clean_path.relative_to(clean_path.parent.parent.parent.parent)),
        "main_text_length": len(text)
    }

    # Update metadata JSON file
    meta_path = DATA_DIR / artifact["meta_path"]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["title"] = title
        meta["clean_path"] = artifact_update["clean_path"]
        meta["main_text_length"] = len(text)
        meta["chunk_count"] = len(chunks)
        meta["chunk_boundaries"] = chunk_boundaries
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Update database
    upsert_artifact(artifact_update)

    print(f"  Cleaned: {artifact_id} - {len(text)} chars, {len(chunks)} chunks")
    return artifact_update


def run_clean(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Run the clean stage for a given week.
    Returns summary of cleaning results.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    clean_cfg = pipeline_config.clean

    print(f"Starting clean for week {week}")

    # Get all artifacts for this week
    artifacts = get_artifacts_by_week(week)
    print(f"Found {len(artifacts)} artifacts to clean")

    cleaned = []
    skipped = []
    errors = []

    for artifact in artifacts:
        try:
            result = clean_artifact(artifact, dirs, clean_cfg)
            if result:
                cleaned.append(result["artifact_id"])
            else:
                skipped.append(artifact["artifact_id"])
        except Exception as e:
            error_msg = f"Error cleaning {artifact['artifact_id']}: {e}"
            print(f"  ERROR: {error_msg}")
            errors.append(error_msg)

    summary = {
        "week": week,
        "total_artifacts": len(artifacts),
        "cleaned": len(cleaned),
        "skipped": len(skipped),
        "errors": len(errors),
        "error_details": errors,
        "completed_at": datetime.utcnow().isoformat()
    }

    # Save clean summary
    summary_path = dirs["runs"] / "clean_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nClean complete: {len(cleaned)} cleaned, {len(skipped)} skipped, {len(errors)} errors")
    return summary


def main(week: str):
    """Entry point for clean stage."""
    return run_clean(week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(week)
