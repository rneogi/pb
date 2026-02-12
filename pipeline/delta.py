"""
Delta Pipeline Stage
====================
Compares current week's artifacts against prior weeks to identify
new, changed, and stale content.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

from .config import (
    load_pipeline_config, get_week_dirs, ensure_week_dirs,
    RUNS_DIR
)
from .database import (
    get_artifacts_by_week, insert_delta, get_crawl_state,
    db_session
)


def get_previous_week(week: str, offset: int = 1) -> str:
    """Get the week string for N weeks ago."""
    # Parse week string (YYYY-WW)
    year, week_num = week.split("-W")
    year = int(year)
    week_num = int(week_num)

    # Calculate previous week
    # Create a date from the week number
    from datetime import datetime
    jan1 = datetime(year, 1, 1)
    # Find the Monday of week 1
    days_to_monday = (7 - jan1.weekday()) % 7
    if jan1.weekday() <= 3:  # Thursday or earlier
        days_to_monday -= 7
    week1_monday = jan1 + timedelta(days=days_to_monday)

    # Get the Monday of the target week
    target_monday = week1_monday + timedelta(weeks=week_num - 1)

    # Subtract offset weeks
    prev_monday = target_monday - timedelta(weeks=offset)

    return prev_monday.strftime("%Y-W%W")


def load_previous_week_index(week: str) -> Dict[str, Dict[str, Any]]:
    """
    Load artifact index from previous week.
    Returns dict mapping canonical_url -> artifact metadata.
    """
    artifacts = get_artifacts_by_week(week)
    return {a["canonical_url"]: a for a in artifacts}


def compute_delta(
    current_artifacts: List[Dict[str, Any]],
    previous_index: Dict[str, Dict[str, Any]],
    week: str,
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute delta between current and previous artifacts.
    Returns dict with new, changed, unchanged, stale lists.
    """
    delta = {
        "new": [],
        "changed": [],
        "unchanged": [],
        "stale": []
    }

    current_urls: Set[str] = set()

    for artifact in current_artifacts:
        canonical_url = artifact["canonical_url"]
        current_urls.add(canonical_url)
        content_hash = artifact["content_hash"]

        if canonical_url not in previous_index:
            # New artifact
            delta["new"].append({
                "artifact_id": artifact["artifact_id"],
                "canonical_url": canonical_url,
                "content_hash": content_hash,
                "source_name": artifact["source_name"],
                "source_kind": artifact["source_kind"],
                "title": artifact.get("title"),
                "delta_type": "new"
            })
            insert_delta(week, canonical_url, "new", artifact["artifact_id"],
                        new_hash=content_hash)
        else:
            prev_artifact = previous_index[canonical_url]
            prev_hash = prev_artifact["content_hash"]

            if content_hash != prev_hash:
                # Changed artifact
                delta["changed"].append({
                    "artifact_id": artifact["artifact_id"],
                    "canonical_url": canonical_url,
                    "content_hash": content_hash,
                    "previous_content_hash": prev_hash,
                    "source_name": artifact["source_name"],
                    "source_kind": artifact["source_kind"],
                    "title": artifact.get("title"),
                    "delta_type": "changed"
                })
                insert_delta(week, canonical_url, "changed", artifact["artifact_id"],
                            previous_hash=prev_hash, new_hash=content_hash)
            else:
                # Unchanged
                delta["unchanged"].append({
                    "artifact_id": artifact["artifact_id"],
                    "canonical_url": canonical_url,
                    "content_hash": content_hash,
                    "source_name": artifact["source_name"],
                    "source_kind": artifact["source_kind"],
                    "title": artifact.get("title"),
                    "delta_type": "unchanged"
                })
                insert_delta(week, canonical_url, "unchanged", artifact["artifact_id"])

    # Check for stale (URLs in previous but not current)
    stale_threshold = config.get("stale_threshold_weeks", 4)
    for url, prev_artifact in previous_index.items():
        if url not in current_urls:
            # Check if it's been missing for too long
            crawl_state = get_crawl_state(url)
            if crawl_state:
                consecutive_missing = crawl_state.get("consecutive_unchanged", 0)
                if consecutive_missing >= stale_threshold:
                    delta["stale"].append({
                        "canonical_url": url,
                        "last_artifact_id": prev_artifact["artifact_id"],
                        "source_name": prev_artifact["source_name"],
                        "source_kind": prev_artifact["source_kind"],
                        "title": prev_artifact.get("title"),
                        "delta_type": "stale",
                        "weeks_missing": consecutive_missing
                    })
                    insert_delta(week, url, "stale", prev_artifact["artifact_id"])

    return delta


def run_delta(week: str, since_week: Optional[str] = None, pipeline_config=None) -> Dict[str, Any]:
    """
    Run the delta stage for a given week.
    Compares against previous week (or since_week if specified).
    Returns delta results.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    delta_cfg = pipeline_config.delta

    print(f"Starting delta computation for week {week}")

    # Get current week artifacts
    current_artifacts = get_artifacts_by_week(week)
    print(f"Current week has {len(current_artifacts)} artifacts")

    # Determine comparison week
    compare_weeks_back = delta_cfg.get("compare_weeks_back", 1)
    if since_week:
        prev_week = since_week
    else:
        prev_week = get_previous_week(week, compare_weeks_back)

    print(f"Comparing against week {prev_week}")

    # Load previous week index
    previous_index = load_previous_week_index(prev_week)
    print(f"Previous week has {len(previous_index)} artifacts")

    # Compute delta
    delta = compute_delta(current_artifacts, previous_index, week, delta_cfg)

    # Build result
    result = {
        "week": week,
        "compared_to_week": prev_week,
        "current_artifact_count": len(current_artifacts),
        "previous_artifact_count": len(previous_index),
        "new": delta["new"],
        "changed": delta["changed"],
        "unchanged_count": len(delta["unchanged"]),
        "stale": delta["stale"],
        "summary": {
            "new_count": len(delta["new"]),
            "changed_count": len(delta["changed"]),
            "unchanged_count": len(delta["unchanged"]),
            "stale_count": len(delta["stale"])
        },
        "computed_at": datetime.utcnow().isoformat()
    }

    # Save delta report
    delta_path = dirs["runs"] / "ingest_delta.json"
    delta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nDelta complete:")
    print(f"  New: {len(delta['new'])}")
    print(f"  Changed: {len(delta['changed'])}")
    print(f"  Unchanged: {len(delta['unchanged'])}")
    print(f"  Stale: {len(delta['stale'])}")

    return result


def main(week: str, since_week: Optional[str] = None):
    """Entry point for delta stage."""
    return run_delta(week, since_week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    since = sys.argv[2] if len(sys.argv) > 2 else None
    main(week, since)
