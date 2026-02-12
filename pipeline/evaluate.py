"""
Evaluate Pipeline Stage
=======================
Deterministic tagger to bucket each delta artifact into categories:
deal_signal, investor_graph_change, company_profile_change, telemetry_change, noise
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

from .config import load_pipeline_config, ensure_week_dirs, DATA_DIR
from .database import insert_evaluation, get_artifacts_by_week, db_session


def load_delta(week: str) -> Dict[str, Any]:
    """Load delta report for the week."""
    delta_path = Path(__file__).parent.parent / "runs" / week / "ingest_delta.json"
    if not delta_path.exists():
        return {"new": [], "changed": [], "stale": []}
    return json.loads(delta_path.read_text(encoding="utf-8"))


def get_artifact_text(artifact_id: str, week: str) -> str:
    """Get cleaned text for an artifact."""
    artifacts = get_artifacts_by_week(week)
    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            if artifact.get("clean_path"):
                clean_path = DATA_DIR / artifact["clean_path"]
                if clean_path.exists():
                    return clean_path.read_text(encoding="utf-8", errors="replace")
    return ""


def match_keywords(text: str, keywords: List[str]) -> List[str]:
    """Find which keywords match in the text (case-insensitive)."""
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        # Use word boundaries for better matching
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(kw)
    return matched


def evaluate_artifact(
    artifact: Dict[str, Any],
    text: str,
    bucket_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Evaluate a single artifact and assign bucket(s).
    Returns list of bucket assignments with rationale.
    """
    source_kind = artifact.get("source_kind", "unknown")
    results = []

    for bucket_name, bucket_def in bucket_config.items():
        keywords = bucket_def.get("keywords", [])
        source_kinds = bucket_def.get("source_kinds", [])

        # Check if source kind matches (if specified)
        source_match = not source_kinds or source_kind in source_kinds

        # Check keyword matches
        matched_keywords = match_keywords(text, keywords)

        # Determine if bucket applies
        if source_match and matched_keywords:
            confidence = min(1.0, 0.5 + 0.1 * len(matched_keywords))
            rationale = f"Matched keywords: {', '.join(matched_keywords[:5])}"
            if len(matched_keywords) > 5:
                rationale += f" (+{len(matched_keywords) - 5} more)"

            results.append({
                "bucket": bucket_name,
                "confidence": confidence,
                "rationale": rationale,
                "keywords_matched": matched_keywords,
                "source_kind_match": source_kind in source_kinds if source_kinds else True
            })

        # Special case: investor portfolio pages are always investor_graph_change
        elif bucket_name == "investor_graph_change" and source_kind == "investor_portfolio":
            results.append({
                "bucket": bucket_name,
                "confidence": 0.9,
                "rationale": "Source is investor portfolio page",
                "keywords_matched": [],
                "source_kind_match": True
            })

        # Special case: telemetry sources
        elif bucket_name == "telemetry_change" and source_kind == "telemetry":
            results.append({
                "bucket": bucket_name,
                "confidence": 0.9,
                "rationale": "Source is telemetry (careers/ATS)",
                "keywords_matched": [],
                "source_kind_match": True
            })

    # If no buckets matched, classify as noise
    if not results:
        results.append({
            "bucket": "noise",
            "confidence": 0.5,
            "rationale": "No significant signals detected",
            "keywords_matched": [],
            "source_kind_match": False
        })

    return results


def run_evaluate(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Run the evaluate stage for a given week.
    Processes delta artifacts and assigns buckets.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    eval_cfg = pipeline_config.evaluate
    bucket_config = eval_cfg.get("buckets", {})

    print(f"Starting evaluation for week {week}")

    # Load delta
    delta = load_delta(week)
    fresh_artifacts = delta.get("new", []) + delta.get("changed", [])
    print(f"Found {len(fresh_artifacts)} fresh artifacts to evaluate")

    # Evaluation results by bucket
    results_by_bucket: Dict[str, List[Dict]] = {
        "deal_signal": [],
        "investor_graph_change": [],
        "company_profile_change": [],
        "telemetry_change": [],
        "noise": []
    }

    all_evaluations = []

    for artifact in fresh_artifacts:
        artifact_id = artifact["artifact_id"]
        text = get_artifact_text(artifact_id, week)

        if not text:
            print(f"  WARNING: No text for {artifact_id}")
            continue

        # Evaluate
        bucket_results = evaluate_artifact(artifact, text, bucket_config)

        for result in bucket_results:
            bucket = result["bucket"]

            # Store evaluation in database
            insert_evaluation(
                artifact_id=artifact_id,
                week=week,
                bucket=bucket,
                confidence=result["confidence"],
                rationale=result["rationale"],
                keywords_matched=result["keywords_matched"]
            )

            # Add to results
            eval_record = {
                "artifact_id": artifact_id,
                "canonical_url": artifact.get("canonical_url"),
                "source_name": artifact.get("source_name"),
                "source_kind": artifact.get("source_kind"),
                "title": artifact.get("title"),
                "bucket": bucket,
                "confidence": result["confidence"],
                "rationale": result["rationale"],
                "keywords_matched": result["keywords_matched"]
            }

            if bucket in results_by_bucket:
                results_by_bucket[bucket].append(eval_record)
            all_evaluations.append(eval_record)

        print(f"  Evaluated: {artifact_id} -> {[r['bucket'] for r in bucket_results]}")

    # Build report
    report = {
        "week": week,
        "total_evaluated": len(fresh_artifacts),
        "buckets": {
            bucket: {
                "count": len(items),
                "artifacts": items
            }
            for bucket, items in results_by_bucket.items()
        },
        "summary": {
            bucket: len(items)
            for bucket, items in results_by_bucket.items()
        },
        "evaluated_at": datetime.utcnow().isoformat()
    }

    # Save report
    report_path = dirs["runs"] / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nEvaluation complete:")
    for bucket, count in report["summary"].items():
        print(f"  {bucket}: {count}")

    return report


def main(week: str):
    """Entry point for evaluate stage."""
    return run_evaluate(week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(week)
