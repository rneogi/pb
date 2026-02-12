"""
Schedule Pipeline Stage
=======================
Generates job queue describing downstream actions to run.
Phase 1: index_refresh, pargv_batch_digest
Phase 2 stubs: extract_claims, entity_resolution, assemble_records (disabled)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from .config import load_pipeline_config, ensure_week_dirs


def load_eval_report(week: str) -> Dict[str, Any]:
    """Load evaluation report for the week."""
    report_path = Path(__file__).parent.parent / "runs" / week / "eval_report.json"
    if not report_path.exists():
        return {"buckets": {}, "summary": {}}
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_delta(week: str) -> Dict[str, Any]:
    """Load delta report for the week."""
    delta_path = Path(__file__).parent.parent / "runs" / week / "ingest_delta.json"
    if not delta_path.exists():
        return {"new": [], "changed": [], "stale": []}
    return json.loads(delta_path.read_text(encoding="utf-8"))


def run_schedule(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Run the schedule stage for a given week.
    Generates job queue based on delta and evaluation results.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    schedule_cfg = pipeline_config.schedule

    print(f"Starting scheduling for week {week}")

    # Load inputs
    delta = load_delta(week)
    eval_report = load_eval_report(week)

    # Count fresh artifacts
    fresh_count = len(delta.get("new", [])) + len(delta.get("changed", []))
    has_fresh_content = fresh_count > 0

    # Check bucket counts
    summary = eval_report.get("summary", {})
    has_deal_signals = summary.get("deal_signal", 0) > 0
    has_investor_changes = summary.get("investor_graph_change", 0) > 0
    has_company_changes = summary.get("company_profile_change", 0) > 0

    # Build job queue
    jobs = []

    # Phase 1 jobs (always created, enabled based on content)
    phase1_jobs = schedule_cfg.get("phase1_jobs", ["index_refresh", "pargv_batch_digest"])

    if "index_refresh" in phase1_jobs:
        jobs.append({
            "job_id": f"{week}_index_refresh",
            "job_type": "index_refresh",
            "enabled": has_fresh_content,
            "priority": 1,
            "reason": f"Fresh content: {fresh_count} artifacts" if has_fresh_content else "No fresh content",
            "parameters": {
                "week": week,
                "artifact_count": fresh_count
            }
        })

    if "pargv_batch_digest" in phase1_jobs:
        jobs.append({
            "job_id": f"{week}_pargv_batch_digest",
            "job_type": "pargv_batch_digest",
            "enabled": has_fresh_content,
            "priority": 2,
            "reason": "Generate weekly digest" if has_fresh_content else "No content for digest",
            "parameters": {
                "week": week,
                "deal_signals": summary.get("deal_signal", 0),
                "investor_changes": summary.get("investor_graph_change", 0),
                "company_changes": summary.get("company_profile_change", 0)
            }
        })

    # Phase 2 jobs (stubs - disabled by default)
    phase2_jobs = schedule_cfg.get("phase2_jobs", {})

    jobs.append({
        "job_id": f"{week}_extract_claims",
        "job_type": "extract_claims",
        "enabled": phase2_jobs.get("extract_claims", {}).get("enabled", False),
        "priority": 3,
        "reason": "Phase 2 stub - not implemented",
        "parameters": {
            "week": week,
            "target_buckets": ["deal_signal"]
        },
        "phase": 2,
        "stub": True
    })

    jobs.append({
        "job_id": f"{week}_entity_resolution",
        "job_type": "entity_resolution",
        "enabled": phase2_jobs.get("entity_resolution", {}).get("enabled", False),
        "priority": 4,
        "reason": "Phase 2 stub - not implemented",
        "parameters": {
            "week": week
        },
        "phase": 2,
        "stub": True
    })

    jobs.append({
        "job_id": f"{week}_assemble_records",
        "job_type": "assemble_records",
        "enabled": phase2_jobs.get("assemble_records", {}).get("enabled", False),
        "priority": 5,
        "reason": "Phase 2 stub - not implemented",
        "parameters": {
            "week": week
        },
        "phase": 2,
        "stub": True
    })

    # Build queue
    job_queue = {
        "week": week,
        "generated_at": datetime.utcnow().isoformat(),
        "input_summary": {
            "fresh_artifacts": fresh_count,
            "new_count": len(delta.get("new", [])),
            "changed_count": len(delta.get("changed", [])),
            "stale_count": len(delta.get("stale", [])),
            "eval_summary": summary
        },
        "jobs": jobs,
        "enabled_jobs": [j["job_type"] for j in jobs if j["enabled"]],
        "disabled_jobs": [j["job_type"] for j in jobs if not j["enabled"]]
    }

    # Save job queue
    queue_path = dirs["runs"] / "job_queue.json"
    queue_path.write_text(json.dumps(job_queue, indent=2), encoding="utf-8")

    print(f"\nSchedule complete:")
    print(f"  Enabled jobs: {job_queue['enabled_jobs']}")
    print(f"  Disabled jobs: {job_queue['disabled_jobs']}")

    return job_queue


def main(week: str):
    """Entry point for schedule stage."""
    return run_schedule(week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(week)
