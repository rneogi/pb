"""
Pipeline Orchestrator
=====================
Main entry point for running the Public PitchBook Observer pipeline.

Usage:
    python -m pipeline.run --week 2026-W05
    python -m pipeline.run --week 2026-W05 --only crawl
    python -m pipeline.run --week 2026-W05 --since-week 2026-W04
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .config import get_current_week, ensure_week_dirs, load_pipeline_config
from .database import init_database
from . import crawl, clean, delta, evaluate, schedule, index, pargv_batch
from .phase2_stubs import extract_claims, entity_resolution, assemble_records


STAGES = [
    "crawl",
    "clean",
    "delta",
    "evaluate",
    "schedule",
    "index",
    "pargv_batch"
]

PHASE2_STAGES = [
    "extract_claims",
    "entity_resolution",
    "assemble_records"
]


def run_stage(stage: str, week: str, since_week: Optional[str] = None, config=None) -> dict:
    """Run a single pipeline stage."""
    print(f"\n{'='*60}")
    print(f"Running stage: {stage}")
    print(f"Week: {week}")
    print(f"{'='*60}\n")

    if stage == "crawl":
        return asyncio.run(crawl.run_crawl(week, pipeline_config=config))
    elif stage == "clean":
        return clean.run_clean(week, pipeline_config=config)
    elif stage == "delta":
        return delta.run_delta(week, since_week=since_week, pipeline_config=config)
    elif stage == "evaluate":
        return evaluate.run_evaluate(week, pipeline_config=config)
    elif stage == "schedule":
        return schedule.run_schedule(week, pipeline_config=config)
    elif stage == "index":
        return index.run_index(week, pipeline_config=config)
    elif stage == "pargv_batch":
        return pargv_batch.run_pargv_batch(week, pipeline_config=config)
    # Phase 2 stubs
    elif stage == "extract_claims":
        return extract_claims.run_extract_claims(week, pipeline_config=config)
    elif stage == "entity_resolution":
        return entity_resolution.run_entity_resolution(week, pipeline_config=config)
    elif stage == "assemble_records":
        return assemble_records.run_assemble_records(week, pipeline_config=config)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_pipeline(
    week: str,
    only: Optional[str] = None,
    since_week: Optional[str] = None,
    include_phase2: bool = False
) -> dict:
    """
    Run the full pipeline or a single stage.

    Args:
        week: Target week (e.g., "2026-W05")
        only: If set, run only this stage
        since_week: For delta stage, compare against this week
        include_phase2: Whether to run Phase 2 stubs

    Returns:
        Summary of pipeline run
    """
    start_time = datetime.utcnow()

    print(f"\n{'#'*60}")
    print(f"# Public PitchBook Observer Pipeline")
    print(f"# Week: {week}")
    print(f"# Started: {start_time.isoformat()}")
    print(f"{'#'*60}")

    # Initialize
    print("\nInitializing database...")
    init_database()

    config = load_pipeline_config()
    dirs = ensure_week_dirs(week)

    results = {}
    stages_to_run = [only] if only else STAGES

    # Add Phase 2 stages if requested
    if include_phase2 and not only:
        stages_to_run.extend(PHASE2_STAGES)

    for stage in stages_to_run:
        try:
            result = run_stage(stage, week, since_week, config)
            results[stage] = {
                "status": "success",
                "result": result
            }
        except Exception as e:
            print(f"\nERROR in stage {stage}: {e}")
            results[stage] = {
                "status": "error",
                "error": str(e)
            }
            # Continue with other stages unless it's a critical failure
            if stage in ["crawl", "clean"]:
                print("Critical stage failed, stopping pipeline")
                break

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    # Build summary
    summary = {
        "week": week,
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "duration_seconds": duration,
        "stages_run": list(results.keys()),
        "stages_succeeded": [s for s, r in results.items() if r["status"] == "success"],
        "stages_failed": [s for s, r in results.items() if r["status"] == "error"],
        "results": results
    }

    # Save summary
    summary_path = dirs["runs"] / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'#'*60}")
    print(f"# Pipeline Complete")
    print(f"# Duration: {duration:.1f} seconds")
    print(f"# Succeeded: {len(summary['stages_succeeded'])} stages")
    print(f"# Failed: {len(summary['stages_failed'])} stages")
    print(f"# Summary saved to: {summary_path}")
    print(f"{'#'*60}\n")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Public PitchBook Observer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline for current week
    python -m pipeline.run

    # Run full pipeline for specific week
    python -m pipeline.run --week 2026-W05

    # Run only crawl stage
    python -m pipeline.run --week 2026-W05 --only crawl

    # Run delta with specific comparison week
    python -m pipeline.run --week 2026-W05 --only delta --since-week 2026-W03

    # Include Phase 2 stubs (no-op)
    python -m pipeline.run --week 2026-W05 --include-phase2

Stages:
    crawl       - Fetch content from configured sources
    clean       - Extract text and chunk content
    delta       - Compare with previous week
    evaluate    - Classify content into buckets
    schedule    - Generate job queue
    index       - Build/update vector index
    pargv_batch - Generate weekly digest

Phase 2 Stubs (no-op):
    extract_claims     - Extract structured claims
    entity_resolution  - Resolve entity mentions
    assemble_records   - Assemble deal records
        """
    )

    parser.add_argument(
        "--week",
        type=str,
        default=get_current_week(),
        help="Target week in YYYY-WW format (default: current week)"
    )

    parser.add_argument(
        "--only",
        type=str,
        choices=STAGES + PHASE2_STAGES,
        help="Run only this stage"
    )

    parser.add_argument(
        "--since-week",
        type=str,
        help="For delta stage: compare against this week instead of previous"
    )

    parser.add_argument(
        "--include-phase2",
        action="store_true",
        help="Include Phase 2 stub stages (no-op)"
    )

    args = parser.parse_args()

    try:
        summary = run_pipeline(
            week=args.week,
            only=args.only,
            since_week=args.since_week,
            include_phase2=args.include_phase2
        )

        # Exit with error code if any stage failed
        if summary["stages_failed"]:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
