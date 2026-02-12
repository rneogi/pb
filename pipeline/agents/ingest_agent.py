"""
Ingest Agent
============
Weekly data ingestion orchestrator.

Runs pipeline stages: CRAWL -> CLEAN -> DELTA -> EVALUATE
Emits: ingest_complete event for Compilation Agent

Schedule: Sunday midnight (configurable) + immediate first run
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from .agent_base import AgentBase

# APScheduler imports (optional - graceful fallback if not installed)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    BackgroundScheduler = None
    CronTrigger = None


class IngestAgent(AgentBase):
    """
    Orchestrates weekly data ingestion pipeline.

    Stages executed:
        1. crawl - Fetch content from configured sources
        2. clean - Extract text and chunk content
        3. delta - Compare with previous week
        4. evaluate - Classify content into buckets

    Events emitted:
        - ingest_complete: Triggers Compilation Agent
    """

    STAGES = ["crawl", "clean", "delta", "evaluate"]

    def __init__(self):
        super().__init__("ingest_agent")
        self.scheduler = None

    def run(
        self,
        week: Optional[str] = None,
        since_week: Optional[str] = None,
        stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the ingest pipeline for a given week.

        Args:
            week: Target week (e.g., "2026-W05"). Defaults to current week.
            since_week: For delta stage, compare against this week.
            stages: Optional list of specific stages to run. Defaults to all.

        Returns:
            Summary of pipeline execution with stage results.
        """
        # Import here to avoid circular imports
        from ..config import get_current_week, ensure_week_dirs
        from ..database import init_database
        from .. import crawl, clean, delta, evaluate

        week = week or get_current_week()
        stages_to_run = stages or self.STAGES

        self.logger.info(f"Starting ingest for week: {week}")
        self.logger.info(f"Stages: {stages_to_run}")

        # Initialize
        init_database()
        dirs = ensure_week_dirs(week)

        start_time = datetime.utcnow()
        results = {}
        stages_succeeded = []
        stages_failed = []

        # Execute stages sequentially
        for stage in stages_to_run:
            if stage not in self.STAGES:
                self.logger.warning(f"Unknown stage: {stage}, skipping")
                continue

            self.logger.info(f"Running stage: {stage}")
            stage_start = datetime.utcnow()

            try:
                if stage == "crawl":
                    result = asyncio.run(crawl.run_crawl(week, pipeline_config=self.config))
                elif stage == "clean":
                    result = clean.run_clean(week, pipeline_config=self.config)
                elif stage == "delta":
                    result = delta.run_delta(week, since_week=since_week, pipeline_config=self.config)
                elif stage == "evaluate":
                    result = evaluate.run_evaluate(week, pipeline_config=self.config)
                else:
                    result = {"status": "skipped", "reason": "unknown stage"}

                stage_duration = (datetime.utcnow() - stage_start).total_seconds()
                results[stage] = {
                    "status": "success",
                    "duration_seconds": stage_duration,
                    "result": result
                }
                stages_succeeded.append(stage)
                self.logger.info(f"Stage {stage} completed in {stage_duration:.1f}s")

            except Exception as e:
                stage_duration = (datetime.utcnow() - stage_start).total_seconds()
                self.logger.error(f"Stage {stage} failed: {e}")
                results[stage] = {
                    "status": "error",
                    "duration_seconds": stage_duration,
                    "error": str(e)
                }
                stages_failed.append(stage)

                # Critical failure - stop pipeline
                if stage in ["crawl", "clean"]:
                    self.logger.error("Critical stage failed, stopping ingest")
                    break

        # Calculate total duration
        total_duration = (datetime.utcnow() - start_time).total_seconds()

        # Build summary
        summary = {
            "week": week,
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": total_duration,
            "stages_requested": stages_to_run,
            "stages_succeeded": stages_succeeded,
            "stages_failed": stages_failed,
            "results": results
        }

        # Emit completion event for Compilation Agent
        self.emit_event("ingest_complete", {
            "week": week,
            "stages_completed": stages_succeeded,
            "stages_failed": stages_failed,
            "duration_seconds": total_duration,
            "success": len(stages_failed) == 0
        })

        self.logger.info(
            f"Ingest complete: {len(stages_succeeded)} succeeded, "
            f"{len(stages_failed)} failed in {total_duration:.1f}s"
        )

        return summary

    def start_scheduler(self, immediate_run: bool = True) -> None:
        """
        Start the APScheduler for weekly Sunday midnight runs.

        Args:
            immediate_run: If True, run ingest immediately on start.
        """
        if not HAS_APSCHEDULER:
            self.logger.error(
                "APScheduler not installed. Install with: pip install apscheduler"
            )
            raise ImportError("APScheduler required for scheduling")

        # Get schedule config
        agent_config = self.config.agents if hasattr(self.config, 'agents') else {}
        ingest_config = agent_config.get('ingest', {})
        schedule_config = ingest_config.get('schedule', {})

        day_of_week = schedule_config.get('day_of_week', 'sun')
        hour = schedule_config.get('hour', 0)
        minute = schedule_config.get('minute', 0)

        self.scheduler = BackgroundScheduler()

        # Schedule for Sunday at midnight (or configured time)
        trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        self.scheduler.add_job(
            self.run,
            trigger,
            id='weekly_ingest',
            name='Weekly PitchBook Ingest',
            replace_existing=True
        )

        self.scheduler.start()
        self.logger.info(
            f"Scheduler started - runs every {day_of_week} at {hour:02d}:{minute:02d}"
        )

        if immediate_run:
            self.logger.info("Running immediate ingest...")
            self.run()

    def stop_scheduler(self) -> None:
        """Stop the scheduler gracefully."""
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            self.logger.info("Scheduler stopped")
            self.scheduler = None

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including scheduler info."""
        status = super().get_status()
        status["scheduler_running"] = self.scheduler is not None and self.scheduler.running
        if self.scheduler and self.scheduler.running:
            jobs = self.scheduler.get_jobs()
            status["scheduled_jobs"] = [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in jobs
            ]
        return status


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Agent")
    parser.add_argument("--week", help="Target week (YYYY-WNN)")
    parser.add_argument("--schedule", action="store_true", help="Start scheduler")
    parser.add_argument("--no-immediate", action="store_true", help="Skip immediate run")

    args = parser.parse_args()

    agent = IngestAgent()

    if args.schedule:
        agent.start_scheduler(immediate_run=not args.no_immediate)
        # Keep running
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            agent.stop_scheduler()
    else:
        agent.run(week=args.week)
