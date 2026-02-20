"""
Compilation Agent
=================
Event-driven knowledge base synthesis.

Triggered by: ingest_complete event from Ingest Agent
Runs: INDEX stage (vector + keyword index building)
Emits: compilation_complete event

Uses file system watching to detect ingest completion events.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_base import AgentBase, EVENTS_DIR

# Watchdog imports (optional - graceful fallback if not installed)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None
    FileSystemEventHandler = object
    FileCreatedEvent = None


class IngestEventHandler(FileSystemEventHandler if HAS_WATCHDOG else object):
    """
    File system event handler for ingest_complete events.
    """

    def __init__(self, compilation_agent: 'CompilationAgent'):
        self.agent = compilation_agent

    def on_created(self, event):
        """Handle file creation events."""
        if hasattr(event, 'src_path'):
            path = Path(event.src_path)
            if path.suffix == '.json' and 'ingest_complete' in path.name:
                self.agent.logger.info(f"Detected ingest event: {path.name}")
                self.agent.handle_ingest_complete(path)


class CompilationAgent(AgentBase):
    """
    Synthesizes knowledge base when ingest completes.

    Responsibilities:
        1. Watch for ingest_complete events
        2. Run index stage (vector + keyword index)
        3. Archive processed events
        4. Emit compilation_complete event

    Events consumed:
        - ingest_complete: From Ingest Agent

    Events emitted:
        - compilation_complete: Signals KB is ready
    """

    def __init__(self):
        super().__init__("compilation_agent")
        self.observer = None
        self._watching = False

    def run(
        self,
        week: str,
        full_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Run knowledge base compilation for a given week.

        Args:
            week: Target week (e.g., "2026-W05")
            full_reindex: If True, reindex all chunks (not just unembedded)

        Returns:
            Compilation results with index statistics
        """
        from ..config import ensure_week_dirs
        from .. import index

        self.logger.info(f"Starting compilation for week: {week}")
        self.logger.info(f"Full reindex: {full_reindex}")

        start_time = datetime.utcnow()
        ensure_week_dirs(week)

        try:
            # Run core index stage
            index_result = index.run_index(
                week,
                pipeline_config=self.config,
                full_reindex=full_reindex
            )

            # Run relationship extraction after indexing
            try:
                from .. import extract_relationships
                rel_result = extract_relationships.run_extract_relationships(
                    week, pipeline_config=self.config
                )
                self.logger.info(
                    f"Extracted {rel_result.get('new_relationships', 0)} relationships"
                )
            except Exception as rel_err:
                self.logger.warning(f"Relationship extraction failed: {rel_err}")

            status = "success"
            error = None

        except Exception as e:
            self.logger.error(f"Index stage failed: {e}")
            index_result = {"error": str(e)}
            status = "error"
            error = str(e)

        total_duration = (datetime.utcnow() - start_time).total_seconds()

        # Build result
        result = {
            "week": week,
            "status": status,
            "started_at": start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "duration_seconds": total_duration,
            "full_reindex": full_reindex,
            "index_result": index_result,
            "error": error
        }

        # Emit completion event
        self.emit_event("compilation_complete", {
            "week": week,
            "success": status == "success",
            "duration_seconds": total_duration,
            "vectors_added": index_result.get("vectors_added", 0) if status == "success" else 0
        })

        self.logger.info(f"Compilation complete in {total_duration:.1f}s")

        return result

    def handle_ingest_complete(self, event_file: Path) -> None:
        """
        Handle an ingest_complete event.

        Args:
            event_file: Path to the event JSON file
        """
        try:
            event = self.read_event(event_file)
            payload = event.get("payload", {})
            week = payload.get("week")
            success = payload.get("success", True)

            if not success:
                self.logger.warning(f"Ingest had failures for {week}, proceeding anyway")

            if not week:
                self.logger.error("No week in ingest event, skipping")
                return

            self.logger.info(f"Processing ingest completion for week: {week}")

            # Run compilation
            self.run(week)

            # Archive the processed event
            self.archive_event(event_file)

        except Exception as e:
            self.logger.error(f"Failed to handle ingest event: {e}")

    def process_pending_events(self) -> int:
        """
        Process any pending ingest_complete events.

        Returns:
            Number of events processed
        """
        events = self.list_pending_events("ingest_complete")
        processed = 0

        for event_file in events:
            self.handle_ingest_complete(event_file)
            processed += 1

        return processed

    def start_watcher(self, process_pending: bool = True) -> None:
        """
        Start watching for ingest_complete events.

        Args:
            process_pending: If True, process any pending events first
        """
        if not HAS_WATCHDOG:
            self.logger.error(
                "Watchdog not installed. Install with: pip install watchdog"
            )
            raise ImportError("Watchdog required for event watching")

        if self._watching:
            self.logger.warning("Watcher already running")
            return

        # Process pending events first
        if process_pending:
            pending = self.process_pending_events()
            if pending:
                self.logger.info(f"Processed {pending} pending events")

        # Start file system observer
        self.observer = Observer()
        handler = IngestEventHandler(self)
        self.observer.schedule(handler, str(EVENTS_DIR), recursive=False)
        self.observer.start()

        self._watching = True
        self.logger.info(f"Watching for events in: {EVENTS_DIR}")

    def stop_watcher(self) -> None:
        """Stop the event watcher gracefully."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self._watching = False
            self.logger.info("Event watcher stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including watcher info."""
        status = super().get_status()
        status["watching"] = self._watching
        status["pending_ingest_events"] = len(self.list_pending_events("ingest_complete"))
        return status


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compilation Agent")
    parser.add_argument("--week", help="Target week (YYYY-WNN) for manual run")
    parser.add_argument("--watch", action="store_true", help="Start event watcher")
    parser.add_argument("--full-reindex", action="store_true", help="Reindex all chunks")

    args = parser.parse_args()

    agent = CompilationAgent()

    if args.watch:
        agent.start_watcher()
        # Keep running
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            agent.stop_watcher()
    elif args.week:
        agent.run(week=args.week, full_reindex=args.full_reindex)
    else:
        # Process any pending events
        processed = agent.process_pending_events()
        print(f"Processed {processed} pending events")
