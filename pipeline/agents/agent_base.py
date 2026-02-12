"""
Agent Base Class
================
Provides common infrastructure for all pipeline agents.

Features:
    - Event emission for inter-agent communication
    - Logging setup
    - Configuration loading
    - Common utilities
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Directory constants
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EVENTS_DIR = DATA_DIR / "events"
MEMORY_DIR = DATA_DIR / "memory"
RESPONSES_DIR = DATA_DIR / "responses"

# Ensure directories exist
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
(EVENTS_DIR / "processed").mkdir(exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)


class AgentBase(ABC):
    """
    Base class for all pipeline agents.

    Provides:
        - Event emission for triggering downstream agents
        - Structured logging
        - Configuration access
        - Common utilities
    """

    def __init__(self, agent_name: str):
        """
        Initialize the agent.

        Args:
            agent_name: Unique identifier for this agent
        """
        self.agent_name = agent_name
        self.logger = self._setup_logger()
        self._config = None

    def _setup_logger(self) -> logging.Logger:
        """Configure agent-specific logger."""
        logger = logging.getLogger(f"agent.{self.agent_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"[%(asctime)s] [{self.agent_name}] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @property
    def config(self):
        """Lazy-load pipeline configuration."""
        if self._config is None:
            from ..config import load_pipeline_config
            self._config = load_pipeline_config()
        return self._config

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main logic.

        Returns:
            Dictionary containing execution results and metadata
        """
        pass

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target_agent: Optional[str] = None
    ) -> Path:
        """
        Emit an event for other agents to consume.

        Args:
            event_type: Type of event (e.g., "ingest_complete", "compilation_complete")
            payload: Event data
            target_agent: Optional specific agent to target

        Returns:
            Path to the created event file
        """
        timestamp = datetime.utcnow()

        event = {
            "event_type": event_type,
            "source_agent": self.agent_name,
            "target_agent": target_agent,
            "timestamp": timestamp.isoformat(),
            "payload": payload
        }

        # Create event file with timestamp-based name
        event_filename = f"{event_type}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        event_file = EVENTS_DIR / event_filename

        event_file.write_text(json.dumps(event, indent=2, default=str), encoding="utf-8")

        self.logger.info(f"Emitted event: {event_type} -> {event_file.name}")

        return event_file

    def read_event(self, event_file: Path) -> Dict[str, Any]:
        """
        Read and parse an event file.

        Args:
            event_file: Path to the event file

        Returns:
            Parsed event dictionary
        """
        return json.loads(event_file.read_text(encoding="utf-8"))

    def archive_event(self, event_file: Path) -> Path:
        """
        Move processed event to archive directory.

        Args:
            event_file: Path to the event file

        Returns:
            Path to archived file
        """
        archive_dir = EVENTS_DIR / "processed"
        archived_path = archive_dir / event_file.name
        event_file.rename(archived_path)
        self.logger.debug(f"Archived event: {event_file.name}")
        return archived_path

    def list_pending_events(self, event_type: Optional[str] = None) -> list:
        """
        List pending (unprocessed) events.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of event file paths
        """
        pattern = f"{event_type}_*.json" if event_type else "*.json"
        events = sorted(EVENTS_DIR.glob(pattern))
        # Exclude directories
        return [e for e in events if e.is_file()]

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.

        Returns:
            Status dictionary with agent metadata
        """
        return {
            "agent_name": self.agent_name,
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "pending_events": len(self.list_pending_events())
        }
