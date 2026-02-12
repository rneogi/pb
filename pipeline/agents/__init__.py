"""
Pipeline Agents
===============
Multi-agent architecture for Public PitchBook Observer.

Agents:
    - IngestAgent: Weekly data collection (crawl/clean/delta/evaluate)
    - CompilationAgent: Event-driven knowledge base synthesis (index)
    - RuntimeAgent: Query-time retrieval + reranking + LLM generation
    - PresentationAgent: Visualization and KPI matrix rendering
    - MemoryAgent: Session context persistence and retrieval
"""

from .agent_base import AgentBase, EVENTS_DIR, MEMORY_DIR
from .ingest_agent import IngestAgent
from .compilation_agent import CompilationAgent
from .memory_agent import MemoryAgent
from .runtime_agent import RuntimeAgent
from .presentation_agent import PresentationAgent

__all__ = [
    "AgentBase",
    "IngestAgent",
    "CompilationAgent",
    "MemoryAgent",
    "RuntimeAgent",
    "PresentationAgent",
    "EVENTS_DIR",
    "MEMORY_DIR",
]
