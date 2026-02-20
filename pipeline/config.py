"""
Configuration Loading
=====================
Loads and validates pipeline configuration from YAML manifests.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

MANIFESTS_DIR = Path(__file__).parent.parent / "manifests"
DATA_DIR = Path(__file__).parent.parent / "data"
RUNS_DIR = Path(__file__).parent.parent / "runs"
PRODUCTS_DIR = Path(__file__).parent.parent / "products"
INDEXES_DIR = Path(__file__).parent.parent / "indexes"


@dataclass
class SourceEntry:
    """A single source entry to crawl."""
    name: str
    url: str
    format: str  # html, rss, atom
    description: str = ""
    fetch_mode: str = "http"
    enabled: bool = True


@dataclass
class SourceCategory:
    """A category of sources (e.g., pr_wires, news_sources)."""
    name: str
    enabled: bool
    source_kind: str
    fetch_mode: str
    rate_limit: float
    entries: List[SourceEntry] = field(default_factory=list)


@dataclass
class SourcesConfig:
    """Complete sources configuration."""
    settings: Dict[str, Any]
    categories: Dict[str, SourceCategory]

    def get_all_enabled_sources(self) -> List[tuple]:
        """Get all enabled source entries with their category info."""
        sources = []
        for cat_name, category in self.categories.items():
            if not category.enabled:
                continue
            for entry in category.entries:
                if entry.enabled:
                    sources.append((category, entry))
        return sources


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    raw: Dict[str, Any]

    @property
    def crawl(self) -> Dict[str, Any]:
        return self.raw.get("crawl", {})

    @property
    def clean(self) -> Dict[str, Any]:
        return self.raw.get("clean", {})

    @property
    def delta(self) -> Dict[str, Any]:
        return self.raw.get("delta", {})

    @property
    def evaluate(self) -> Dict[str, Any]:
        return self.raw.get("evaluate", {})

    @property
    def schedule(self) -> Dict[str, Any]:
        return self.raw.get("schedule", {})

    @property
    def index(self) -> Dict[str, Any]:
        return self.raw.get("index", {})

    @property
    def extract_relationships(self) -> Dict[str, Any]:
        return self.raw.get("extract_relationships", {})

    @property
    def pargv_batch(self) -> Dict[str, Any]:
        return self.raw.get("pargv_batch", {})

    @property
    def chat(self) -> Dict[str, Any]:
        return self.raw.get("chat", {})


def load_sources_config(path: Optional[Path] = None) -> SourcesConfig:
    """Load sources configuration from YAML."""
    path = path or MANIFESTS_DIR / "sources.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    settings = data.get("settings", {})
    categories = {}

    for cat_name, cat_data in data.get("sources", {}).items():
        entries = []
        for entry_data in cat_data.get("entries", []):
            # Entry can override category fetch_mode
            fetch_mode = entry_data.get("fetch_mode", cat_data.get("fetch_mode", "http"))
            entries.append(SourceEntry(
                name=entry_data["name"],
                url=entry_data["url"],
                format=entry_data.get("format", "html"),
                description=entry_data.get("description", ""),
                fetch_mode=fetch_mode,
                enabled=entry_data.get("enabled", True)
            ))

        categories[cat_name] = SourceCategory(
            name=cat_name,
            enabled=cat_data.get("enabled", True),
            source_kind=cat_data.get("source_kind", "unknown"),
            fetch_mode=cat_data.get("fetch_mode", "http"),
            rate_limit=cat_data.get("rate_limit", 2.0),
            entries=entries
        )

    return SourcesConfig(settings=settings, categories=categories)


def load_pipeline_config(path: Optional[Path] = None) -> PipelineConfig:
    """Load pipeline configuration from YAML."""
    path = path or MANIFESTS_DIR / "pipeline.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineConfig(raw=data)


def get_current_week() -> str:
    """Get current week in YYYY-WW format."""
    now = datetime.utcnow()
    return now.strftime("%Y-W%W")


def get_week_dirs(week: str) -> Dict[str, Path]:
    """Get data directory paths for a given week."""
    return {
        "raw": DATA_DIR / "raw" / week,
        "clean": DATA_DIR / "clean" / week,
        "meta": DATA_DIR / "meta" / week,
        "runs": RUNS_DIR / week,
    }


def ensure_week_dirs(week: str) -> Dict[str, Path]:
    """Ensure all directories for a week exist and return paths."""
    dirs = get_week_dirs(week)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
