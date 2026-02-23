"""
Shared Test Fixtures
====================
Provides temp database, sample artifacts, chunks, and mock utilities
for all test modules.
"""

import pytest
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Database isolation — every test gets its own empty, schema-ready SQLite
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Redirect all DB operations to a temp SQLite file with schema initialized."""
    db_file = tmp_path / "test_observer.sqlite"
    monkeypatch.setattr("pipeline.database.DB_PATH", db_file)
    from pipeline.database import init_database
    init_database()
    yield db_file


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_artifact():
    """Minimal valid artifact dict."""
    return {
        "artifact_id": "art_abc123",
        "canonical_url": "https://example.com/article1",
        "content_hash": "h_abc123",
        "source_name": "Test Source",
        "source_kind": "news",
        "domain": "example.com",
        "title": "Test Article",
        "url": "https://example.com/article1",
        "week": "2026-W07",
        "retrieved_at": datetime.utcnow().isoformat(),
        "published_at": "2026-02-10",
        "fetch_mode": "static",
        "http_status": 200,
        "etag": None,
        "last_modified": None,
        "raw_path": "raw/test/art_abc123.html",
        "clean_path": None,
        "meta_path": "meta/test/art_abc123.json",
        "main_text_length": None,
    }


@pytest.fixture
def sample_chunk(sample_artifact):
    """Minimal valid chunk dict linked to sample_artifact."""
    return {
        "chunk_id": f"{sample_artifact['artifact_id']}_0000",
        "artifact_id": sample_artifact["artifact_id"],
        "chunk_index": 0,
        "text": "Acme Corp raises $50M Series B funding round.",
        "start_char": 0,
        "end_char": 46,
        "token_count_approx": 12,
        "week": sample_artifact["week"],
        "source_kind": sample_artifact["source_kind"],
        "source_name": sample_artifact["source_name"],
        "canonical_url": sample_artifact["canonical_url"],
        "title": sample_artifact["title"],
        "published_at": sample_artifact["published_at"],
        "retrieved_at": sample_artifact["retrieved_at"],
    }


@pytest.fixture
def sample_claim():
    """Minimal valid claim dict."""
    return {
        "claim_id": "clm_test000001",
        "claim_type": "funding_round",
        "subject_entity_id": None,
        "object_entity_id": None,
        "predicate": "raised",
        "value": "$50M",
        "unit": "USD",
        "confidence": 0.85,
        "source_artifact_id": "art_abc123",
        "source_chunk_id": "art_abc123_0000",
        "evidence_text": "Acme Corp raises $50M Series B funding round.",
        "extracted_at": datetime.utcnow().isoformat(),
        "validated": False,
        "week": "2026-W07",
    }


def make_artifact(id_suffix, url_path="page", week="2026-W07", **overrides):
    """Factory for creating multiple distinct artifacts."""
    base = {
        "artifact_id": f"art_{id_suffix}",
        "canonical_url": f"https://example.com/{url_path}/{id_suffix}",
        "content_hash": f"h_{id_suffix}",
        "source_name": "Test Source",
        "source_kind": "news",
        "domain": "example.com",
        "title": f"Article {id_suffix}",
        "url": f"https://example.com/{url_path}/{id_suffix}",
        "week": week,
        "retrieved_at": datetime.utcnow().isoformat(),
    }
    base.update(overrides)
    return base


def make_chunk(artifact_id, index=0, text="Sample chunk text.", week="2026-W07"):
    """Factory for creating chunks."""
    return {
        "chunk_id": f"{artifact_id}_{index:04d}",
        "artifact_id": artifact_id,
        "chunk_index": index,
        "text": text,
        "start_char": 0,
        "end_char": len(text),
        "token_count_approx": len(text) // 4,
        "week": week,
    }
