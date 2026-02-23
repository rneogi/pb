"""
Tests for Database Schema and Operations
=========================================
Covers schema initialization, CRUD operations, constraint enforcement,
and error paths for all tables in pipeline/database.py.
"""

import pytest
from datetime import datetime
from pipeline.database import (
    init_database, db_session,
    upsert_artifact, get_artifact, get_artifacts_by_week,
    insert_chunks, get_chunks_by_artifact, get_chunks_by_week, mark_chunks_embedded,
    get_crawl_state, update_crawl_state,
    insert_evaluation, get_evaluations_by_week,
    insert_delta, get_deltas_by_week,
    insert_claims, get_claims_by_week, get_claims_by_type,
)
from tests.conftest import make_artifact, make_chunk


# =========================================================================
# Schema Initialization
# =========================================================================

class TestInitDatabase:
    """Tests for init_database()."""

    def test_creates_all_tables(self):
        """All expected tables should exist after init."""
        expected = {
            "artifacts", "chunks", "crawl_state", "delta_history",
            "evaluations", "entities", "claims", "deal_records",
            "entity_mentions",
        }
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cursor.fetchall()}
        assert expected.issubset(tables)

    def test_idempotent(self):
        """Calling init_database() twice should not raise."""
        init_database()  # already called by conftest, call again
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) as cnt FROM sqlite_master WHERE type='table'")
            assert cursor.fetchone()["cnt"] >= 9

    def test_indexes_created(self):
        """Core indexes should exist."""
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_artifacts_week" in indexes
        assert "idx_chunks_week" in indexes
        assert "idx_claims_type" in indexes


# =========================================================================
# Artifact Operations
# =========================================================================

class TestArtifactCRUD:
    """Tests for artifact CRUD operations."""

    def test_upsert_and_get(self, sample_artifact):
        """Insert then retrieve an artifact."""
        upsert_artifact(sample_artifact)
        result = get_artifact(sample_artifact["artifact_id"])
        assert result is not None
        assert result["artifact_id"] == sample_artifact["artifact_id"]
        assert result["canonical_url"] == sample_artifact["canonical_url"]
        assert result["source_kind"] == "news"

    def test_get_nonexistent_returns_none(self):
        """Getting a missing artifact returns None."""
        assert get_artifact("nonexistent_id") is None

    def test_upsert_updates_existing(self, sample_artifact):
        """Upserting with same artifact_id should update."""
        upsert_artifact(sample_artifact)
        sample_artifact["title"] = "Updated Title"
        upsert_artifact(sample_artifact)
        result = get_artifact(sample_artifact["artifact_id"])
        assert result["title"] == "Updated Title"

    def test_get_artifacts_by_week(self, sample_artifact):
        """Filter artifacts by week."""
        upsert_artifact(sample_artifact)
        other = make_artifact("other", week="2026-W08")
        other["url"] = other["canonical_url"]
        other["retrieved_at"] = datetime.utcnow().isoformat()
        upsert_artifact(other)

        w07 = get_artifacts_by_week("2026-W07")
        w08 = get_artifacts_by_week("2026-W08")
        assert len(w07) == 1
        assert len(w08) == 1
        assert w07[0]["artifact_id"] == sample_artifact["artifact_id"]

    def test_get_artifacts_empty_week(self):
        """Querying a week with no data returns empty list."""
        assert get_artifacts_by_week("1999-W01") == []

    def test_unique_canonical_url_week_constraint(self, sample_artifact):
        """Two artifacts with same (canonical_url, week) should upsert, not duplicate."""
        upsert_artifact(sample_artifact)
        dup = dict(sample_artifact)
        dup["artifact_id"] = "art_different_id"
        # This uses INSERT OR REPLACE keyed on artifact_id (UNIQUE),
        # but canonical_url+week also has a UNIQUE constraint
        upsert_artifact(dup)
        results = get_artifacts_by_week(sample_artifact["week"])
        # Should be 1 row because of UNIQUE(canonical_url, week)
        assert len(results) == 1


# =========================================================================
# Chunk Operations
# =========================================================================

class TestChunkCRUD:
    """Tests for chunk CRUD operations."""

    def test_insert_and_get_by_artifact(self, sample_artifact, sample_chunk):
        """Insert chunks and retrieve by artifact_id."""
        upsert_artifact(sample_artifact)
        insert_chunks([sample_chunk])
        chunks = get_chunks_by_artifact(sample_artifact["artifact_id"])
        assert len(chunks) == 1
        assert chunks[0]["text"] == sample_chunk["text"]

    def test_insert_multiple_chunks(self, sample_artifact):
        """Insert several chunks for one artifact."""
        upsert_artifact(sample_artifact)
        chunks = [
            make_chunk(sample_artifact["artifact_id"], i, f"Chunk text {i}.")
            for i in range(5)
        ]
        insert_chunks(chunks)
        result = get_chunks_by_artifact(sample_artifact["artifact_id"])
        assert len(result) == 5
        # Should be ordered by chunk_index
        for i, c in enumerate(result):
            assert c["chunk_index"] == i

    def test_get_chunks_by_week(self, sample_artifact, sample_chunk):
        """Filter chunks by week."""
        upsert_artifact(sample_artifact)
        insert_chunks([sample_chunk])
        result = get_chunks_by_week("2026-W07")
        assert len(result) == 1
        assert get_chunks_by_week("2026-W99") == []

    def test_mark_chunks_embedded(self, sample_artifact, sample_chunk):
        """Mark chunks as embedded and filter."""
        upsert_artifact(sample_artifact)
        insert_chunks([sample_chunk])

        # Initially not embedded
        all_chunks = get_chunks_by_week("2026-W07", embedded_only=True)
        assert len(all_chunks) == 0

        # Mark embedded
        mark_chunks_embedded([sample_chunk["chunk_id"]])
        embedded = get_chunks_by_week("2026-W07", embedded_only=True)
        assert len(embedded) == 1

    def test_insert_empty_list(self):
        """Inserting empty chunk list should be a no-op."""
        insert_chunks([])  # Should not raise


# =========================================================================
# Crawl State Operations
# =========================================================================

class TestCrawlState:
    """Tests for crawl state operations."""

    def test_get_nonexistent_returns_none(self):
        """Missing URL returns None."""
        assert get_crawl_state("https://never-crawled.example.com") is None

    def test_update_creates_new(self):
        """First update for a URL creates the record."""
        url = "https://example.com/new-page"
        update_crawl_state(url, "hash1", "2026-W07")
        state = get_crawl_state(url)
        assert state is not None
        assert state["last_content_hash"] == "hash1"
        assert state["crawl_count"] == 1
        assert state["consecutive_unchanged"] == 0

    def test_update_increments_crawl_count(self):
        """Subsequent updates increment crawl_count."""
        url = "https://example.com/counter"
        update_crawl_state(url, "h1", "2026-W07")
        update_crawl_state(url, "h2", "2026-W08")
        state = get_crawl_state(url)
        assert state["crawl_count"] == 2
        assert state["last_content_hash"] == "h2"

    def test_consecutive_unchanged_tracking(self):
        """Consecutive unchanged flag increments correctly."""
        url = "https://example.com/stable"
        update_crawl_state(url, "h1", "2026-W07")
        update_crawl_state(url, "h1", "2026-W08", unchanged=True)
        update_crawl_state(url, "h1", "2026-W09", unchanged=True)
        state = get_crawl_state(url)
        assert state["consecutive_unchanged"] == 2

    def test_unchanged_resets_on_change(self):
        """consecutive_unchanged resets when content changes."""
        url = "https://example.com/flip"
        update_crawl_state(url, "h1", "2026-W07")
        update_crawl_state(url, "h1", "2026-W08", unchanged=True)
        update_crawl_state(url, "h2", "2026-W09", unchanged=False)
        state = get_crawl_state(url)
        assert state["consecutive_unchanged"] == 0


# =========================================================================
# Evaluation Operations
# =========================================================================

class TestEvaluationCRUD:
    """Tests for evaluation CRUD."""

    def test_insert_and_get_by_week(self):
        """Insert evaluation and retrieve by week."""
        insert_evaluation("art_1", "2026-W07", "deal_signal", 0.9, "matched keywords", ["funding"])
        results = get_evaluations_by_week("2026-W07")
        assert len(results) == 1
        assert results[0]["bucket"] == "deal_signal"
        assert results[0]["confidence"] == 0.9

    def test_multiple_evaluations_same_artifact(self):
        """One artifact can have multiple bucket assignments."""
        insert_evaluation("art_1", "2026-W07", "deal_signal", 0.9, "r1", ["funding"])
        insert_evaluation("art_1", "2026-W07", "company_profile_change", 0.7, "r2", ["CEO"])
        results = get_evaluations_by_week("2026-W07")
        assert len(results) == 2
        buckets = {r["bucket"] for r in results}
        assert buckets == {"deal_signal", "company_profile_change"}

    def test_empty_week(self):
        """Querying empty week returns empty list."""
        assert get_evaluations_by_week("1999-W01") == []


# =========================================================================
# Delta History Operations
# =========================================================================

class TestDeltaCRUD:
    """Tests for delta history CRUD."""

    def test_insert_and_get_by_week(self):
        """Insert delta records and query by week."""
        insert_delta("2026-W07", "https://example.com/a", "new", "art_1", new_hash="h1")
        insert_delta("2026-W07", "https://example.com/b", "changed", "art_2",
                      previous_hash="h_old", new_hash="h_new")
        results = get_deltas_by_week("2026-W07")
        assert len(results) == 2
        types = {r["delta_type"] for r in results}
        assert types == {"new", "changed"}

    def test_empty_week(self):
        """Querying week with no deltas returns empty list."""
        assert get_deltas_by_week("1999-W01") == []


# =========================================================================
# Claims Operations (Phase 2)
# =========================================================================

class TestClaimsCRUD:
    """Tests for Phase 2 claims CRUD."""

    def test_insert_and_get_by_week(self, sample_claim):
        """Insert claim and retrieve by week."""
        count = insert_claims([sample_claim])
        assert count == 1
        results = get_claims_by_week("2026-W07")
        assert len(results) == 1
        assert results[0]["claim_type"] == "funding_round"
        assert results[0]["value"] == "$50M"

    def test_insert_empty_list(self):
        """Inserting empty list returns 0."""
        assert insert_claims([]) == 0

    def test_duplicate_claim_id_replaces(self, sample_claim):
        """Inserting same claim_id again should replace (INSERT OR REPLACE)."""
        insert_claims([sample_claim])
        sample_claim["value"] = "$75M"
        insert_claims([sample_claim])
        results = get_claims_by_week("2026-W07")
        assert len(results) == 1
        assert results[0]["value"] == "$75M"

    def test_get_claims_by_type(self, sample_claim):
        """Filter claims by type."""
        acq_claim = dict(sample_claim)
        acq_claim["claim_id"] = "clm_acq_001"
        acq_claim["claim_type"] = "acquisition"
        insert_claims([sample_claim, acq_claim])

        funding = get_claims_by_type("funding_round")
        assert len(funding) == 1
        acquisitions = get_claims_by_type("acquisition")
        assert len(acquisitions) == 1

    def test_get_claims_by_type_with_week(self, sample_claim):
        """Filter claims by type and week."""
        insert_claims([sample_claim])
        result = get_claims_by_type("funding_round", week="2026-W07")
        assert len(result) == 1
        result_empty = get_claims_by_type("funding_round", week="2026-W99")
        assert len(result_empty) == 0

    def test_insert_claims_with_missing_optional_fields(self):
        """Claims with only required fields should still insert."""
        minimal = {
            "claim_id": "clm_minimal",
            "claim_type": "valuation",
            "week": "2026-W07",
        }
        assert insert_claims([minimal]) == 1
        results = get_claims_by_week("2026-W07")
        assert len(results) == 1
        assert results[0]["claim_type"] == "valuation"
        assert results[0]["value"] is None


# =========================================================================
# Transaction / Rollback
# =========================================================================

class TestTransactions:
    """Tests for db_session rollback behavior."""

    def test_rollback_on_error(self):
        """Failed insert inside db_session should not persist partial data."""
        try:
            with db_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO delta_history (week, canonical_url, delta_type) "
                    "VALUES (?, ?, ?)",
                    ("2026-W07", "https://example.com/rollback", "new"),
                )
                # Force an error before commit
                raise ValueError("simulated error")
        except ValueError:
            pass

        results = get_deltas_by_week("2026-W07")
        assert len(results) == 0
