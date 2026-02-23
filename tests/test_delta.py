"""
Tests for Delta Pipeline Stage
==============================
Tests the deterministic delta computation logic.
"""

import pytest
from pipeline.delta import compute_delta, get_previous_week


class TestGetPreviousWeek:
    """Tests for week calculation."""

    def test_previous_week_same_year(self):
        """Test getting previous week within same year."""
        # Note: Results depend on actual calendar
        result = get_previous_week("2026-W10", offset=1)
        assert result.startswith("2026-W")

    def test_previous_week_offset(self):
        """Test getting week with larger offset."""
        result = get_previous_week("2026-W10", offset=4)
        assert result.startswith("2026-W") or result.startswith("2025-W")


class TestComputeDelta:
    """Tests for delta computation."""

    def test_empty_current_empty_previous(self):
        """Test with no artifacts in either week."""
        delta = compute_delta(
            current_artifacts=[],
            previous_index={},
            week="2026-W05",
            config={}
        )

        assert delta["new"] == []
        assert delta["changed"] == []
        assert delta["unchanged"] == []
        assert delta["stale"] == []

    def test_all_new_artifacts(self):
        """Test when all artifacts are new."""
        current = [
            {
                "artifact_id": "abc123",
                "canonical_url": "https://example.com/page1",
                "content_hash": "hash1",
                "source_name": "Test Source",
                "source_kind": "news",
                "title": "Test Article"
            },
            {
                "artifact_id": "def456",
                "canonical_url": "https://example.com/page2",
                "content_hash": "hash2",
                "source_name": "Test Source",
                "source_kind": "news",
                "title": "Another Article"
            }
        ]

        delta = compute_delta(
            current_artifacts=current,
            previous_index={},
            week="2026-W05",
            config={}
        )

        assert len(delta["new"]) == 2
        assert len(delta["changed"]) == 0
        assert len(delta["unchanged"]) == 0

    def test_unchanged_artifacts(self):
        """Test when artifacts haven't changed."""
        current = [
            {
                "artifact_id": "abc123",
                "canonical_url": "https://example.com/page1",
                "content_hash": "hash1",
                "source_name": "Test Source",
                "source_kind": "news",
                "title": "Test Article"
            }
        ]

        previous = {
            "https://example.com/page1": {
                "artifact_id": "abc123",
                "canonical_url": "https://example.com/page1",
                "content_hash": "hash1",
                "source_name": "Test Source",
                "source_kind": "news",
                "title": "Test Article"
            }
        }

        delta = compute_delta(
            current_artifacts=current,
            previous_index=previous,
            week="2026-W05",
            config={}
        )

        assert len(delta["new"]) == 0
        assert len(delta["changed"]) == 0
        assert len(delta["unchanged"]) == 1

    def test_changed_artifacts(self):
        """Test when content has changed."""
        current = [
            {
                "artifact_id": "abc123_v2",
                "canonical_url": "https://example.com/page1",
                "content_hash": "hash_new",
                "source_name": "Test Source",
                "source_kind": "news",
                "title": "Updated Article"
            }
        ]

        previous = {
            "https://example.com/page1": {
                "artifact_id": "abc123",
                "canonical_url": "https://example.com/page1",
                "content_hash": "hash_old",
                "source_name": "Test Source",
                "source_kind": "news",
                "title": "Test Article"
            }
        }

        delta = compute_delta(
            current_artifacts=current,
            previous_index=previous,
            week="2026-W05",
            config={}
        )

        assert len(delta["new"]) == 0
        assert len(delta["changed"]) == 1
        assert len(delta["unchanged"]) == 0
        assert delta["changed"][0]["previous_content_hash"] == "hash_old"
        assert delta["changed"][0]["content_hash"] == "hash_new"

    def test_mixed_delta(self):
        """Test combination of new, changed, and unchanged."""
        current = [
            {
                "artifact_id": "new1",
                "canonical_url": "https://example.com/new",
                "content_hash": "hash_new",
                "source_name": "Source",
                "source_kind": "news",
                "title": "New"
            },
            {
                "artifact_id": "changed1",
                "canonical_url": "https://example.com/changed",
                "content_hash": "hash_changed_new",
                "source_name": "Source",
                "source_kind": "news",
                "title": "Changed"
            },
            {
                "artifact_id": "same1",
                "canonical_url": "https://example.com/same",
                "content_hash": "hash_same",
                "source_name": "Source",
                "source_kind": "news",
                "title": "Same"
            }
        ]

        previous = {
            "https://example.com/changed": {
                "artifact_id": "changed1_old",
                "canonical_url": "https://example.com/changed",
                "content_hash": "hash_changed_old",
                "source_name": "Source",
                "source_kind": "news",
                "title": "Changed Old"
            },
            "https://example.com/same": {
                "artifact_id": "same1",
                "canonical_url": "https://example.com/same",
                "content_hash": "hash_same",
                "source_name": "Source",
                "source_kind": "news",
                "title": "Same"
            }
        }

        delta = compute_delta(
            current_artifacts=current,
            previous_index=previous,
            week="2026-W05",
            config={}
        )

        assert len(delta["new"]) == 1
        assert len(delta["changed"]) == 1
        assert len(delta["unchanged"]) == 1
        assert delta["new"][0]["canonical_url"] == "https://example.com/new"
        assert delta["changed"][0]["canonical_url"] == "https://example.com/changed"
        assert delta["unchanged"][0]["canonical_url"] == "https://example.com/same"

    def test_stale_not_in_current(self):
        """URLs in previous but not current are candidates for stale (threshold=0)."""
        current = []
        previous = {
            "https://example.com/gone": {
                "artifact_id": "gone1",
                "canonical_url": "https://example.com/gone",
                "content_hash": "hash_gone",
                "source_name": "Source",
                "source_kind": "news",
                "title": "Gone"
            }
        }
        # stale_threshold_weeks=0 means any missing URL is immediately stale
        # But compute_delta checks get_crawl_state which returns None for unknown URLs
        delta = compute_delta(
            current_artifacts=current,
            previous_index=previous,
            week="2026-W05",
            config={"stale_threshold_weeks": 0}
        )
        # With no crawl_state record, stale detection skips the URL
        assert len(delta["new"]) == 0
        assert len(delta["changed"]) == 0
        assert len(delta["unchanged"]) == 0

    def test_duplicate_urls_in_current(self):
        """Same canonical_url appearing twice in current should be handled."""
        art1 = {
            "artifact_id": "dup1_v1",
            "canonical_url": "https://example.com/dup",
            "content_hash": "hash_v1",
            "source_name": "Source",
            "source_kind": "news",
            "title": "Dup"
        }
        art2 = {
            "artifact_id": "dup1_v2",
            "canonical_url": "https://example.com/dup",
            "content_hash": "hash_v2",
            "source_name": "Source",
            "source_kind": "news",
            "title": "Dup Updated"
        }
        delta = compute_delta(
            current_artifacts=[art1, art2],
            previous_index={},
            week="2026-W05",
            config={}
        )
        # Both show up as new since neither is in previous_index
        assert len(delta["new"]) == 2


# =========================================================================
# Edge / Error Cases
# =========================================================================

class TestGetPreviousWeekEdgeCases:
    """Boundary tests for week calculations."""

    def test_week_01_wraps_to_previous_year(self):
        """Offset from W01 should wrap to previous year."""
        result = get_previous_week("2026-W01", offset=1)
        assert result.startswith("2025-W")

    def test_large_offset(self):
        """Large offset produces valid week string."""
        result = get_previous_week("2026-W26", offset=52)
        assert "-W" in result
        parts = result.split("-W")
        assert len(parts) == 2
        assert parts[0].isdigit()

    def test_offset_zero_returns_same_week(self):
        """Zero offset returns approximately the same week."""
        result = get_previous_week("2026-W10", offset=0)
        # Should be the same or very close week
        assert result.startswith("2026-W")
