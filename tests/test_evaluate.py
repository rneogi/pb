"""
Tests for Evaluate Pipeline Stage
=================================
Tests the deterministic bucket classification logic.
"""

import pytest
from pipeline.evaluate import match_keywords, evaluate_artifact


class TestMatchKeywords:
    """Tests for keyword matching."""

    def test_no_matches(self):
        """Test text with no keyword matches."""
        text = "This is a regular news article about technology."
        keywords = ["funding", "acquisition", "series"]

        matches = match_keywords(text, keywords)
        assert matches == []

    def test_single_match(self):
        """Test text with single keyword match."""
        text = "The company announced a new funding round."
        keywords = ["funding", "acquisition", "series"]

        matches = match_keywords(text, keywords)
        assert matches == ["funding"]

    def test_multiple_matches(self):
        """Test text with multiple keyword matches."""
        text = "The company raised Series A funding of $10M."
        keywords = ["funding", "raised", "series", "acquisition"]

        matches = match_keywords(text, keywords)
        assert "funding" in matches
        assert "raised" in matches
        # "series" might match as part of "Series A"

    def test_case_insensitive(self):
        """Test that matching is case insensitive."""
        text = "FUNDING ANNOUNCED FOR SERIES B"
        keywords = ["funding", "series"]

        matches = match_keywords(text, keywords)
        assert "funding" in matches

    def test_word_boundary(self):
        """Test that matching respects word boundaries."""
        text = "The refunding process was completed."
        keywords = ["funding"]

        matches = match_keywords(text, keywords)
        # Should not match "funding" in "refunding"
        assert "funding" not in matches


class TestEvaluateArtifact:
    """Tests for artifact evaluation."""

    def test_deal_signal_detection(self):
        """Test detection of deal signals."""
        artifact = {
            "artifact_id": "test1",
            "source_kind": "pr_wire",
            "source_name": "PR Newswire"
        }
        text = "Acme Corp raises $50M Series B funding round led by Top Ventures."

        bucket_config = {
            "deal_signal": {
                "keywords": ["raises", "raised", "funding", "Series", "seed"],
                "source_kinds": ["filing", "pr_wire", "news"]
            }
        }

        results = evaluate_artifact(artifact, text, bucket_config)

        assert len(results) >= 1
        buckets = [r["bucket"] for r in results]
        assert "deal_signal" in buckets

        deal_result = next(r for r in results if r["bucket"] == "deal_signal")
        assert deal_result["confidence"] > 0.5
        assert len(deal_result["keywords_matched"]) > 0

    def test_investor_portfolio_detection(self):
        """Test detection of investor portfolio changes."""
        artifact = {
            "artifact_id": "test2",
            "source_kind": "investor_portfolio",
            "source_name": "VC Portfolio"
        }
        text = "Our portfolio companies include various startups."

        bucket_config = {
            "investor_graph_change": {
                "keywords": ["portfolio", "invested"],
                "source_kinds": ["investor_portfolio"]
            }
        }

        results = evaluate_artifact(artifact, text, bucket_config)

        buckets = [r["bucket"] for r in results]
        assert "investor_graph_change" in buckets

    def test_noise_classification(self):
        """Test that non-matching content is classified as noise."""
        artifact = {
            "artifact_id": "test3",
            "source_kind": "news",
            "source_name": "News Site"
        }
        text = "The weather today is sunny with a high of 75 degrees."

        bucket_config = {
            "deal_signal": {
                "keywords": ["funding", "raises"],
                "source_kinds": ["pr_wire"]
            }
        }

        results = evaluate_artifact(artifact, text, bucket_config)

        assert len(results) == 1
        assert results[0]["bucket"] == "noise"

    def test_source_kind_filtering(self):
        """Test that source_kind filters work correctly."""
        artifact = {
            "artifact_id": "test4",
            "source_kind": "telemetry",  # Not in deal_signal source_kinds
            "source_name": "Careers Page"
        }
        text = "We are raising our hiring goals for the funding team."

        bucket_config = {
            "deal_signal": {
                "keywords": ["funding", "raising"],
                "source_kinds": ["filing", "pr_wire", "news"]  # Excludes telemetry
            },
            "telemetry_change": {
                "keywords": ["hiring", "careers"],
                "source_kinds": ["telemetry"]
            }
        }

        results = evaluate_artifact(artifact, text, bucket_config)

        buckets = [r["bucket"] for r in results]
        # Should match telemetry_change but not deal_signal
        assert "telemetry_change" in buckets

    def test_multiple_bucket_assignment(self):
        """Test that artifact can match multiple buckets."""
        artifact = {
            "artifact_id": "test5",
            "source_kind": "news",
            "source_name": "Tech News"
        }
        text = "The startup raised funding and the CEO announced leadership changes."

        bucket_config = {
            "deal_signal": {
                "keywords": ["raised", "funding"],
                "source_kinds": ["news"]
            },
            "company_profile_change": {
                "keywords": ["CEO", "leadership"],
                "source_kinds": ["news", "company_press"]
            }
        }

        results = evaluate_artifact(artifact, text, bucket_config)

        buckets = [r["bucket"] for r in results]
        assert "deal_signal" in buckets
        assert "company_profile_change" in buckets

    def test_confidence_scaling(self):
        """Test that confidence scales with keyword matches."""
        artifact = {
            "artifact_id": "test6",
            "source_kind": "pr_wire",
            "source_name": "PR Wire"
        }

        # Few keywords
        text_few = "Company raised funding."
        # Many keywords
        text_many = "Company raised Series A funding investment capital valuation."

        bucket_config = {
            "deal_signal": {
                "keywords": ["raised", "funding", "Series", "investment", "capital", "valuation"],
                "source_kinds": ["pr_wire"]
            }
        }

        results_few = evaluate_artifact(artifact, text_few, bucket_config)
        results_many = evaluate_artifact(artifact, text_many, bucket_config)

        conf_few = next(r["confidence"] for r in results_few if r["bucket"] == "deal_signal")
        conf_many = next(r["confidence"] for r in results_many if r["bucket"] == "deal_signal")

        # More keyword matches should result in higher confidence
        assert conf_many >= conf_few


# =========================================================================
# Edge / Error Cases
# =========================================================================

class TestEvaluateEdgeCases:
    """Boundary and error tests for evaluation logic."""

    def test_empty_text(self):
        """Empty text should classify as noise."""
        artifact = {"artifact_id": "t1", "source_kind": "news"}
        bucket_config = {
            "deal_signal": {
                "keywords": ["funding"],
                "source_kinds": ["news"],
            }
        }
        results = evaluate_artifact(artifact, "", bucket_config)
        assert len(results) == 1
        assert results[0]["bucket"] == "noise"

    def test_empty_bucket_config(self):
        """Empty bucket config should classify everything as noise."""
        artifact = {"artifact_id": "t2", "source_kind": "news"}
        results = evaluate_artifact(artifact, "Company raised funding.", {})
        assert len(results) == 1
        assert results[0]["bucket"] == "noise"

    def test_confidence_never_exceeds_one(self):
        """Even with many keyword matches, confidence is capped at 1.0."""
        artifact = {"artifact_id": "t3", "source_kind": "news"}
        # 20 keywords all present — confidence formula: 0.5 + 0.1*20 = 2.5 → capped at 1.0
        keywords = [f"kw{i}" for i in range(20)]
        text = " ".join(keywords)
        bucket_config = {
            "test_bucket": {"keywords": keywords, "source_kinds": ["news"]}
        }
        results = evaluate_artifact(artifact, text, bucket_config)
        matched = next(r for r in results if r["bucket"] == "test_bucket")
        assert matched["confidence"] <= 1.0

    def test_confidence_minimum_with_one_match(self):
        """Single keyword match gives confidence = 0.6 (0.5 + 0.1*1)."""
        artifact = {"artifact_id": "t4", "source_kind": "news"}
        results = evaluate_artifact(
            artifact, "This is about funding.",
            {"deal_signal": {"keywords": ["funding"], "source_kinds": ["news"]}}
        )
        deal = next(r for r in results if r["bucket"] == "deal_signal")
        assert deal["confidence"] == pytest.approx(0.6, abs=0.01)

    def test_investor_portfolio_special_case(self):
        """investor_portfolio source_kind always matches investor_graph_change."""
        artifact = {"artifact_id": "t5", "source_kind": "investor_portfolio"}
        # No keywords match, but special case should fire
        results = evaluate_artifact(
            artifact, "Completely unrelated text about cooking.",
            {"investor_graph_change": {"keywords": ["invested"], "source_kinds": ["filing"]}}
        )
        buckets = [r["bucket"] for r in results]
        assert "investor_graph_change" in buckets

    def test_telemetry_special_case(self):
        """telemetry source_kind always matches telemetry_change."""
        artifact = {"artifact_id": "t6", "source_kind": "telemetry"}
        results = evaluate_artifact(
            artifact, "Random text.",
            {"telemetry_change": {"keywords": ["hiring"], "source_kinds": ["news"]}}
        )
        buckets = [r["bucket"] for r in results]
        assert "telemetry_change" in buckets

    def test_source_kind_mismatch_no_match(self):
        """Keywords match but source_kind doesn't → no bucket assigned."""
        artifact = {"artifact_id": "t7", "source_kind": "blog"}
        results = evaluate_artifact(
            artifact, "Company raised funding.",
            {"deal_signal": {"keywords": ["funding"], "source_kinds": ["filing"]}}
        )
        assert all(r["bucket"] == "noise" for r in results)

    def test_missing_source_kind_field(self):
        """Artifact without source_kind defaults to 'unknown'."""
        artifact = {"artifact_id": "t8"}
        results = evaluate_artifact(
            artifact, "Company raised funding.",
            {"deal_signal": {"keywords": ["funding"], "source_kinds": []}}
        )
        # source_kinds=[] means any source matches
        buckets = [r["bucket"] for r in results]
        assert "deal_signal" in buckets


class TestMatchKeywordsEdgeCases:
    """Boundary tests for keyword matching."""

    def test_special_regex_characters(self):
        """Keywords with regex special chars are escaped properly (no crash).

        Note: \b word-boundary before non-word chars like $ or ( means
        the keyword won't actually match, but the function must not raise.
        """
        text = "Company valued at $50M (2x revenue)."
        matches = match_keywords(text, ["$50M", "(2x"])
        # These won't match due to \b + non-word-char, but must not crash
        assert isinstance(matches, list)

    def test_keyword_at_start_and_end(self):
        """Keywords at very start or end of text match."""
        text = "funding at the end of this sentence is funding"
        matches = match_keywords(text, ["funding"])
        assert "funding" in matches

    def test_empty_keywords_list(self):
        """Empty keyword list returns empty matches."""
        assert match_keywords("anything", []) == []

    def test_empty_text(self):
        """Empty text returns no matches."""
        assert match_keywords("", ["funding"]) == []
