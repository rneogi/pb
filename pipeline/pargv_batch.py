"""
PARGV Batch Pipeline Stage
==========================
Generates weekly digest from evaluation report and indexed content.
PARGV = Parse, Abstract, Retrieve, Generate, Validate

Phase 1 implementation uses rule-based planning and templated generation.
No external LLM required - produces grounded summaries from retrieved chunks.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .config import load_pipeline_config, ensure_week_dirs, PRODUCTS_DIR, DATA_DIR
from .database import get_artifacts_by_week, get_evaluations_by_week
from .index import get_retriever


def load_eval_report(week: str) -> Dict[str, Any]:
    """Load evaluation report for the week."""
    report_path = Path(__file__).parent.parent / "runs" / week / "eval_report.json"
    if not report_path.exists():
        return {"buckets": {}}
    return json.loads(report_path.read_text(encoding="utf-8"))


def extract_snippet(text: str, keywords: List[str], max_length: int = 300) -> str:
    """
    Extract a relevant snippet containing keywords.
    Returns the most relevant sentence/paragraph.
    """
    if not text:
        return ""

    text = text.strip()

    # If text is short enough, return it
    if len(text) <= max_length:
        return text

    # Find sentences containing keywords
    sentences = re.split(r'(?<=[.!?])\s+', text)
    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for kw in keywords if kw.lower() in sentence_lower)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    # If found a good sentence, return it (truncated if needed)
    if best_sentence:
        if len(best_sentence) > max_length:
            return best_sentence[:max_length-3] + "..."
        return best_sentence

    # Fallback: return beginning of text
    return text[:max_length-3] + "..."


def get_artifact_snippet(artifact_id: str, week: str, keywords: List[str], max_length: int = 300) -> str:
    """Get a relevant snippet from an artifact's cleaned text."""
    artifacts = get_artifacts_by_week(week)
    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            if artifact.get("clean_path"):
                clean_path = DATA_DIR / artifact["clean_path"]
                if clean_path.exists():
                    text = clean_path.read_text(encoding="utf-8", errors="replace")
                    return extract_snippet(text, keywords, max_length)
    return ""


def format_deal_signals(
    artifacts: List[Dict[str, Any]],
    week: str,
    max_items: int = 20,
    snippet_length: int = 300
) -> str:
    """Format deal signal artifacts into markdown section."""
    if not artifacts:
        return "_No deal signals detected this week._\n"

    lines = []
    deal_keywords = ["raises", "raised", "funding", "series", "seed", "investment",
                     "acquires", "acquisition", "merger", "valuation", "million", "billion"]

    for item in artifacts[:max_items]:
        title = item.get("title", "Untitled")
        url = item.get("canonical_url", "")
        source = item.get("source_name", "Unknown")
        keywords_matched = item.get("keywords_matched", [])

        # Get snippet
        snippet = get_artifact_snippet(
            item["artifact_id"], week,
            keywords_matched or deal_keywords,
            snippet_length
        )

        lines.append(f"### {title}")
        lines.append(f"- **Source**: {source}")
        lines.append(f"- **URL**: [{url}]({url})")
        if keywords_matched:
            lines.append(f"- **Signals**: {', '.join(keywords_matched[:5])}")
        if snippet:
            lines.append(f"- **Excerpt**: _{snippet}_")
        lines.append("")

    return "\n".join(lines)


def format_investor_changes(
    artifacts: List[Dict[str, Any]],
    week: str,
    max_items: int = 10
) -> str:
    """Format investor graph changes into markdown section."""
    if not artifacts:
        return "_No investor portfolio changes detected this week._\n"

    lines = []
    for item in artifacts[:max_items]:
        title = item.get("title", "Untitled")
        url = item.get("canonical_url", "")
        source = item.get("source_name", "Unknown")

        lines.append(f"- **{source}**: [{title}]({url})")

    return "\n".join(lines)


def format_company_changes(
    artifacts: List[Dict[str, Any]],
    week: str,
    max_items: int = 10
) -> str:
    """Format company profile changes into markdown section."""
    if not artifacts:
        return "_No company profile changes detected this week._\n"

    lines = []
    for item in artifacts[:max_items]:
        title = item.get("title", "Untitled")
        url = item.get("canonical_url", "")
        source = item.get("source_name", "Unknown")

        lines.append(f"- **{source}**: [{title}]({url})")

    return "\n".join(lines)


def validate_citations(content: str) -> List[str]:
    """
    Validate that all bullet points have URL citations.
    Returns list of validation issues.
    """
    issues = []

    # Find all bullet points
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("- "):
            # Check if line or nearby lines contain a URL
            context = "\n".join(lines[max(0, i-2):i+3])
            if "http" not in context and "](http" not in context:
                issues.append(f"Line {i+1}: Bullet point may lack URL citation")

    return issues


def run_pargv_batch(week: str, pipeline_config=None) -> Dict[str, Any]:
    """
    Run PARGV batch digest generation for a given week.

    PARGV Steps:
    1. Parse: Load eval report, classify content
    2. Abstract: Plan digest structure
    3. Retrieve: Pull relevant chunks/artifacts
    4. Generate: Create templated markdown digest
    5. Validate: Ensure all claims have citations
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    pargv_cfg = pipeline_config.pargv_batch

    max_deal_signals = pargv_cfg.get("max_deal_signals", 20)
    max_investor_changes = pargv_cfg.get("max_investor_changes", 10)
    max_company_changes = pargv_cfg.get("max_company_changes", 10)
    snippet_length = pargv_cfg.get("snippet_length", 300)

    print(f"Starting PARGV batch digest for week {week}")

    # =========================================================================
    # 1. PARSE: Load and classify inputs
    # =========================================================================
    print("  Step 1: Parse - Loading evaluation report...")
    eval_report = load_eval_report(week)
    buckets = eval_report.get("buckets", {})

    deal_signals = buckets.get("deal_signal", {}).get("artifacts", [])
    investor_changes = buckets.get("investor_graph_change", {}).get("artifacts", [])
    company_changes = buckets.get("company_profile_change", {}).get("artifacts", [])

    print(f"    Found: {len(deal_signals)} deal signals, "
          f"{len(investor_changes)} investor changes, "
          f"{len(company_changes)} company changes")

    # =========================================================================
    # 2. ABSTRACT: Plan digest structure
    # =========================================================================
    print("  Step 2: Abstract - Planning digest structure...")
    has_content = bool(deal_signals or investor_changes or company_changes)

    # =========================================================================
    # 3. RETRIEVE: Content already loaded from eval report
    # =========================================================================
    print("  Step 3: Retrieve - Gathering content snippets...")

    # =========================================================================
    # 4. GENERATE: Create markdown digest
    # =========================================================================
    print("  Step 4: Generate - Creating digest...")

    # Format week for display
    year, week_num = week.split("-W")
    week_display = f"Week {int(week_num)}, {year}"

    digest_lines = [
        f"# Public PitchBook Observer - Weekly Digest",
        f"## {week_display}",
        "",
        f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "---",
        "",
        "> **Disclaimer**: This digest is compiled from publicly accessible sources only. "
        "It may be incomplete and should not be used as the sole basis for investment decisions. "
        "All information should be independently verified.",
        "",
        "---",
        "",
    ]

    if not has_content:
        digest_lines.extend([
            "## Summary",
            "",
            "_No significant updates detected this week._",
            "",
        ])
    else:
        # Deal Signals Section
        digest_lines.extend([
            "## Deal Signals",
            "",
            "Funding announcements, acquisitions, and investment activity detected from public sources.",
            "",
            format_deal_signals(deal_signals, week, max_deal_signals, snippet_length),
            "",
        ])

        # Investor Portfolio Changes Section
        digest_lines.extend([
            "## Investor Portfolio Updates",
            "",
            "Changes detected on investor portfolio pages.",
            "",
            format_investor_changes(investor_changes, week, max_investor_changes),
            "",
        ])

        # Company Profile Changes Section
        digest_lines.extend([
            "## Company Updates",
            "",
            "Changes detected on company press/news pages.",
            "",
            format_company_changes(company_changes, week, max_company_changes),
            "",
        ])

    # Footer
    digest_lines.extend([
        "---",
        "",
        "## Methodology",
        "",
        "This digest was generated using the Public PitchBook Observer pipeline:",
        "",
        "1. **Crawl**: Public sources (SEC EDGAR, PR wires, news RSS, company pages) are fetched weekly",
        "2. **Clean**: Content is extracted and normalized",
        "3. **Delta**: New and changed content is identified",
        "4. **Evaluate**: Content is classified by signal type using keyword matching",
        "5. **Generate**: This digest summarizes findings with source citations",
        "",
        "All URLs link directly to the original public sources.",
        "",
        "---",
        "",
        f"_Public PitchBook Observer v1.0 | Week {week}_",
    ])

    digest_content = "\n".join(digest_lines)

    # =========================================================================
    # 5. VALIDATE: Ensure citations exist
    # =========================================================================
    print("  Step 5: Validate - Checking citations...")
    validation_issues = validate_citations(digest_content)
    if validation_issues:
        print(f"    WARNING: {len(validation_issues)} potential citation issues")
        for issue in validation_issues[:5]:
            print(f"      - {issue}")

    # Save digest
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    digest_filename = f"weekly_digest_{week.replace('-', '_')}.md"
    digest_path = PRODUCTS_DIR / digest_filename
    digest_path.write_text(digest_content, encoding="utf-8")

    print(f"\nDigest saved to: {digest_path}")

    # Build summary
    summary = {
        "week": week,
        "digest_path": str(digest_path),
        "content_summary": {
            "deal_signals": len(deal_signals),
            "investor_changes": len(investor_changes),
            "company_changes": len(company_changes)
        },
        "validation_issues": validation_issues,
        "generated_at": datetime.utcnow().isoformat()
    }

    # Save summary
    summary_path = dirs["runs"] / "pargv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main(week: str):
    """Entry point for pargv_batch stage."""
    return run_pargv_batch(week)


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(week)
