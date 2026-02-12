"""
Batch Runner for Demo Questions
===============================
Runs all demo questions and generates detailed profiling report.
"""

import sys
import os
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.demo_questions import DEMO_QUESTIONS, get_questions_by_category, get_question_categories
from app.chat_interface import ChatInterface


class BatchRunner:
    """Runs batch of questions with profiling."""

    def __init__(self, output_dir: str = "batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chat = ChatInterface()
        self.results: List[Dict[str, Any]] = []

    def run_batch(
        self,
        questions: List[Dict[str, Any]] = None,
        top_k: int = 8,
        mode: str = "hybrid",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a batch of questions and collect results.

        Args:
            questions: List of question dicts (default: all demo questions)
            top_k: Number of results per query
            mode: Retrieval mode (hybrid/vector/keyword)
            verbose: Print progress

        Returns:
            Batch results summary
        """
        if questions is None:
            questions = DEMO_QUESTIONS

        # Initialize chat system
        if verbose:
            print("\n" + "="*70)
            print("  PUBLIC PITCHBOOK OBSERVER - BATCH RUNNER")
            print("="*70)

        self.chat.initialize()

        total = len(questions)
        start_time = time.perf_counter()

        if verbose:
            print(f"\nRunning {total} questions...")
            print("-"*70)

        self.results = []

        for i, q in enumerate(questions, 1):
            question = q["question"]
            expected_intent = q.get("expected_intent")
            category = q.get("category", "unknown")

            if verbose:
                print(f"[{i:3d}/{total}] {question[:50]}...", end=" ", flush=True)

            try:
                result = self.chat.process_query(
                    query=question,
                    top_k=top_k,
                    mode=mode,
                    verbose=False
                )

                # Add metadata
                result["question_id"] = q["id"]
                result["category"] = category
                result["expected_intent"] = expected_intent
                result["actual_intent"] = result["query_info"]["intent"]
                result["intent_match"] = (expected_intent == result["query_info"]["intent"])
                result["success"] = True

                self.results.append(result)

                if verbose:
                    conf = result["confidence_label"]
                    time_ms = result["timings"]["total_ms"]
                    print(f"[{conf:6s}] {time_ms:7.1f}ms")

            except Exception as e:
                error_result = {
                    "question_id": q["id"],
                    "query": question,
                    "category": category,
                    "expected_intent": expected_intent,
                    "success": False,
                    "error": str(e),
                    "timings": {"total_ms": 0}
                }
                self.results.append(error_result)

                if verbose:
                    print(f"[ERROR] {str(e)[:30]}")

        total_time = time.perf_counter() - start_time

        if verbose:
            print("-"*70)
            print(f"Completed in {total_time:.2f} seconds")

        return self.generate_report(total_time)

    def generate_report(self, total_batch_time: float) -> Dict[str, Any]:
        """Generate detailed profiling report from results."""

        # Basic stats
        total = len(self.results)
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]

        # Timing statistics
        all_times = [r["timings"]["total_ms"] for r in successful]
        retrieval_times = [r["timings"].get("retrieval_ms", 0) for r in successful]
        generation_times = [r["timings"].get("generation_ms", 0) for r in successful]

        timing_stats = {}
        if all_times:
            timing_stats = {
                "total": {
                    "min_ms": min(all_times),
                    "max_ms": max(all_times),
                    "mean_ms": statistics.mean(all_times),
                    "median_ms": statistics.median(all_times),
                    "stdev_ms": statistics.stdev(all_times) if len(all_times) > 1 else 0,
                    "p95_ms": sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0,
                    "p99_ms": sorted(all_times)[int(len(all_times) * 0.99)] if all_times else 0,
                },
                "retrieval": {
                    "min_ms": min(retrieval_times) if retrieval_times else 0,
                    "max_ms": max(retrieval_times) if retrieval_times else 0,
                    "mean_ms": statistics.mean(retrieval_times) if retrieval_times else 0,
                    "median_ms": statistics.median(retrieval_times) if retrieval_times else 0,
                },
                "generation": {
                    "min_ms": min(generation_times) if generation_times else 0,
                    "max_ms": max(generation_times) if generation_times else 0,
                    "mean_ms": statistics.mean(generation_times) if generation_times else 0,
                    "median_ms": statistics.median(generation_times) if generation_times else 0,
                }
            }

        # Confidence distribution
        confidence_dist = defaultdict(int)
        for r in successful:
            confidence_dist[r.get("confidence_label", "unknown")] += 1

        # Intent classification accuracy
        intent_results = [r for r in successful if r.get("expected_intent")]
        intent_correct = sum(1 for r in intent_results if r.get("intent_match"))
        intent_accuracy = intent_correct / len(intent_results) if intent_results else 0

        # Intent confusion matrix
        intent_matrix = defaultdict(lambda: defaultdict(int))
        for r in intent_results:
            expected = r.get("expected_intent", "unknown")
            actual = r.get("actual_intent", "unknown")
            intent_matrix[expected][actual] += 1

        # Category breakdown
        category_stats = {}
        for category in get_question_categories():
            cat_results = [r for r in successful if r.get("category") == category]
            if cat_results:
                cat_times = [r["timings"]["total_ms"] for r in cat_results]
                cat_confidence = defaultdict(int)
                for r in cat_results:
                    cat_confidence[r.get("confidence_label", "unknown")] += 1

                category_stats[category] = {
                    "count": len(cat_results),
                    "mean_time_ms": statistics.mean(cat_times),
                    "confidence_distribution": dict(cat_confidence)
                }

        # Citation statistics
        citation_counts = [len(r.get("citations", [])) for r in successful]
        citation_stats = {
            "min": min(citation_counts) if citation_counts else 0,
            "max": max(citation_counts) if citation_counts else 0,
            "mean": statistics.mean(citation_counts) if citation_counts else 0,
            "zero_citation_queries": sum(1 for c in citation_counts if c == 0)
        }

        # Build report
        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_questions": total,
                "successful": len(successful),
                "failed": len(failed),
                "total_batch_time_seconds": total_batch_time,
                "queries_per_second": total / total_batch_time if total_batch_time > 0 else 0
            },
            "timing_statistics": timing_stats,
            "confidence_distribution": dict(confidence_dist),
            "intent_classification": {
                "total_with_expected": len(intent_results),
                "correct": intent_correct,
                "accuracy": intent_accuracy,
                "confusion_matrix": {k: dict(v) for k, v in intent_matrix.items()}
            },
            "category_breakdown": category_stats,
            "citation_statistics": citation_stats,
            "errors": [
                {"question_id": r["question_id"], "query": r["query"], "error": r.get("error")}
                for r in failed
            ],
            "slowest_queries": sorted(
                [{"question_id": r["question_id"], "query": r["query"][:50], "time_ms": r["timings"]["total_ms"]}
                 for r in successful],
                key=lambda x: x["time_ms"],
                reverse=True
            )[:10],
            "lowest_confidence_queries": [
                {"question_id": r["question_id"], "query": r["query"][:50], "confidence": r["confidence_label"]}
                for r in successful if r.get("confidence_label") == "low"
            ][:10]
        }

        return report

    def save_results(self, report: Dict[str, Any], prefix: str = "batch"):
        """Save results and report to files."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"{prefix}_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to: {results_file}")

        # Save report
        report_file = self.output_dir / f"{prefix}_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_file}")

        # Save human-readable summary
        summary_file = self.output_dir / f"{prefix}_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self.format_report(report))
        print(f"Summary saved to: {summary_file}")

        return results_file, report_file, summary_file

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = []
        lines.append("="*70)
        lines.append("  PUBLIC PITCHBOOK OBSERVER - BATCH RUN REPORT")
        lines.append("="*70)
        lines.append("")

        # Metadata
        meta = report["metadata"]
        lines.append("SUMMARY")
        lines.append("-"*70)
        lines.append(f"  Generated:          {meta['generated_at']}")
        lines.append(f"  Total Questions:    {meta['total_questions']}")
        lines.append(f"  Successful:         {meta['successful']}")
        lines.append(f"  Failed:             {meta['failed']}")
        lines.append(f"  Total Time:         {meta['total_batch_time_seconds']:.2f} seconds")
        lines.append(f"  Throughput:         {meta['queries_per_second']:.2f} queries/second")
        lines.append("")

        # Timing
        timing = report.get("timing_statistics", {}).get("total", {})
        if timing:
            lines.append("TIMING STATISTICS (Total Response Time)")
            lines.append("-"*70)
            lines.append(f"  Minimum:            {timing.get('min_ms', 0):.2f} ms")
            lines.append(f"  Maximum:            {timing.get('max_ms', 0):.2f} ms")
            lines.append(f"  Mean:               {timing.get('mean_ms', 0):.2f} ms")
            lines.append(f"  Median:             {timing.get('median_ms', 0):.2f} ms")
            lines.append(f"  Std Dev:            {timing.get('stdev_ms', 0):.2f} ms")
            lines.append(f"  P95:                {timing.get('p95_ms', 0):.2f} ms")
            lines.append(f"  P99:                {timing.get('p99_ms', 0):.2f} ms")
            lines.append("")

        # Retrieval timing
        retrieval = report.get("timing_statistics", {}).get("retrieval", {})
        if retrieval:
            lines.append("RETRIEVAL TIMING")
            lines.append("-"*70)
            lines.append(f"  Mean:               {retrieval.get('mean_ms', 0):.2f} ms")
            lines.append(f"  Median:             {retrieval.get('median_ms', 0):.2f} ms")
            lines.append("")

        # Confidence distribution
        conf = report.get("confidence_distribution", {})
        if conf:
            lines.append("CONFIDENCE DISTRIBUTION")
            lines.append("-"*70)
            total_conf = sum(conf.values())
            for level in ["high", "medium", "low"]:
                count = conf.get(level, 0)
                pct = (count / total_conf * 100) if total_conf > 0 else 0
                bar = "█" * int(pct / 2)
                lines.append(f"  {level:8s}: {count:4d} ({pct:5.1f}%) {bar}")
            lines.append("")

        # Intent classification
        intent = report.get("intent_classification", {})
        if intent.get("total_with_expected", 0) > 0:
            lines.append("INTENT CLASSIFICATION")
            lines.append("-"*70)
            lines.append(f"  Accuracy:           {intent['accuracy']*100:.1f}%")
            lines.append(f"  Correct:            {intent['correct']} / {intent['total_with_expected']}")
            lines.append("")
            lines.append("  Confusion Matrix:")
            matrix = intent.get("confusion_matrix", {})
            for expected, actuals in matrix.items():
                for actual, count in actuals.items():
                    if count > 0:
                        match = "✓" if expected == actual else "✗"
                        lines.append(f"    {match} {expected:10s} -> {actual:10s}: {count}")
            lines.append("")

        # Category breakdown
        cats = report.get("category_breakdown", {})
        if cats:
            lines.append("CATEGORY BREAKDOWN")
            lines.append("-"*70)
            for cat, stats in sorted(cats.items()):
                conf_str = ", ".join(f"{k}:{v}" for k, v in stats["confidence_distribution"].items())
                lines.append(f"  {cat:20s}: {stats['count']:3d} queries, {stats['mean_time_ms']:.1f}ms avg")
                lines.append(f"                        Confidence: {conf_str}")
            lines.append("")

        # Citation stats
        cit = report.get("citation_statistics", {})
        if cit:
            lines.append("CITATION STATISTICS")
            lines.append("-"*70)
            lines.append(f"  Average citations:  {cit.get('mean', 0):.1f}")
            lines.append(f"  Min/Max:            {cit.get('min', 0)} / {cit.get('max', 0)}")
            lines.append(f"  Zero-citation:      {cit.get('zero_citation_queries', 0)} queries")
            lines.append("")

        # Slowest queries
        slowest = report.get("slowest_queries", [])
        if slowest:
            lines.append("SLOWEST QUERIES (Top 10)")
            lines.append("-"*70)
            for i, q in enumerate(slowest[:10], 1):
                lines.append(f"  {i:2d}. [{q['time_ms']:7.1f}ms] {q['query']}")
            lines.append("")

        # Errors
        errors = report.get("errors", [])
        if errors:
            lines.append("ERRORS")
            lines.append("-"*70)
            for e in errors:
                lines.append(f"  Q{e['question_id']}: {e['error']}")
            lines.append("")

        lines.append("="*70)
        lines.append("  END OF REPORT")
        lines.append("="*70)

        return "\n".join(lines)

    def print_report(self, report: Dict[str, Any]):
        """Print formatted report to console."""
        print(self.format_report(report))


def main():
    """Main entry point for batch runner."""
    parser = argparse.ArgumentParser(
        description="Run batch demo questions with profiling"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=get_question_categories() + ["all"],
        default="all",
        help="Question category to run (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions (0 = all)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of results per query (default: 8)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "vector", "keyword"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )

    args = parser.parse_args()

    # Select questions
    if args.category == "all":
        questions = DEMO_QUESTIONS
    else:
        questions = get_questions_by_category(args.category)

    if args.limit > 0:
        questions = questions[:args.limit]

    # Run batch
    runner = BatchRunner(output_dir=args.output_dir)
    report = runner.run_batch(
        questions=questions,
        top_k=args.top_k,
        mode=args.mode,
        verbose=not args.quiet
    )

    # Print report
    runner.print_report(report)

    # Save results
    if not args.no_save:
        runner.save_results(report)


if __name__ == "__main__":
    main()
