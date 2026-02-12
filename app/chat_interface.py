"""
Interactive Chat Interface (v2)
===============================
CLI-based chat interface using the Runtime Agent.
Features: LLM generation, reranking, memory, auto-visualization.
"""

import sys
import os
import json
import time
import subprocess
import webbrowser
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present (for ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.config import load_pipeline_config
from pipeline.database import init_database


class ChatInterface:
    """Interactive CLI chat interface with Runtime Agent."""

    def __init__(self, use_llm: bool = False, use_memory: bool = True, auto_visualize: bool = True):
        """
        Initialize chat interface.

        Args:
            use_llm: If True, use Claude LLM for generation (requires ANTHROPIC_API_KEY)
            use_memory: If True, use Memory Agent for session context
            auto_visualize: If True, launch Streamlit dashboard on startup
        """
        self.use_llm = use_llm
        self.use_memory = use_memory
        self.auto_visualize = auto_visualize
        self.runtime_agent = None
        self.presentation_agent = None
        self.config = None
        self.history: List[Dict[str, Any]] = []
        self.session_start = datetime.utcnow()
        self._dashboard_process = None

    def _clear_episodic_memory(self):
        """Clear SmartCard memory for episodic (session-scoped) behavior."""
        try:
            from pipeline.agents.memory_agent import MemoryAgent
            mem = MemoryAgent()
            if mem.clear():
                print("         Episodic memory cleared (fresh session)")
            else:
                print("         Episodic memory: clean slate")
        except Exception as e:
            print(f"         Could not clear memory: {e}")

    def initialize(self):
        """Initialize the chat system."""
        print("\n" + "="*60)
        print("  Public PitchBook Observer - Chat Interface v2")
        print("  (Runtime Agent + Episodic Memory + Visualization)")
        print("="*60)
        print("\nInitializing...")

        # Initialize database
        print("  [1/4] Checking database...")
        init_database()

        # Load config
        print("  [2/4] Loading configuration...")
        self.config = load_pipeline_config()

        # Initialize Runtime Agent
        print("  [3/4] Loading Runtime Agent...")
        try:
            from pipeline.agents.runtime_agent import RuntimeAgent
            self.runtime_agent = RuntimeAgent(
                rerank_strategy="cross_encoder",
                use_llm=self.use_llm,
                use_memory=self.use_memory
            )
            status = self.runtime_agent.get_status()
            print(f"         LLM: {'enabled' if self.use_llm else 'disabled (template mode)'}")
            print(f"         Reranking: enabled (strategy: {status.get('rerank_strategy', 'cross_encoder')})")
            print(f"         Memory: {'enabled (episodic)' if self.use_memory else 'disabled'}")

            # Clear memory at session start for episodic behavior
            if self.use_memory:
                self._clear_episodic_memory()
        except Exception as e:
            print(f"  WARNING: Could not load Runtime Agent: {e}")
            print("  Falling back to legacy mode.")
            self.runtime_agent = None

        # Launch visualization dashboard
        if self.auto_visualize:
            print("  [4/4] Launching visualization dashboard...")
            self._launch_dashboard()
        else:
            print("  [4/4] Visualization: disabled")

        print("\nReady! Type your question or 'help' for commands.\n")

    def _launch_dashboard(self):
        """Launch Streamlit dashboard in background."""
        try:
            dashboard_path = Path(__file__).parent / "streamlit_dashboard.py"
            if dashboard_path.exists():
                self._dashboard_process = subprocess.Popen(
                    [
                        sys.executable, "-m", "streamlit", "run",
                        str(dashboard_path),
                        "--server.port", "8501",
                        "--server.headless", "true"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Give it a moment to start
                time.sleep(2)
                webbrowser.open("http://localhost:8501")
                print("         Dashboard launched at http://localhost:8501")
            else:
                print("         Dashboard not found, skipping")
        except Exception as e:
            print(f"         Could not launch dashboard: {e}")

    def process_query(
        self,
        query: str,
        top_k: int = 8,
        rerank_top_k: int = 5,
        mode: str = "hybrid",
        filter_source_kinds: Optional[List[str]] = None,
        filter_weeks: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query using Runtime Agent.

        Returns dict with: answer, confidence, citations, timing, etc.
        """
        if self.runtime_agent:
            # Use Runtime Agent (v2)
            result = self.runtime_agent.run(
                query=query,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                mode=mode,
                filter_source_kinds=filter_source_kinds,
                filter_weeks=filter_weeks
            )
        else:
            # Fallback to legacy mode
            result = self._legacy_process_query(
                query, top_k, mode, filter_source_kinds, filter_weeks, verbose
            )

        # Add to history
        self.history.append(result)

        return result

    def _legacy_process_query(
        self,
        query: str,
        top_k: int,
        mode: str,
        filter_source_kinds: Optional[List[str]],
        filter_weeks: Optional[List[str]],
        verbose: bool
    ) -> Dict[str, Any]:
        """Legacy query processing (without Runtime Agent)."""
        from pipeline.index import get_retriever
        from app.main import (
            classify_intent, get_retrieval_filters, generate_answer, validate_response
        )

        start_time = time.perf_counter()
        timings = {}

        # Initialize retriever if needed
        retriever = get_retriever(self.config)

        # 1. Parse: Classify intent
        t0 = time.perf_counter()
        intent = classify_intent(query)
        timings["intent_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Abstract: Determine filters
        filter_fn = get_retrieval_filters(intent, filter_source_kinds, filter_weeks)

        # 3. Retrieve: Search corpus
        t0 = time.perf_counter()
        results = retriever.search(query=query, top_k=top_k, mode=mode, filter_fn=filter_fn)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # Build citations
        citations = []
        for result in results:
            meta = result.get("metadata", {})
            citations.append({
                "url": meta.get("canonical_url", ""),
                "title": meta.get("title", "Untitled"),
                "snippet": meta.get("text", "")[:300],
                "source_name": meta.get("source_name"),
                "source_kind": meta.get("source_kind"),
                "week": meta.get("week"),
                "score": result.get("score")
            })

        # 4. Generate: Create answer
        t0 = time.perf_counter()
        chat_cfg = self.config.chat if self.config else {}
        answer, confidence = generate_answer(query, intent, results, chat_cfg)
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Validate
        notes = validate_response(
            [type('obj', (object,), c)() for c in citations],
            answer
        )

        timings["total_ms"] = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "answer": answer,
            "confidence_label": confidence,
            "citations": citations,
            "notes": notes,
            "query_info": {"intent": intent.value, "mode": mode, "results_count": len(results)},
            "timings": timings,
            "timestamp": datetime.utcnow().isoformat()
        }

    def format_response(self, result: Dict[str, Any], show_timing: bool = True) -> str:
        """Format a response for display."""
        lines = []

        # Answer
        lines.append("\n" + "-"*60)
        lines.append("ANSWER:")
        lines.append("-"*60)
        lines.append(result["answer"])

        # Confidence
        conf = result["confidence_label"]
        conf_display = {"high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}.get(conf, conf)
        lines.append(f"\nConfidence: {conf_display}")

        # Query info
        query_info = result.get("query_info", {})
        model = query_info.get("model", "template")
        lines.append(f"Model: {model}")

        # Citations
        if result["citations"]:
            lines.append("\n" + "-"*60)
            lines.append(f"CITATIONS ({len(result['citations'])} sources):")
            lines.append("-"*60)
            for i, cit in enumerate(result["citations"][:5], 1):
                score = f" (score: {cit['score']:.3f})" if cit.get('score') else ""
                lines.append(f"\n[{i}] {cit['title']}{score}")
                lines.append(f"    URL: {cit['url']}")
                lines.append(f"    Source: {cit.get('source_name', 'Unknown')} ({cit.get('source_kind', '')})")
                if cit.get('snippet'):
                    snippet = cit['snippet'][:150].replace('\n', ' ')
                    lines.append(f"    Snippet: {snippet}...")

        # Notes
        if result["notes"]:
            lines.append("\n" + "-"*60)
            lines.append("NOTES:")
            lines.append("-"*60)
            for note in result["notes"]:
                lines.append(f"  * {note}")

        # Timing
        if show_timing:
            lines.append("\n" + "-"*60)
            lines.append("TIMING:")
            lines.append("-"*60)
            timings = result.get("timings", {})
            lines.append(f"  Retrieval:  {timings.get('retrieval_ms', 0):.2f} ms")
            lines.append(f"  Reranking:  {timings.get('reranking_ms', 0):.2f} ms")
            lines.append(f"  Generation: {timings.get('generation_ms', 0):.2f} ms")
            lines.append(f"  Total:      {timings.get('total_ms', 0):.2f} ms")

        lines.append("")
        return "\n".join(lines)

    def print_help(self):
        """Print help message."""
        help_text = """
Commands:
  help              Show this help message
  quit / exit       Exit the chat
  clear             Clear chat history
  history           Show query history
  stats             Show session statistics
  memory            Show current memory context
  mode <m>          Set retrieval mode (hybrid/vector/keyword)
  topk <n>          Set number of results (1-50)
  timing on/off     Toggle timing display
  llm on/off        Toggle LLM generation
  viz               Refresh visualization dashboard
  export <file>     Export history to JSON file

Query modifiers (append to query):
  --deals           Filter to deal-related sources
  --news            Filter to news sources only
  --filings         Filter to SEC filings only

Examples:
  What funding rounds were announced?
  Tell me about recent acquisitions --deals
  What's happening in AI startups?
"""
        print(help_text)

    def run(self):
        """Run the interactive chat loop."""
        self.initialize()

        # Settings
        mode = "hybrid"
        top_k = 8
        rerank_top_k = 5
        show_timing = True

        while True:
            try:
                # Get input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                cmd = user_input.lower()

                if cmd in ("quit", "exit", "q"):
                    # Clear episodic memory on session end
                    if self.use_memory:
                        self._clear_episodic_memory()
                    print("\nGoodbye!")
                    if self._dashboard_process:
                        self._dashboard_process.terminate()
                    break

                elif cmd == "help":
                    self.print_help()
                    continue

                elif cmd == "clear":
                    self.history = []
                    print("History cleared.")
                    continue

                elif cmd == "history":
                    if not self.history:
                        print("No history yet.")
                    else:
                        print(f"\nQuery History ({len(self.history)} queries):")
                        for i, h in enumerate(self.history[-10:], 1):
                            print(f"  {i}. {h['query'][:50]}... [{h['confidence_label']}]")
                    continue

                elif cmd == "memory":
                    if self.use_memory:
                        try:
                            from pipeline.agents.memory_agent import MemoryAgent
                            mem = MemoryAgent()
                            data = mem.load()
                            if data:
                                print("\nMemory Context:")
                                print(f"  Session: {data.get('session_id')}")
                                print(f"  Last query: {data.get('last_query', 'N/A')[:50]}...")
                                print(f"  Topics: {', '.join(data.get('topics_covered', []))}")
                                print(f"  Entities: {', '.join(data.get('entities_discussed', [])[:5])}")
                            else:
                                print("No memory stored yet.")
                        except Exception as e:
                            print(f"Could not load memory: {e}")
                    else:
                        print("Memory is disabled.")
                    continue

                elif cmd == "stats":
                    total_queries = len(self.history)
                    if total_queries > 0:
                        avg_time = sum(h['timings']['total_ms'] for h in self.history) / total_queries
                        conf_dist = {}
                        for h in self.history:
                            c = h['confidence_label']
                            conf_dist[c] = conf_dist.get(c, 0) + 1
                        print(f"\nSession Statistics:")
                        print(f"  Total queries: {total_queries}")
                        print(f"  Avg response time: {avg_time:.2f} ms")
                        print(f"  Confidence distribution: {conf_dist}")
                        print(f"  LLM enabled: {self.use_llm}")
                        print(f"  Memory enabled: {self.use_memory}")
                    else:
                        print("No queries yet.")
                    continue

                elif cmd.startswith("mode "):
                    new_mode = cmd.split()[1]
                    if new_mode in ("hybrid", "vector", "keyword"):
                        mode = new_mode
                        print(f"Mode set to: {mode}")
                    else:
                        print("Invalid mode. Use: hybrid, vector, or keyword")
                    continue

                elif cmd.startswith("topk "):
                    try:
                        new_k = int(cmd.split()[1])
                        if 1 <= new_k <= 50:
                            top_k = new_k
                            print(f"Top-k set to: {top_k}")
                        else:
                            print("Top-k must be between 1 and 50")
                    except ValueError:
                        print("Invalid number")
                    continue

                elif cmd.startswith("timing "):
                    if "on" in cmd:
                        show_timing = True
                        print("Timing display enabled")
                    elif "off" in cmd:
                        show_timing = False
                        print("Timing display disabled")
                    continue

                elif cmd.startswith("llm "):
                    if "on" in cmd:
                        self.use_llm = True
                        if self.runtime_agent:
                            self.runtime_agent.use_llm = True
                        print("LLM generation enabled (requires ANTHROPIC_API_KEY)")
                    elif "off" in cmd:
                        self.use_llm = False
                        if self.runtime_agent:
                            self.runtime_agent.use_llm = False
                        print("LLM generation disabled (template mode)")
                    continue

                elif cmd == "viz":
                    print("Refreshing dashboard...")
                    webbrowser.open("http://localhost:8501")
                    continue

                elif cmd.startswith("export "):
                    filename = cmd.split(maxsplit=1)[1]
                    try:
                        with open(filename, 'w') as f:
                            json.dump(self.history, f, indent=2)
                        print(f"History exported to {filename}")
                    except Exception as e:
                        print(f"Export failed: {e}")
                    continue

                # Parse query modifiers
                filter_source_kinds = None
                query = user_input

                if "--deals" in query:
                    filter_source_kinds = ["filing", "pr_wire", "news"]
                    query = query.replace("--deals", "").strip()
                elif "--news" in query:
                    filter_source_kinds = ["news"]
                    query = query.replace("--news", "").strip()
                elif "--filings" in query:
                    filter_source_kinds = ["filing"]
                    query = query.replace("--filings", "").strip()

                # Process query
                print("\nProcessing...")
                result = self.process_query(
                    query=query,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k,
                    mode=mode,
                    filter_source_kinds=filter_source_kinds,
                    verbose=True
                )

                # Display response
                print(self.format_response(result, show_timing=show_timing))

            except KeyboardInterrupt:
                # Clear episodic memory on forced exit
                if self.use_memory:
                    self._clear_episodic_memory()
                if self._dashboard_process:
                    self._dashboard_process.terminate()
                print("\n\nSession ended. Episodic memory cleared.")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Entry point for chat interface."""
    import argparse

    parser = argparse.ArgumentParser(description="PitchBook Observer Chat Interface")
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    parser.add_argument("--no-viz", action="store_true", help="Disable auto-visualization")

    args = parser.parse_args()

    # Clear dashboard flag for new session (allows auto-launch on first response)
    responses_dir = Path(__file__).parent.parent / "data" / "responses"
    dashboard_flag = responses_dir / ".dashboard_launched"
    if dashboard_flag.exists():
        dashboard_flag.unlink()
        print("Dashboard flag cleared - will auto-launch on first response")

    chat = ChatInterface(
        use_llm=args.llm,
        use_memory=not args.no_memory,
        auto_visualize=not args.no_viz
    )
    chat.run()


if __name__ == "__main__":
    main()
