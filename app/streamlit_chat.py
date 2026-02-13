"""
PitchBook Observer Agent - Web Chat Interface
==============================================
Streamlit-based chat UI with integrated visualization and memory.
Run with: streamlit run app/streamlit_chat.py
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# Optional visualization imports
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PitchBook Observer Agent",
    page_icon="PB",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stChatMessage { max-width: 900px; margin: 0 auto; }

    .agent-header {
        text-align: center; padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e0e0e0; margin-bottom: 1rem;
    }
    .agent-header h1 { font-size: 1.6rem; font-weight: 600; color: #1a1a2e; margin: 0; }
    .agent-header p { font-size: 0.85rem; color: #888; margin: 0.25rem 0 0 0; }

    .status-online {
        display: inline-block; background: #d4edda; color: #155724;
        padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; margin-top: 0.5rem;
    }
    .status-offline {
        display: inline-block; background: #fff3cd; color: #856404;
        padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; margin-top: 0.5rem;
    }

    .citation-card {
        background: #f8f9fa; border-left: 3px solid #4a90d9;
        padding: 0.5rem 0.75rem; margin: 0.25rem 0;
        border-radius: 0 4px 4px 0; font-size: 0.85rem;
    }
    .citation-card .source-name { color: #4a90d9; font-weight: 600; }
    .citation-card .snippet { color: #555; font-size: 0.8rem; }

    .conf-high { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .conf-medium { background: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .conf-low { background: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }

    .kpi-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.5rem 0; }
    .kpi-chip {
        background: #e8f4fd; color: #1a5276;
        padding: 3px 10px; border-radius: 12px; font-size: 0.78rem;
    }

    .mem-stat { font-size: 0.8rem; color: #555; margin: 2px 0; }
    .mem-label { font-weight: 600; color: #333; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        from pipeline.agents.memory_agent import MemoryAgent
        MemoryAgent().clear()
    except Exception:
        pass
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "runtime_agent" not in st.session_state:
    st.session_state.runtime_agent = None
if "use_llm" not in st.session_state:
    st.session_state.use_llm = False
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------
def _ensure_demo_data():
    """Inject demo data if the index is empty (stateless cloud deploy)."""
    index_dir = Path(__file__).parent.parent / "indexes" / "chroma"
    if (index_dir / "vectors.npy").exists():
        return

    base = Path(__file__).parent.parent
    for d in ["data/raw", "data/clean", "data/meta", "data/events",
              "data/memory", "data/responses", "indexes/chroma",
              "db", "products", "batch_results", "runs"]:
        (base / d).mkdir(parents=True, exist_ok=True)

    from scripts.inject_demo_data import inject_demo_data
    inject_demo_data()


def init_agent(api_key: Optional[str] = None):
    """Initialize the Runtime Agent."""
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        st.session_state.use_llm = True

    try:
        _ensure_demo_data()

        from pipeline.config import load_pipeline_config
        from pipeline.database import init_database

        init_database()
        load_pipeline_config()

        from pipeline.agents.runtime_agent import RuntimeAgent
        agent = RuntimeAgent(
            rerank_strategy="cross_encoder",
            use_llm=st.session_state.use_llm,
            use_memory=True,
        )

        try:
            from pipeline.agents.memory_agent import MemoryAgent
            MemoryAgent().clear()
        except Exception:
            pass

        st.session_state.runtime_agent = agent
        st.session_state.agent_ready = True
        return True
    except Exception as e:
        st.error(f"Agent initialization failed: {e}")
        return False


def run_query(query: str) -> Dict[str, Any]:
    """Run a query through the Runtime Agent."""
    agent = st.session_state.runtime_agent
    if not agent:
        return {"answer": "Agent not initialized.", "confidence_label": "low",
                "citations": [], "timings": {}, "query_info": {}, "notes": []}
    return agent.run(query=query, top_k=8, rerank_top_k=5, mode="hybrid")


def get_memory_status() -> Dict[str, Any]:
    """Get current memory agent status for sidebar."""
    try:
        from pipeline.agents.memory_agent import MemoryAgent
        return MemoryAgent().get_status()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Inline visualization helpers
# ---------------------------------------------------------------------------
def render_confidence_gauge(conf_label: str):
    """Confidence gauge using Plotly."""
    if not HAS_PLOTLY:
        return
    conf_map = {"high": 85, "medium": 55, "low": 25}
    value = conf_map.get(conf_label, 50)
    color = "#2ecc71" if value > 70 else "#f39c12" if value > 40 else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": "Confidence", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33], "color": "#ffcccc"},
                {"range": [33, 66], "color": "#ffffcc"},
                {"range": [66, 100], "color": "#ccffcc"},
            ],
        },
        number={"suffix": "%", "font": {"size": 20}},
    ))
    fig.update_layout(height=180, margin=dict(t=40, b=10, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)


def render_timing_bars(timings: Dict[str, float]):
    """Pipeline timing breakdown as horizontal bars."""
    if not HAS_PLOTLY or not timings:
        return
    stages, values = [], []
    for key, label in [("retrieval_ms", "Retrieval"), ("reranking_ms", "Reranking"),
                       ("memory_ms", "Memory"), ("generation_ms", "Generation"),
                       ("validation_ms", "Validation")]:
        if key in timings:
            stages.append(label)
            values.append(round(timings[key], 1))
    if not stages:
        return

    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f39c12"]
    fig = go.Figure(go.Bar(
        x=values, y=stages, orientation="h",
        marker_color=colors[:len(stages)],
        text=[f"{v:.0f}ms" for v in values], textposition="auto",
    ))
    fig.update_layout(
        title={"text": "Pipeline Timing", "font": {"size": 14}},
        height=180, margin=dict(t=40, b=10, l=80, r=20),
        xaxis_title="ms", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_source_breakdown(citations: List[Dict]):
    """Pie chart of source types."""
    if not HAS_PLOTLY or not citations:
        return
    sources = {}
    for c in citations:
        src = c.get("source_kind", c.get("source_name", "Unknown"))
        sources[src] = sources.get(src, 0) + 1
    if not sources:
        return

    fig = go.Figure(go.Pie(
        labels=list(sources.keys()), values=list(sources.values()),
        hole=0.4, textinfo="label+value",
        marker=dict(colors=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]),
    ))
    fig.update_layout(
        title={"text": "Source Types", "font": {"size": 14}},
        height=200, margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Welcome / API key screen
# ---------------------------------------------------------------------------
if not st.session_state.agent_ready:
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key and env_key.startswith("sk-ant"):
        with st.spinner("Agent is initializing..."):
            if init_agent(env_key):
                st.rerun()

    st.markdown("""
    <div class="agent-header">
        <h1>PitchBook Observer Agent</h1>
        <p>AI-powered deal intelligence from public sources</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### Welcome")
        st.markdown(
            "This agent monitors public deal activity — funding rounds, "
            "acquisitions, IPOs — and answers your questions with cited sources."
        )
        st.markdown("")
        st.markdown("**Enter your Anthropic API key for the full Claude Opus experience.**")
        st.markdown(
            '<span style="font-size:0.8rem; color:#888;">'
            'Get one free at <a href="https://console.anthropic.com" target="_blank">'
            'console.anthropic.com</a></span>',
            unsafe_allow_html=True,
        )

        api_key = st.text_input(
            "API Key", type="password", placeholder="sk-ant-...",
            label_visibility="collapsed",
        )

        col_go, col_skip = st.columns(2)
        with col_go:
            if st.button("Go", type="primary", use_container_width=True):
                if api_key and api_key.startswith("sk-ant"):
                    with st.spinner("Agent is initializing..."):
                        if init_agent(api_key):
                            st.rerun()
                else:
                    st.warning("Please enter a valid API key (starts with sk-ant)")
        with col_skip:
            if st.button("Skip (offline mode)", use_container_width=True):
                with st.spinner("Agent is initializing..."):
                    if init_agent():
                        st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

# Header
mode_label = "Claude Opus" if st.session_state.use_llm else "Offline"
status_class = "status-online" if st.session_state.use_llm else "status-offline"
st.markdown(f"""
<div class="agent-header">
    <h1>PitchBook Observer Agent</h1>
    <span class="{status_class}">{mode_label}</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: controls + memory status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Agent Controls")

    if st.button("New Session", use_container_width=True):
        try:
            from pipeline.agents.memory_agent import MemoryAgent
            MemoryAgent().clear()
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()

    st.markdown("---")

    # Memory Agent Status
    st.markdown("### Memory Agent")
    mem_status = get_memory_status()
    if mem_status.get("has_memory", True) and mem_status.get("total_queries", 0) > 0:
        st.markdown(
            f'<div class="mem-stat"><span class="mem-label">Queries:</span> '
            f'{mem_status.get("total_queries", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="mem-stat"><span class="mem-label">Claims:</span> '
            f'{mem_status.get("claims_count", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="mem-stat"><span class="mem-label">Entities:</span> '
            f'{mem_status.get("entity_profiles_count", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="mem-stat"><span class="mem-label">Relationships:</span> '
            f'{mem_status.get("relationships_count", 0)}</div>',
            unsafe_allow_html=True,
        )
        coherence = mem_status.get("coherence", 0)
        st.progress(min(1.0, coherence), text=f"Coherence: {coherence:.0%}")

        if mem_status.get("entity"):
            st.markdown(
                f'<div class="mem-stat"><span class="mem-label">Focus:</span> '
                f'{mem_status["entity"]}</div>',
                unsafe_allow_html=True,
            )
        if mem_status.get("last_intent"):
            st.markdown(
                f'<div class="mem-stat"><span class="mem-label">Last intent:</span> '
                f'{mem_status["last_intent"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No memory yet - ask a question to start.")

    st.markdown("---")

    st.markdown("**Try asking:**")
    for q in [
        "What funding rounds were announced?",
        "Tell me about recent acquisitions",
        "Which investors are most active?",
        "What's happening in AI startups?",
        "Compare the top funded companies",
    ]:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    st.markdown("---")
    st.caption("PitchBook Observer Agent v2")


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input + response with inline visualization
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask about deals, funding, acquisitions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            t0 = time.time()
            result = run_query(prompt)
            elapsed = time.time() - t0

        st.session_state.query_count += 1

        answer = result.get("answer", "I couldn't find relevant information.")
        conf = result.get("confidence_label", "medium")
        citations = result.get("citations", [])
        timings = result.get("timings", {})
        query_info = result.get("query_info", {})
        kpis = result.get("kpis", [])
        mem_aug = query_info.get("memory_augmentation", {})

        # Answer
        st.markdown(answer)

        # Memory augmentation indicator
        if mem_aug.get("used"):
            st.caption(f"Memory: active ({mem_aug.get('reason', '')})")

        # KPI chips
        if kpis:
            chips = "".join(
                f'<span class="kpi-chip">{k.get("entity", "")}: '
                f'{k.get("metric_value", "")}</span>'
                for k in kpis[:8]
            )
            st.markdown(f'<div class="kpi-row">{chips}</div>', unsafe_allow_html=True)

        # Inline visualization panel
        if HAS_PLOTLY:
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            with viz_col1:
                render_confidence_gauge(conf)
            with viz_col2:
                render_timing_bars(timings)
            with viz_col3:
                render_source_breakdown(citations)

        # Citations expander
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for i, cit in enumerate(citations[:5], 1):
                    source = cit.get("source_name", "Unknown")
                    title = cit.get("title", "Untitled")
                    score = cit.get("score", 0)
                    snippet = cit.get("snippet", "")[:200]
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<span class="source-name">[{i}] {source}</span> — {title} '
                        f'(score: {score:.2f})<br>'
                        f'<span class="snippet">{snippet}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Footer
        total_ms = timings.get("total_ms", elapsed * 1000)
        model = query_info.get("model", "template")
        results_n = query_info.get("results_after_rerank", 0)
        st.caption(
            f"{total_ms:.0f}ms | {model} | "
            f"{results_n} sources reranked | "
            f"query #{st.session_state.query_count}"
        )

        # Store for chat history
        stored_answer = answer
        if citations:
            stored_answer += f"\n\n*{len(citations)} sources cited*"
        stored_answer += f"\n\nConfidence: **{conf.upper()}** | {total_ms:.0f}ms"

        st.session_state.messages.append({
            "role": "assistant",
            "content": stored_answer,
        })

        # Save visualization data
        try:
            responses_dir = Path(__file__).parent.parent / "data" / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)
            viz_data = {
                "response": result, "kpis": kpis,
                "visualizations": result.get("visualizations", []),
                "timestamp": datetime.utcnow().isoformat(),
            }
            (responses_dir / "latest_visualization.json").write_text(
                json.dumps(viz_data, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass
