"""
Streamlit Dashboard - Advanced KPI Visualization
=================================================
Interactive visualization with comparison matrices, network graphs,
waterfall charts, and statistical instruments.

Features:
    - Comparison matrix (entities x metrics)
    - Network graph (entity relationships)
    - Waterfall charts (cumulative funding)
    - Statistical gauges and distributions
    - Memory-augmented insights

Run with: streamlit run app/streamlit_dashboard.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Streamlit and visualization imports
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None
    pd = None
    np = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    px = None
    go = None

# Data directories
BASE_DIR = Path(__file__).parent.parent
RESPONSES_DIR = BASE_DIR / "data" / "responses"
MEMORY_DIR = BASE_DIR / "data" / "memory"


# =============================================================================
# Data Loading
# =============================================================================

def load_visualization_data() -> Optional[Dict]:
    """Load the latest visualization data."""
    viz_file = RESPONSES_DIR / "latest_visualization.json"
    if viz_file.exists():
        try:
            return json.loads(viz_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return None


def load_memory_data() -> Optional[Dict]:
    """Load SmartCard memory data."""
    memory_file = MEMORY_DIR / "smartcard.json"
    if memory_file.exists():
        try:
            return json.loads(memory_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return None


def get_memory_agent_data() -> Dict[str, Any]:
    """Get visualization data from Memory Agent."""
    try:
        from pipeline.agents.memory_agent import MemoryAgent
        agent = MemoryAgent()
        return agent.get_visualization_data()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# Advanced Visualizations
# =============================================================================

def render_network_graph(memory_data: Dict):
    """Render entity relationship network graph."""
    if not HAS_PLOTLY:
        st.warning("Plotly required for network graph")
        return

    st.subheader("Entity Relationship Network")

    relationships = memory_data.get("relationship_graph", [])
    profiles = memory_data.get("entity_profiles", {})

    if not relationships:
        st.info("No relationships discovered yet. Query about company connections, investors, or acquisitions.")
        return

    # Build node positions using simple force-directed layout simulation
    entities = set()
    for rel in relationships:
        entities.add(rel.get("source_entity", ""))
        entities.add(rel.get("target_entity", ""))
    entities.discard("")

    if not entities:
        return

    entity_list = list(entities)
    n = len(entity_list)

    # Simple circular layout
    import math
    positions = {}
    for i, entity in enumerate(entity_list):
        angle = 2 * math.pi * i / n
        positions[entity] = (math.cos(angle), math.sin(angle))

    # Create edge traces
    edge_traces = []
    for rel in relationships:
        src = rel.get("source_entity", "")
        tgt = rel.get("target_entity", "")
        rel_type = rel.get("relation_type", "related")

        if src in positions and tgt in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]

            # Color by relationship type
            colors = {
                "investor_of": "#2ecc71",
                "acquired": "#e74c3c",
                "competitor": "#f39c12",
                "partner": "#3498db"
            }
            color = colors.get(rel_type, "#95a5a6")

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color=color),
                hoverinfo='text',
                text=f"{src} → {tgt} ({rel_type})",
                showlegend=False
            ))

    # Create node trace
    node_x = [positions[e][0] for e in entity_list]
    node_y = [positions[e][1] for e in entity_list]
    node_sizes = [profiles.get(e, {}).get("mentions_count", 1) * 10 + 20 for e in entity_list]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color='#3498db',
            line=dict(width=2, color='white')
        ),
        text=entity_list,
        textposition="top center",
        hoverinfo='text',
        hovertext=[f"{e}: {profiles.get(e, {}).get('mentions_count', 0)} mentions" for e in entity_list]
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Entity Relationships",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    **Relationship Types:**
    - Investor (green) | Acquired (red) | Competitor (orange) | Partner (blue)
    """)


def render_comparison_matrix(memory_data: Dict):
    """Render entity comparison matrix."""
    if not HAS_PLOTLY:
        return

    st.subheader("Entity Comparison Matrix")

    profiles = memory_data.get("entity_profiles", {})
    claims = memory_data.get("knowledge_claims", [])

    if not profiles:
        st.info("No entity profiles built yet. Query about multiple companies to enable comparison.")
        return

    entities = list(profiles.keys())[:10]  # Limit to 10 entities
    metrics = ["Funding ($M)", "Valuation ($M)", "Mentions", "Relationships"]

    # Build matrix data
    matrix_data = []
    for entity in entities:
        profile = profiles.get(entity, {})
        rel_count = len([r for r in memory_data.get("relationship_graph", [])
                        if entity in [r.get("source_entity"), r.get("target_entity")]])

        # Parse funding from claims
        funding = 0
        valuation = 0
        for claim in claims:
            if claim.get("subject", "").lower() == entity.lower():
                value_str = claim.get("object_value", "")
                import re
                match = re.search(r'\$?([\d.]+)\s*(M|B)', value_str)
                if match:
                    val = float(match.group(1))
                    if match.group(2) == 'B':
                        val *= 1000
                    if claim.get("predicate") == "raised":
                        funding += val
                    elif claim.get("predicate") == "valued_at":
                        valuation = max(valuation, val)

        matrix_data.append({
            "Entity": entity,
            "Funding ($M)": funding,
            "Valuation ($M)": valuation,
            "Mentions": profile.get("mentions_count", 0),
            "Relationships": rel_count
        })

    if not matrix_data:
        return

    df = pd.DataFrame(matrix_data)

    # Heatmap of normalized values
    numeric_cols = ["Funding ($M)", "Valuation ($M)", "Mentions", "Relationships"]
    df_normalized = df[numeric_cols].copy()

    # Normalize each column
    for col in numeric_cols:
        max_val = df_normalized[col].max()
        if max_val > 0:
            df_normalized[col] = df_normalized[col] / max_val

    fig = go.Figure(data=go.Heatmap(
        z=df_normalized.values,
        x=numeric_cols,
        y=df["Entity"].tolist(),
        colorscale="Blues",
        showscale=True,
        text=df[numeric_cols].round(1).values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(
        title="Entity Comparison (Normalized)",
        height=max(300, len(entities) * 50),
        xaxis_title="Metrics",
        yaxis_title="Entities"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Also show raw data table
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)


def render_waterfall_chart(memory_data: Dict):
    """Render waterfall chart for cumulative funding."""
    if not HAS_PLOTLY:
        return

    st.subheader("Funding Waterfall")

    time_series = memory_data.get("time_series", [])
    funding_points = [p for p in time_series if p.get("metric") == "funding"]

    if not funding_points:
        st.info("No funding data tracked yet. Query about funding rounds to build the waterfall.")
        return

    # Sort by entity and timestamp
    funding_points.sort(key=lambda x: (x.get("entity", ""), x.get("timestamp", "")))

    # Build waterfall data
    labels = []
    values = []
    measures = []

    cumulative = 0
    for i, point in enumerate(funding_points[:15]):  # Limit to 15 steps
        entity = point.get("entity", "Unknown")
        amount = point.get("value", 0)

        labels.append(f"{entity}")
        values.append(amount)
        measures.append("relative")
        cumulative += amount

    # Add total
    labels.append("Total")
    values.append(cumulative)
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        name="Funding",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        textposition="outside",
        text=[f"${v:.0f}M" for v in values],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}}
    ))

    fig.update_layout(
        title="Cumulative Funding Flow",
        showlegend=False,
        height=400,
        xaxis_title="Funding Events",
        yaxis_title="Amount ($M)"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_statistical_instruments(memory_data: Dict, response: Dict):
    """Render statistical gauges and distributions."""
    if not HAS_PLOTLY:
        return

    st.subheader("Statistical Instruments")

    stats = memory_data.get("statistical_summary", {})
    epistemic = memory_data.get("epistemic_metadata", {})

    # Create gauge subplots
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}, {"type": "indicator"},
                {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["Confidence", "Data Quality", "Coverage", "Coherence"]
    )

    # 1. Response Confidence
    conf_map = {"high": 85, "medium": 55, "low": 25}
    conf_label = response.get("confidence_label", "medium")
    conf_value = conf_map.get(conf_label, 50)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=conf_value,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71" if conf_value > 70 else "#f39c12" if conf_value > 40 else "#e74c3c"},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 70], "color": "#ffffcc"},
                {"range": [70, 100], "color": "#ccffcc"}
            ]
        },
        number={"suffix": "%"}
    ), row=1, col=1)

    # 2. Data Quality (based on discrepancies)
    discrepancies = len(epistemic.get("unresolved_discrepancies", []))
    quality = max(0, 100 - discrepancies * 20)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=quality,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 80], "color": "#ffffcc"},
                {"range": [80, 100], "color": "#ccffcc"}
            ]
        },
        number={"suffix": "%"}
    ), row=1, col=2)

    # 3. Coverage (entities tracked)
    entities_count = stats.get("entities_tracked", 0)
    coverage = min(100, entities_count * 10)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=coverage,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#9b59b6"},
            "steps": [
                {"range": [0, 30], "color": "#f0e6f6"},
                {"range": [30, 70], "color": "#d4b8e0"},
                {"range": [70, 100], "color": "#b388c9"}
            ]
        },
        number={"suffix": "%"}
    ), row=1, col=3)

    # 4. Context Coherence
    coherence = epistemic.get("context_coherence_score", 0.5) * 100

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=coherence,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1abc9c"},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 70], "color": "#ffffcc"},
                {"range": [70, 100], "color": "#ccffcc"}
            ]
        },
        number={"suffix": "%"}
    ), row=1, col=4)

    fig.update_layout(height=250, margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of claims by confidence
    claims = memory_data.get("knowledge_claims", [])
    if claims:
        confidences = [c.get("confidence", 0) for c in claims]

        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="Claim Confidence Distribution",
                labels={"x": "Confidence", "y": "Count"}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Claims by predicate
            predicates = [c.get("predicate", "unknown") for c in claims]
            pred_counts = pd.Series(predicates).value_counts()

            fig_pie = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Claims by Type"
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)


def render_funding_timeline(viz_data: List[Dict], memory_data: Dict):
    """Enhanced funding timeline with memory context."""
    if not HAS_PLOTLY or not viz_data:
        return

    st.subheader("Funding Timeline")

    # Combine viz data with memory time series
    time_series = memory_data.get("time_series", [])
    funding_ts = [p for p in time_series if p.get("metric") == "funding"]

    # Prepare data
    df_data = []
    for item in viz_data:
        amount = item.get("amount", 0)
        if isinstance(amount, str):
            import re
            match = re.search(r'([\d.]+)', amount)
            amount = float(match.group(1)) if match else 0

        df_data.append({
            "Company": item.get("company", "Unknown"),
            "Amount ($M)": amount,
            "Round": item.get("round", "Unknown"),
            "Date": item.get("date", "Recent"),
            "Source": "Current Query"
        })

    # Add historical from memory
    for ts in funding_ts[-10:]:
        if ts.get("entity") not in [d["Company"] for d in df_data]:
            df_data.append({
                "Company": ts.get("entity", "Unknown"),
                "Amount ($M)": ts.get("value", 0),
                "Round": "Historical",
                "Date": ts.get("timestamp", "Past"),
                "Source": "Memory"
            })

    if not df_data:
        st.info("No funding data to display")
        return

    df = pd.DataFrame(df_data)

    # Bubble chart
    fig = px.scatter(
        df,
        x="Date",
        y="Amount ($M)",
        size="Amount ($M)",
        color="Company",
        symbol="Source",
        hover_data=["Round"],
        title="Funding Rounds (Current + Historical)",
        size_max=60
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_memory_insights(memory_data: Dict):
    """Render insights derived from memory."""
    st.subheader("Memory Insights")

    stats = memory_data.get("statistical_summary", {})
    profiles = memory_data.get("entity_profiles", {})
    claims = memory_data.get("knowledge_claims", [])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Funding Tracked",
            f"${stats.get('total_funding_observed', 0):,.0f}M"
        )

    with col2:
        st.metric(
            "Entities Profiled",
            stats.get("entities_tracked", 0)
        )

    with col3:
        st.metric(
            "Knowledge Claims",
            stats.get("claims_count", len(claims))
        )

    # Top entities by activity
    if profiles:
        st.markdown("**Most Active Entities:**")
        sorted_entities = sorted(
            profiles.items(),
            key=lambda x: x[1].get("mentions_count", 0),
            reverse=True
        )[:5]

        for entity, profile in sorted_entities:
            st.markdown(f"- **{entity}**: {profile.get('mentions_count', 0)} mentions")

    # Recent claims
    if claims:
        st.markdown("**Recent Verified Claims:**")
        high_conf = [c for c in claims if c.get("confidence", 0) >= 0.7][:3]
        for claim in high_conf:
            st.markdown(
                f"- {claim.get('subject', '?')} **{claim.get('predicate', '?')}** "
                f"{claim.get('object_value', '?')} (conf: {claim.get('confidence', 0):.0%})"
            )


def render_response_summary(response: Dict):
    """Render query response with confidence gauge."""
    query = response.get("query", "N/A")
    answer = response.get("answer", "No answer available")
    timings = response.get("timings", {})
    query_info = response.get("query_info", {})

    st.markdown(f"**Query:** {query}")

    # Memory augmentation status
    mem_aug = query_info.get("memory_augmentation", {})
    if mem_aug.get("used"):
        st.success(f"Memory augmentation: ENABLED ({mem_aug.get('reason', '')})")
    else:
        st.warning(f"Memory augmentation: DISABLED ({mem_aug.get('reason', '')})")

    # Answer and confidence
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.expander("Full Answer", expanded=True):
            st.markdown(answer)

    with col2:
        conf_map = {"high": 85, "medium": 55, "low": 25}
        conf_label = response.get("confidence_label", "medium")
        conf_value = conf_map.get(conf_label, 50)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_value,
            title={"text": "Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if conf_value > 70 else "#f39c12" if conf_value > 40 else "#e74c3c"},
                "steps": [
                    {"range": [0, 33], "color": "#ffcccc"},
                    {"range": [33, 66], "color": "#ffffcc"},
                    {"range": [66, 100], "color": "#ccffcc"}
                ]
            },
            number={"suffix": "%"}
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Timing metrics
    if timings:
        mcols = st.columns(5)
        mcols[0].metric("Total", f"{timings.get('total_ms', 0):.0f}ms")
        mcols[1].metric("Retrieval", f"{timings.get('retrieval_ms', 0):.0f}ms")
        mcols[2].metric("Rerank", f"{timings.get('reranking_ms', 0):.0f}ms")
        mcols[3].metric("Memory", f"{timings.get('memory_ms', 0):.0f}ms")
        mcols[4].metric("Generation", f"{timings.get('generation_ms', 0):.0f}ms")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Install with: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="PitchBook Observer - Advanced Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header with prominent refresh button
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.title("📊 PitchBook Observer - Analytics Dashboard")
        st.caption("Visualization powered by Memory Agent insights | Auto-updates on new queries")
    with header_col2:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.rerun()

    # Sidebar
    st.sidebar.title("Dashboard Controls")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()

    st.sidebar.markdown("---")

    # View selection
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Standard", "Network Analysis", "Comparison Matrix", "Statistical Deep Dive"]
    )

    # Load data
    viz_data = load_visualization_data()
    memory_data = load_memory_data() or {}

    # Summary metrics
    st.markdown("---")
    if viz_data:
        response = viz_data.get("response", {})
        kpis = viz_data.get("kpis", [])
        visualizations = viz_data.get("visualizations", [])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("KPIs", len(kpis))
        col2.metric("Entities", len(memory_data.get("entity_profiles", {})))
        col3.metric("Claims", len(memory_data.get("knowledge_claims", [])))
        col4.metric("Relationships", len(memory_data.get("relationship_graph", [])))
        col5.metric("Queries", memory_data.get("total_queries", 0))

        st.markdown("---")

        # Response Summary
        st.header("Query Response")
        render_response_summary(response)

        st.markdown("---")

        # View-specific content
        if view_mode == "Standard":
            # Standard visualizations
            st.header("Visualizations")

            # Funding timeline
            funding_data = []
            for v in visualizations:
                if v.get("type") == "timeline":
                    funding_data = v.get("data", [])
                    break
            if funding_data:
                render_funding_timeline(funding_data, memory_data)

            # Memory insights
            if memory_data:
                render_memory_insights(memory_data)

        elif view_mode == "Network Analysis":
            st.header("Entity Network Analysis")
            render_network_graph(memory_data)

            # Also show relationships table
            relationships = memory_data.get("relationship_graph", [])
            if relationships:
                with st.expander("Relationship Details"):
                    rel_df = pd.DataFrame(relationships)
                    st.dataframe(rel_df, use_container_width=True)

        elif view_mode == "Comparison Matrix":
            st.header("Entity Comparison")
            render_comparison_matrix(memory_data)
            render_waterfall_chart(memory_data)

        elif view_mode == "Statistical Deep Dive":
            st.header("Statistical Analysis")
            render_statistical_instruments(memory_data, response)

        st.markdown("---")

        # Citations
        st.header("Source Citations")
        citations = response.get("citations", [])
        if citations:
            for i, cit in enumerate(citations[:6], 1):
                with st.expander(f"{i}. {cit.get('title', 'Untitled')[:50]}..."):
                    st.markdown(f"**Source:** {cit.get('source_name', 'Unknown')}")
                    st.markdown(f"**Score:** {cit.get('score', 0):.3f}")
                    st.markdown(f"**Snippet:** {cit.get('snippet', '')[:200]}...")

    else:
        st.warning("No data available. Run a query using the chat interface first.")

        if st.button("Load Demo Data"):
            # Create demo data
            demo_viz = {
                "response": {
                    "query": "What funding rounds were announced?",
                    "answer": "Based on recent data, Stripe raised $600M Series I, OpenAI received $10B from Microsoft, and Anthropic raised $450M Series C.",
                    "confidence_label": "high",
                    "citations": [
                        {"title": "Stripe Series I", "source_name": "TechCrunch", "score": 0.92},
                        {"title": "OpenAI Microsoft", "source_name": "Bloomberg", "score": 0.88}
                    ],
                    "timings": {"total_ms": 1250, "retrieval_ms": 320, "reranking_ms": 180, "memory_ms": 50, "generation_ms": 700},
                    "query_info": {"memory_augmentation": {"used": True, "reason": "context_coherent"}}
                },
                "kpis": [
                    {"entity": "Stripe", "category": "funding", "metric_value": 600},
                    {"entity": "OpenAI", "category": "funding", "metric_value": 10000},
                    {"entity": "Anthropic", "category": "funding", "metric_value": 450}
                ],
                "visualizations": [
                    {"type": "timeline", "data": [
                        {"company": "Stripe", "amount": 600, "round": "Series I", "date": "2026-W05"},
                        {"company": "OpenAI", "amount": 10000, "round": "Investment", "date": "2026-W04"},
                        {"company": "Anthropic", "amount": 450, "round": "Series C", "date": "2026-W05"}
                    ]}
                ]
            }

            demo_memory = {
                "entity_profiles": {
                    "Stripe": {"mentions_count": 5, "funding_history": []},
                    "OpenAI": {"mentions_count": 3, "funding_history": []},
                    "Anthropic": {"mentions_count": 2, "funding_history": []}
                },
                "knowledge_claims": [
                    {"subject": "Stripe", "predicate": "raised", "object_value": "$600M", "confidence": 0.92},
                    {"subject": "OpenAI", "predicate": "raised", "object_value": "$10B", "confidence": 0.88},
                    {"subject": "Anthropic", "predicate": "raised", "object_value": "$450M", "confidence": 0.85}
                ],
                "relationship_graph": [
                    {"source_entity": "Microsoft", "target_entity": "OpenAI", "relation_type": "investor_of", "strength": 0.9},
                    {"source_entity": "Google", "target_entity": "Anthropic", "relation_type": "investor_of", "strength": 0.8},
                    {"source_entity": "OpenAI", "target_entity": "Anthropic", "relation_type": "competitor", "strength": 0.7}
                ],
                "time_series": [
                    {"entity": "Stripe", "metric": "funding", "value": 600, "timestamp": "2026-W05"},
                    {"entity": "OpenAI", "metric": "funding", "value": 10000, "timestamp": "2026-W04"}
                ],
                "statistical_summary": {"total_funding_observed": 11050, "entities_tracked": 3, "claims_count": 3},
                "epistemic_metadata": {"context_coherence_score": 0.85, "unresolved_discrepancies": []},
                "total_queries": 5
            }

            RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)

            (RESPONSES_DIR / "latest_visualization.json").write_text(json.dumps(demo_viz, indent=2))
            (MEMORY_DIR / "smartcard.json").write_text(json.dumps(demo_memory, indent=2))

            st.rerun()

    # Footer
    st.markdown("---")
    st.caption(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Public PitchBook Observer | Claude Opus 4.5"
    )


if __name__ == "__main__":
    main()
