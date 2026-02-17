# PresentationAgent Spec

**File:** `pipeline/agents/presentation_agent.py`

## Role

Extracts KPIs from query responses, generates visualization data, and writes to `data/responses/latest_visualization.json` for consumption by the dashboard.

## Interface

```python
agent = PresentationAgent()
result = agent.run(response=response_dict, launch_dashboard=False)
# → {"kpis_extracted": int, "chart_types": ["timeline", "bar", ...]}
```

## KPI Extraction

Parses the answer and citations for:
- Funding amounts (e.g. "$150M Series B")
- Valuation figures
- Company names and round types
- Investor names
- Dates and deal stages

## Visualization Output Contract

`data/responses/latest_visualization.json`:

```json
{
    "response": { ...RuntimeAgent response... },
    "kpis": [{"entity", "category", "metric_value", "unit"}],
    "visualizations": [
        {"type": "timeline", "data": [{"company", "amount", "round", "date"}]},
        {"type": "bar", "data": [...]}
    ],
    "timestamp": "ISO8601"
}
```

## Inline Charts (Streamlit)

The web chat UI (`app/streamlit_chat.py`) renders three Plotly charts inline per response:

| Chart | Data source |
|-------|-------------|
| Confidence gauge | `response.confidence_label` |
| Pipeline timing bars | `response.timings` |
| Source type donut | `response.citations[].source_kind` |
