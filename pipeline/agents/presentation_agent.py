"""
Presentation Agent
==================
Visualization and KPI matrix rendering.

Converts RAG responses into visual formats:
    - Enterprise KPI matrix (companies x categories)
    - Funding timelines and bubble charts
    - Growth gauges and confidence meters
    - Interactive dashboards with auto-chart selection

Triggered by: Query response or manual invocation
Output: Streamlit dashboard popup or data export
"""

import json
import re
import subprocess
import sys
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_base import AgentBase, RESPONSES_DIR

# Import intelligent KPI extractor
try:
    from app.kpi_extractor import (
        IntelligentKPIExtractor,
        extract_kpis_with_intelligence,
        ChartType,
        ExtractedKPI,
        VisualizationSpec
    )
    HAS_INTELLIGENT_EXTRACTOR = True
except ImportError:
    HAS_INTELLIGENT_EXTRACTOR = False


@dataclass
class EnterpriseKPI:
    """
    KPI data point for enterprise visualization.

    Represents a single metric extracted from
    retrieved documents.
    """
    company: str
    category: str
    metric: str
    value: str
    source: str
    source_kind: str
    confidence: float
    week: str
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PresentationAgent(AgentBase):
    """
    Handles visualization and KPI matrix generation.

    Responsibilities:
        1. Extract KPIs from chat responses
        2. Generate matrix data structures
        3. Launch Streamlit dashboard
        4. Export data for external tools

    KPI Categories:
        - funding: Funding rounds, valuations
        - acquisition: M&A activity
        - leadership: Executive changes
        - product: Product launches
        - market: Market trends
    """

    # KPI category patterns
    KPI_PATTERNS = {
        "funding": [
            r"raised?\s+\$[\d.]+\s*(?:million|billion|M|B)",
            r"series\s+[A-Z]",
            r"funding\s+round",
            r"valuation",
            r"seed\s+(?:round|funding)"
        ],
        "acquisition": [
            r"acquir(?:ed|es|ing)",
            r"merger",
            r"acquisition",
            r"buyout",
            r"takeover"
        ],
        "leadership": [
            r"(?:CEO|CTO|CFO|COO)\b",
            r"appoint(?:ed|s|ment)",
            r"executive",
            r"founder",
            r"leadership"
        ],
        "product": [
            r"launch(?:ed|es|ing)",
            r"announc(?:ed|es|ing)",
            r"release(?:d|s)?",
            r"new\s+product",
            r"feature"
        ],
        "market": [
            r"market\s+(?:share|growth|trend)",
            r"industry",
            r"sector",
            r"competitive"
        ]
    }

    # Known company patterns
    COMPANY_PATTERNS = [
        r"Stripe", r"OpenAI", r"Anthropic", r"SpaceX", r"Databricks",
        r"Figma", r"Notion", r"Canva", r"Discord", r"Snap",
        r"[A-Z][a-z]+(?:\.(?:ai|io|com|co))?"  # Generic company pattern
    ]

    def __init__(self, streamlit_port: int = 8501):
        super().__init__("presentation_agent")
        self.streamlit_port = streamlit_port
        self._streamlit_process = None

    def run(
        self,
        response: Optional[Dict[str, Any]] = None,
        launch_dashboard: bool = True
    ) -> Dict[str, Any]:
        """
        Process response and optionally launch visualization.

        Args:
            response: Chat response to visualize (loads latest if None)
            launch_dashboard: If True, launch Streamlit dashboard

        Returns:
            Visualization data with KPIs, matrix, and visualization specs
        """
        start_time = datetime.utcnow()

        # Load response if not provided
        if response is None:
            response = self._load_latest_response()

        if response is None:
            self.logger.warning("No response available for visualization")
            return {
                "status": "no_data",
                "message": "No response data available"
            }

        # Use intelligent extractor if available
        if HAS_INTELLIGENT_EXTRACTOR:
            self.logger.info("Using intelligent KPI extraction")
            intelligent_result = extract_kpis_with_intelligence(response)
            kpis_data = intelligent_result["kpis"]
            visualizations = intelligent_result["visualizations"]
            summary = intelligent_result["summary"]
        else:
            # Fallback to basic extraction
            self.logger.info("Using basic KPI extraction")
            kpis = self.extract_kpis(response)
            kpis_data = [kpi.to_dict() for kpi in kpis]
            visualizations = []
            summary = {"total_kpis": len(kpis)}

        self.logger.info(f"Extracted {len(kpis_data)} KPIs")

        # Generate matrix data (for backward compatibility)
        matrix_data = self._generate_matrix_from_kpis(kpis_data)

        # Save visualization data with enhanced structure
        viz_data = {
            "response": response,
            "kpis": kpis_data,
            "matrix": matrix_data,
            "visualizations": visualizations,
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat()
        }
        viz_file = self._save_visualization_data(viz_data)

        # Launch dashboard if requested
        dashboard_url = None
        if launch_dashboard:
            dashboard_url = self.launch_dashboard()

        duration = (datetime.utcnow() - start_time).total_seconds()

        result = {
            "status": "success",
            "kpis_extracted": len(kpis_data),
            "companies": summary.get("entities", matrix_data.get("companies", [])),
            "categories": summary.get("categories", matrix_data.get("categories", [])),
            "chart_types": [v["type"] for v in visualizations] if visualizations else ["matrix"],
            "visualization_file": str(viz_file),
            "dashboard_url": dashboard_url,
            "duration_seconds": duration
        }

        self.emit_event("visualization_generated", {
            "kpis_count": len(kpis_data),
            "companies_count": len(result["companies"]),
            "chart_types": result["chart_types"],
            "dashboard_launched": dashboard_url is not None
        })

        return result

    def _generate_matrix_from_kpis(self, kpis_data: List[Dict]) -> Dict[str, Any]:
        """Generate matrix structure from KPI data."""
        companies = set()
        categories = set()
        data = {}

        for kpi in kpis_data:
            company = kpi.get("entity", kpi.get("company", "Unknown"))
            category = kpi.get("category", "other")

            companies.add(company)
            categories.add(category)

            key = f"{company}|{category}"
            if key not in data:
                data[key] = []
            data[key].append(kpi)

        return {
            "companies": sorted(companies),
            "categories": sorted(categories),
            "data": data,
            "total_kpis": len(kpis_data),
            "summary": {
                "companies_count": len(companies),
                "categories_count": len(categories),
                "cells_with_data": len(data)
            }
        }

    def extract_kpis(self, response: Dict[str, Any]) -> List[EnterpriseKPI]:
        """
        Extract KPIs from a chat response.

        Args:
            response: Chat response with citations

        Returns:
            List of extracted KPI objects
        """
        kpis = []
        citations = response.get("citations", [])

        for citation in citations:
            title = citation.get("title", "")
            snippet = citation.get("snippet", "")
            source_name = citation.get("source_name", "Unknown")
            source_kind = citation.get("source_kind", "unknown")
            score = citation.get("score", 0)
            week = citation.get("week", "")
            url = citation.get("url", "")

            # Combined text for analysis
            text = f"{title} {snippet}"

            # Detect category
            category = self._detect_category(text, source_kind)
            if not category:
                continue

            # Extract company
            company = self._extract_company(title, snippet)

            # Extract metric and value
            metric, value = self._extract_metric_value(text, category)

            kpi = EnterpriseKPI(
                company=company,
                category=category,
                metric=metric,
                value=value,
                source=source_name,
                source_kind=source_kind,
                confidence=score if isinstance(score, float) else 0.0,
                week=week,
                url=url
            )
            kpis.append(kpi)

        return kpis

    def generate_matrix(self, kpis: List[EnterpriseKPI]) -> Dict[str, Any]:
        """
        Generate matrix data structure for visualization.

        Args:
            kpis: List of extracted KPIs

        Returns:
            Matrix data with companies, categories, and cell data
        """
        companies = set()
        categories = set()
        data = {}

        for kpi in kpis:
            companies.add(kpi.company)
            categories.add(kpi.category)

            key = f"{kpi.company}|{kpi.category}"
            if key not in data:
                data[key] = []
            data[key].append(kpi.to_dict())

        return {
            "companies": sorted(companies),
            "categories": sorted(categories),
            "data": data,
            "total_kpis": len(kpis),
            "summary": {
                "companies_count": len(companies),
                "categories_count": len(categories),
                "cells_with_data": len(data)
            }
        }

    def launch_dashboard(self) -> Optional[str]:
        """
        Launch Streamlit dashboard in browser.

        Returns:
            Dashboard URL if successful, None otherwise
        """
        dashboard_path = Path(__file__).parent.parent.parent / "app" / "streamlit_dashboard.py"

        if not dashboard_path.exists():
            self.logger.warning(f"Dashboard not found: {dashboard_path}")
            return None

        try:
            # Start Streamlit process
            self._streamlit_process = subprocess.Popen(
                [
                    "streamlit", "run",
                    str(dashboard_path),
                    "--server.port", str(self.streamlit_port),
                    "--server.headless", "true"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Open browser
            url = f"http://localhost:{self.streamlit_port}"
            webbrowser.open(url)

            self.logger.info(f"Dashboard launched at: {url}")
            return url

        except Exception as e:
            self.logger.error(f"Failed to launch dashboard: {e}")
            return None

    def stop_dashboard(self) -> None:
        """Stop the Streamlit dashboard process."""
        if self._streamlit_process:
            self._streamlit_process.terminate()
            self._streamlit_process = None
            self.logger.info("Dashboard stopped")

    def export_data(
        self,
        kpis: List[EnterpriseKPI],
        format: str = "json"
    ) -> str:
        """
        Export KPI data in specified format.

        Args:
            kpis: List of KPIs to export
            format: Export format (json, csv)

        Returns:
            Path to exported file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_dir = RESPONSES_DIR / "exports"
        export_dir.mkdir(exist_ok=True)

        if format == "csv":
            import csv
            export_file = export_dir / f"kpis_{timestamp}.csv"
            with open(export_file, "w", newline="", encoding="utf-8") as f:
                if kpis:
                    writer = csv.DictWriter(f, fieldnames=kpis[0].to_dict().keys())
                    writer.writeheader()
                    for kpi in kpis:
                        writer.writerow(kpi.to_dict())
        else:
            export_file = export_dir / f"kpis_{timestamp}.json"
            export_file.write_text(
                json.dumps([kpi.to_dict() for kpi in kpis], indent=2),
                encoding="utf-8"
            )

        self.logger.info(f"Exported {len(kpis)} KPIs to: {export_file}")
        return str(export_file)

    def _detect_category(self, text: str, source_kind: str) -> Optional[str]:
        """Detect KPI category from text."""
        text_lower = text.lower()

        for category, patterns in self.KPI_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return category

        # Fallback based on source kind
        source_category_map = {
            "filing": "funding",
            "pr_wire": "funding",
            "investor_portfolio": "funding"
        }

        return source_category_map.get(source_kind)

    def _extract_company(self, title: str, text: str) -> str:
        """Extract company name from text."""
        combined = f"{title} {text}"

        # Try known patterns
        for pattern in self.COMPANY_PATTERNS:
            match = re.search(pattern, combined)
            if match:
                return match.group(0)

        # Fallback: first capitalized word sequence
        match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', title)
        if match:
            return match.group(1)

        return "Unknown"

    def _extract_metric_value(
        self,
        text: str,
        category: str
    ) -> tuple:
        """Extract metric type and value from text."""
        if category == "funding":
            # Look for dollar amounts
            match = re.search(r'\$[\d.]+\s*(?:million|billion|M|B)', text, re.I)
            if match:
                return "Funding Amount", match.group(0)
            return "Funding Round", "N/A"

        elif category == "acquisition":
            match = re.search(r'acquir(?:ed|es)\s+(\w+)', text, re.I)
            if match:
                return "Acquisition", match.group(1)
            return "M&A Activity", "N/A"

        elif category == "leadership":
            match = re.search(r'(CEO|CTO|CFO|COO|founder)', text, re.I)
            if match:
                return "Executive", match.group(1)
            return "Leadership Change", "N/A"

        elif category == "product":
            return "Product Update", text[:50] + "..."

        return category.title(), "N/A"

    def _load_latest_response(self) -> Optional[Dict[str, Any]]:
        """Load the most recent response file."""
        response_files = sorted(RESPONSES_DIR.glob("response_*.json"), reverse=True)

        if response_files:
            latest = response_files[0]
            try:
                return json.loads(latest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        return None

    def _save_visualization_data(self, data: Dict[str, Any]) -> Path:
        """Save visualization data for dashboard."""
        viz_file = RESPONSES_DIR / "latest_visualization.json"
        viz_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return viz_file

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = super().get_status()
        status["streamlit_port"] = self.streamlit_port
        status["dashboard_running"] = self._streamlit_process is not None
        status["responses_available"] = len(list(RESPONSES_DIR.glob("response_*.json")))
        return status


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Presentation Agent")
    parser.add_argument("--launch", action="store_true", help="Launch dashboard")
    parser.add_argument("--extract", action="store_true", help="Extract KPIs from latest")
    parser.add_argument("--export", choices=["json", "csv"], help="Export KPIs")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    agent = PresentationAgent(streamlit_port=args.port)

    if args.status:
        print(json.dumps(agent.get_status(), indent=2))
    elif args.extract:
        result = agent.run(launch_dashboard=False)
        print(json.dumps(result, indent=2))
    elif args.export:
        response = agent._load_latest_response()
        if response:
            kpis = agent.extract_kpis(response)
            export_path = agent.export_data(kpis, format=args.export)
            print(f"Exported to: {export_path}")
        else:
            print("No response available")
    elif args.launch:
        result = agent.run(launch_dashboard=True)
        print(f"Dashboard URL: {result.get('dashboard_url')}")
    else:
        parser.print_help()
