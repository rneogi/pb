"""
Intelligent KPI Extractor
=========================
Uses LLM-generated insights to extract structured KPIs for visualization.

Features:
    - Parses LLM answer for structured data points
    - Extracts funding amounts, percentages, dates
    - Identifies trends and comparisons
    - Suggests optimal visualization type
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ChartType(Enum):
    """Supported visualization types."""
    MATRIX = "matrix"           # Company x Category grid
    TIMELINE = "timeline"       # Time-series funding events
    FUNNEL = "funnel"          # Deal stages pipeline
    TREEMAP = "treemap"        # Hierarchical funding by sector
    NETWORK = "network"        # Investor-company relationships
    GAUGE = "gauge"            # Confidence/sentiment meters
    BAR_RACE = "bar_race"      # Animated ranking over time
    SANKEY = "sankey"          # Money flow visualization
    BUBBLE = "bubble"          # Multi-dimensional comparison
    RADAR = "radar"            # Company profile comparison


@dataclass
class ExtractedKPI:
    """Rich KPI with visualization metadata."""
    id: str
    category: str
    entity: str
    metric_name: str
    metric_value: Any
    metric_unit: str = ""
    change_direction: str = ""  # up, down, stable
    change_percent: Optional[float] = None
    date: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    related_entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "entity": self.entity,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_unit": self.metric_unit,
            "change_direction": self.change_direction,
            "change_percent": self.change_percent,
            "date": self.date,
            "source": self.source,
            "confidence": self.confidence,
            "related_entities": self.related_entities,
            "tags": self.tags
        }


@dataclass
class VisualizationSpec:
    """Specification for a visualization."""
    chart_type: ChartType
    title: str
    data: List[Dict[str, Any]]
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more prominent


class IntelligentKPIExtractor:
    """
    Extracts structured KPIs from LLM responses with intelligent parsing.

    Uses pattern matching and semantic analysis to identify:
    - Funding amounts and rounds
    - Valuations and multiples
    - Growth percentages
    - Time periods and trends
    - Entity relationships
    """

    # Currency patterns
    CURRENCY_PATTERNS = [
        r'\$(\d+(?:\.\d+)?)\s*(billion|million|B|M|bn|mn|k|K)',
        r'(\d+(?:\.\d+)?)\s*(billion|million|B|M|bn|mn)\s*(?:dollars?|USD)',
        r'USD\s*(\d+(?:\.\d+)?)\s*(billion|million|B|M|bn|mn)?',
    ]

    # Percentage patterns
    PERCENT_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*percent',
        r'(up|down|grew|declined|increased|decreased)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%?',
    ]

    # Funding round patterns
    ROUND_PATTERNS = [
        r'series\s+([A-H])',
        r'seed\s+(?:round|funding)',
        r'pre-?seed',
        r'Series\s+([A-H])\d*',
        r'growth\s+(?:round|equity)',
        r'IPO',
        r'SPAC',
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'(Q[1-4])\s*(\d{4})',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        r'(\d{4})',
        r'(last\s+(?:week|month|quarter|year))',
        r'(this\s+(?:week|month|quarter|year))',
    ]

    # Company indicators
    COMPANY_INDICATORS = [
        'Inc', 'Corp', 'LLC', 'Ltd', 'Co', 'Technologies',
        'Labs', 'AI', 'Tech', 'Software', 'Health', 'Bio'
    ]

    def __init__(self):
        self.kpi_counter = 0

    def extract_from_response(
        self,
        response: Dict[str, Any]
    ) -> Tuple[List[ExtractedKPI], List[VisualizationSpec]]:
        """
        Extract KPIs and generate visualization specs from a response.

        Args:
            response: Chat response with answer and citations

        Returns:
            Tuple of (KPIs list, Visualization specs list)
        """
        kpis = []

        # Extract from LLM answer
        answer = response.get("answer", "")
        answer_kpis = self._extract_from_text(answer, source="llm_answer", confidence=0.9)
        kpis.extend(answer_kpis)

        # Extract from citations
        for citation in response.get("citations", []):
            title = citation.get("title", "")
            snippet = citation.get("snippet", "")
            text = f"{title}. {snippet}"
            source = citation.get("source_name", "citation")
            score = citation.get("score", 0.5)

            citation_kpis = self._extract_from_text(
                text,
                source=source,
                confidence=score if isinstance(score, float) else 0.5
            )
            kpis.extend(citation_kpis)

        # Deduplicate KPIs
        kpis = self._deduplicate_kpis(kpis)

        # Generate visualization specs
        viz_specs = self._generate_visualizations(kpis, response)

        return kpis, viz_specs

    def _extract_from_text(
        self,
        text: str,
        source: str = "",
        confidence: float = 0.5
    ) -> List[ExtractedKPI]:
        """Extract KPIs from a text block."""
        kpis = []

        # Split into sentences for context
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue

            # Extract funding KPIs
            funding_kpis = self._extract_funding(sentence, source, confidence)
            kpis.extend(funding_kpis)

            # Extract percentage/growth KPIs
            growth_kpis = self._extract_growth(sentence, source, confidence)
            kpis.extend(growth_kpis)

            # Extract valuation KPIs
            valuation_kpis = self._extract_valuation(sentence, source, confidence)
            kpis.extend(valuation_kpis)

        return kpis

    def _extract_funding(
        self,
        text: str,
        source: str,
        confidence: float
    ) -> List[ExtractedKPI]:
        """Extract funding-related KPIs."""
        kpis = []

        # Look for funding amounts
        for pattern in self.CURRENCY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = float(match.group(1))
                unit = match.group(2).upper() if len(match.groups()) > 1 else ""

                # Normalize to millions
                multiplier = 1
                if unit in ('B', 'BILLION', 'BN'):
                    multiplier = 1000
                    unit = "B"
                elif unit in ('M', 'MILLION', 'MN'):
                    unit = "M"
                elif unit in ('K', 'K'):
                    multiplier = 0.001
                    unit = "K"

                # Find associated company
                entity = self._find_entity(text)

                # Find funding round
                round_match = None
                for rp in self.ROUND_PATTERNS:
                    rm = re.search(rp, text, re.IGNORECASE)
                    if rm:
                        round_match = rm.group(0)
                        break

                # Determine metric name
                if round_match:
                    metric_name = f"{round_match} Funding"
                elif 'raised' in text.lower():
                    metric_name = "Funding Raised"
                else:
                    metric_name = "Investment"

                kpi = ExtractedKPI(
                    id=f"funding_{self._next_id()}",
                    category="funding",
                    entity=entity,
                    metric_name=metric_name,
                    metric_value=amount * multiplier,
                    metric_unit=f"${unit}",
                    date=self._extract_date(text),
                    source=source,
                    confidence=confidence,
                    tags=["funding", round_match.lower() if round_match else "investment"]
                )
                kpis.append(kpi)

        return kpis

    def _extract_growth(
        self,
        text: str,
        source: str,
        confidence: float
    ) -> List[ExtractedKPI]:
        """Extract growth/percentage KPIs."""
        kpis = []

        # Look for percentage changes
        for pattern in self.PERCENT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()

                if len(groups) == 1:
                    # Simple percentage
                    pct = float(groups[0])
                    direction = "up" if any(w in text.lower() for w in ['grew', 'increased', 'up', 'rise']) else "down" if any(w in text.lower() for w in ['declined', 'decreased', 'down', 'fell']) else "stable"
                else:
                    # Direction + percentage
                    direction = "up" if groups[0].lower() in ['up', 'grew', 'increased'] else "down"
                    pct = float(groups[1])

                entity = self._find_entity(text)

                # Determine what metric is changing
                metric_name = "Growth Rate"
                if 'revenue' in text.lower():
                    metric_name = "Revenue Growth"
                elif 'headcount' in text.lower() or 'employee' in text.lower():
                    metric_name = "Headcount Change"
                elif 'valuation' in text.lower():
                    metric_name = "Valuation Change"
                elif 'market' in text.lower():
                    metric_name = "Market Share"

                kpi = ExtractedKPI(
                    id=f"growth_{self._next_id()}",
                    category="growth",
                    entity=entity,
                    metric_name=metric_name,
                    metric_value=pct,
                    metric_unit="%",
                    change_direction=direction,
                    change_percent=pct if direction == "up" else -pct,
                    date=self._extract_date(text),
                    source=source,
                    confidence=confidence,
                    tags=["growth", direction]
                )
                kpis.append(kpi)

        return kpis

    def _extract_valuation(
        self,
        text: str,
        source: str,
        confidence: float
    ) -> List[ExtractedKPI]:
        """Extract valuation KPIs."""
        kpis = []

        if 'valuation' in text.lower() or 'valued at' in text.lower():
            for pattern in self.CURRENCY_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount = float(match.group(1))
                    unit = match.group(2).upper() if len(match.groups()) > 1 else "M"

                    if unit in ('B', 'BILLION', 'BN'):
                        unit = "B"
                    else:
                        unit = "M"

                    entity = self._find_entity(text)

                    kpi = ExtractedKPI(
                        id=f"valuation_{self._next_id()}",
                        category="valuation",
                        entity=entity,
                        metric_name="Valuation",
                        metric_value=amount,
                        metric_unit=f"${unit}",
                        date=self._extract_date(text),
                        source=source,
                        confidence=confidence,
                        tags=["valuation", "unicorn" if (unit == "B" and amount >= 1) or (unit == "M" and amount >= 1000) else "startup"]
                    )
                    kpis.append(kpi)
                    break

        return kpis

    def _find_entity(self, text: str) -> str:
        """Find company/entity name in text."""
        # Look for capitalized sequences with company indicators
        for indicator in self.COMPANY_INDICATORS:
            pattern = rf'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*{indicator}'
            match = re.search(pattern, text)
            if match:
                return match.group(1) + " " + indicator

        # Look for quoted names
        quote_match = re.search(r'"([^"]+)"', text)
        if quote_match:
            return quote_match.group(1)

        # Look for capitalized proper nouns
        proper_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text)
        if proper_match and proper_match.group(1) not in ['The', 'This', 'That', 'These', 'Series', 'Based']:
            return proper_match.group(1)

        return "Unknown Company"

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text."""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _deduplicate_kpis(self, kpis: List[ExtractedKPI]) -> List[ExtractedKPI]:
        """Remove duplicate KPIs, keeping highest confidence."""
        seen = {}
        for kpi in kpis:
            key = (kpi.entity, kpi.metric_name, str(kpi.metric_value))
            if key not in seen or kpi.confidence > seen[key].confidence:
                seen[key] = kpi
        return list(seen.values())

    def _next_id(self) -> int:
        """Generate unique ID."""
        self.kpi_counter += 1
        return self.kpi_counter

    def _generate_visualizations(
        self,
        kpis: List[ExtractedKPI],
        response: Dict[str, Any]
    ) -> List[VisualizationSpec]:
        """Generate visualization specifications based on KPI types."""
        specs = []

        # Group KPIs by category
        by_category = {}
        for kpi in kpis:
            cat = kpi.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(kpi)

        # 1. Funding Timeline (if funding KPIs exist)
        if "funding" in by_category and len(by_category["funding"]) >= 1:
            funding_data = [
                {
                    "company": k.entity,
                    "amount": k.metric_value,
                    "unit": k.metric_unit,
                    "round": k.metric_name,
                    "date": k.date or "Recent"
                }
                for k in by_category["funding"]
            ]
            specs.append(VisualizationSpec(
                chart_type=ChartType.TIMELINE,
                title="Funding Timeline",
                data=funding_data,
                config={"y_axis": "amount", "color_by": "round"},
                priority=10
            ))

            # Also add bubble chart for funding comparison
            if len(funding_data) >= 2:
                specs.append(VisualizationSpec(
                    chart_type=ChartType.BUBBLE,
                    title="Funding Comparison",
                    data=funding_data,
                    config={"size_by": "amount", "color_by": "company"},
                    priority=8
                ))

        # 2. Growth Gauges (if growth KPIs exist)
        if "growth" in by_category:
            growth_data = [
                {
                    "company": k.entity,
                    "metric": k.metric_name,
                    "value": k.metric_value,
                    "direction": k.change_direction
                }
                for k in by_category["growth"]
            ]
            specs.append(VisualizationSpec(
                chart_type=ChartType.GAUGE,
                title="Growth Indicators",
                data=growth_data,
                config={"max_value": 100, "thresholds": [25, 50, 75]},
                priority=7
            ))

        # 3. Valuation Treemap
        if "valuation" in by_category and len(by_category["valuation"]) >= 2:
            val_data = [
                {
                    "company": k.entity,
                    "valuation": k.metric_value,
                    "unit": k.metric_unit
                }
                for k in by_category["valuation"]
            ]
            specs.append(VisualizationSpec(
                chart_type=ChartType.TREEMAP,
                title="Valuation Landscape",
                data=val_data,
                config={"size_by": "valuation"},
                priority=9
            ))

        # 4. Overall KPI Matrix (always include)
        matrix_data = [kpi.to_dict() for kpi in kpis]
        specs.append(VisualizationSpec(
            chart_type=ChartType.MATRIX,
            title="KPI Overview Matrix",
            data=matrix_data,
            config={"rows": "entity", "cols": "category"},
            priority=5
        ))

        # 5. Confidence Gauge for overall response
        confidence_map = {
            "high": 85,
            "medium": 60,
            "low": 30
        }
        conf_label = response.get("confidence_label", "medium")
        specs.append(VisualizationSpec(
            chart_type=ChartType.GAUGE,
            title="Response Confidence",
            data=[{
                "metric": "Confidence",
                "value": confidence_map.get(conf_label, 50),
                "label": conf_label.upper()
            }],
            config={"max_value": 100, "color_ranges": ["#ff4444", "#ffaa00", "#44ff44"]},
            priority=6
        ))

        # Sort by priority
        specs.sort(key=lambda x: x.priority, reverse=True)

        return specs


def extract_kpis_with_intelligence(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to extract KPIs with full intelligence.

    Returns:
        Dict with kpis, visualizations, and summary
    """
    extractor = IntelligentKPIExtractor()
    kpis, viz_specs = extractor.extract_from_response(response)

    return {
        "kpis": [k.to_dict() for k in kpis],
        "visualizations": [
            {
                "type": v.chart_type.value,
                "title": v.title,
                "data": v.data,
                "config": v.config,
                "priority": v.priority
            }
            for v in viz_specs
        ],
        "summary": {
            "total_kpis": len(kpis),
            "categories": list(set(k.category for k in kpis)),
            "entities": list(set(k.entity for k in kpis)),
            "chart_types": [v.chart_type.value for v in viz_specs]
        }
    }
