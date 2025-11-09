"""Agent interfaces and simple mock implementations for FinSage MVP."""
from dataclasses import dataclass
from typing import Any, Dict, List


class BaseAgent:
    name: str

    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


@dataclass
class DataAgent(BaseAgent):
    name: str = "DataAgent"

    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Mocked structured data
        print(f"[DataAgent] fetching structured data for: {query}")
        return {
            "source": "mock",
            "data": {
                "ticker": "TSLA",
                "price": 250.0,
                "pe_ratio": 65.3,
                "market_cap": 800_000_000_000,
            },
        }


@dataclass
class DocumentAgent(BaseAgent):
    name: str = "DocumentAgent"

    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[DocumentAgent] extracting documents for: {query}")
        return {
            "source": "mock_docs",
            "excerpts": [
                {"doc": "10-Q Q3", "page": 12, "text": "Revenue grew 30% YoY."}
            ],
        }


@dataclass
class CalculationAgent(BaseAgent):
    name: str = "CalculationAgent"

    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[CalculationAgent] computing metrics for: {query}")
        data = context.get("data", {})
        price = data.get("price", 0)
        eps = data.get("eps", 3.84)  # mocked
        pe = price / eps if eps else None
        return {"pe_calculated": pe, "eps": eps}


@dataclass
class ReasoningAgent(BaseAgent):
    name: str = "ReasoningAgent"

    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[ReasoningAgent] analyzing: {query}")
        # Mock reasoning
        return {"thesis": "Company shows strong top-line growth; valuation rich vs peers."}
