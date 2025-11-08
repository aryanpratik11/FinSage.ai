"""ReasoningAgent: synthesize findings into a concise investment thesis."""
from typing import Dict, Any


class ReasoningAgent:
    name = "ReasoningAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        fundamentals = context.get("fundamentals", {})
        news = context.get("news", {})
        calc = context.get("metrics", {})
        forecast = context.get("prediction", {})

        parts = []
        name = fundamentals.get("shortName") or ticker
        parts.append(f"{name} ({ticker}):")

        pe = calc.get("pe_calculated")
        if pe:
            parts.append(f"PE ~ {pe:.1f}")

        if news and news.get("sentiment"):
            parts.append(f"News sentiment: {news.get('sentiment')}")

        if forecast and not forecast.get("error"):
            parts.append(f"Short-term forecast next close: {forecast.get('forecast_next_close'):.2f}")

        thesis = "; ".join(parts)
        return {"ticker": ticker, "thesis": thesis}
