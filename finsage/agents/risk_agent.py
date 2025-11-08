"""RiskAgent: lightweight risk identification and scoring."""
from typing import Dict, Any


class RiskAgent:
    name = "RiskAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        fundamentals = context.get("fundamentals", {})
        news = context.get("news", {})
        score = 50
        notes = []

        # Market cap: small caps get higher risk
        mcap = fundamentals.get("marketCap") or 0
        if mcap and mcap < 2_000_000_000:
            score += 20
            notes.append("small market cap")

        # Negative news lowers score
        if news and news.get("sentiment") == "negative":
            score += 20
            notes.append("recent negative news")

        score = max(0, min(100, score))
        return {"ticker": ticker, "risk_score": score, "notes": notes}
