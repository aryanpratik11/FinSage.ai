"""ComparisonAgent: compare a ticker to a list of peers using DataAgent outputs."""
from typing import Dict, Any, List


class ComparisonAgent:
    name = "ComparisonAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        peers = context.get("peers", [])
        base = context.get("fundamentals", {})
        comparisons: List[Dict[str, Any]] = []
        for p in peers:
            # Each peer entry expected to contain fundamentals from DataAgent
            f = p.get("fundamentals", {})
            comparisons.append({"ticker": p.get("ticker"), "pe": f.get("trailingPE"), "marketCap": f.get("marketCap")})

        return {"ticker": ticker, "comparisons": comparisons}
