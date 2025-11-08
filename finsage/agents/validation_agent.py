"""ValidationAgent: cross-validate key data points and flag discrepancies."""
from typing import Dict, Any


class ValidationAgent:
    name = "ValidationAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        fundamentals = context.get("fundamentals", {})
        history = context.get("history", [])
        issues = []

        if not fundamentals:
            issues.append("missing fundamentals")

        if not history:
            issues.append("missing price history")

        # Check PE consistency if both available
        pe_reported = fundamentals.get("trailingPE")
        eps = fundamentals.get("epsTrailingTwelveMonths")
        price = fundamentals.get("regularMarketPrice")
        if pe_reported and eps and price:
            try:
                pe_calc = price / eps
                if abs(pe_calc - float(pe_reported)) / float(pe_reported) > 0.2:
                    issues.append("reported PE differs >20% from calculated PE")
            except Exception:
                issues.append("failed to validate PE")

        return {"ticker": ticker, "issues": issues, "ok": len(issues) == 0}
