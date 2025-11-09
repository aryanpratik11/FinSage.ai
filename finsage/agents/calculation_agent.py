"""CalculationAgent: compute common financial ratios and a simple DCF stub."""
from typing import Dict, Any
import math


class CalculationAgent:
    name = "CalculationAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        fundamentals = context.get("fundamentals", {})
        history = context.get("history", [])

        price = fundamentals.get("regularMarketPrice") or (history[-1]["close"] if history else None)
        eps = fundamentals.get("epsTrailingTwelveMonths")
        pe = None
        if price is not None and eps:
            try:
                pe = float(price) / float(eps)
            except Exception:
                pe = None

        # Simple DCF stub: assume last year EPS * (1+g) / (r - g) as a perpetuity of earnings
        dcf = None
        try:
            g = 0.05  # growth assumption
            r = 0.10  # discount rate
            if eps and r > g:
                dcf = (eps * (1 + g)) / (r - g)
        except Exception:
            dcf = None

        return {
            "ticker": ticker,
            "pe_calculated": pe,
            "eps": eps,
            "dcf_per_share": dcf,
        }
