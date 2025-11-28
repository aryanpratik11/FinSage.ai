"""Lightweight prediction service stub.

This provides a simple sync function `predict_financial_outcome` used by
`backend/agents/market_agent.py`. For now it returns a placeholder string
or a simple heuristic. Later you can replace this with a real model call.
"""
from typing import Optional

def predict_financial_outcome(query: str, horizon: Optional[str] = None) -> Optional[str]:
    """Return a simple placeholder qualitative prediction.

    Keep this synchronous to match how `market_agent` calls it.
    """
    if not query:
        return None

    # Very basic heuristic placeholder
    return (
        "[Prediction Stub] No numeric model is configured. "
        "Use the Planner's LLM-based fallback for qualitative outlooks."
    )
