"""Simple responder agent: formats or post-processes planner output.

This is a minimal implementation so the module exists and can be extended.
"""
from typing import Any, Dict

def format_response(planner_output: Dict[str, Any]) -> str:
    """Return a short, human-friendly string summary of the planner output.

    Keep this lightweight — planner already returns a polished `response` field.
    """
    if not planner_output:
        return "No response available."

    # If planner returns nested dict with 'response' key, return that first
    resp = planner_output.get("response") if isinstance(planner_output, dict) else planner_output
    if isinstance(resp, str) and resp.strip():
        return resp.strip()

    # Fallback: stringify minimal metadata
    company = planner_output.get("company", "unknown")
    intent = planner_output.get("intent", "general")
    confidence = planner_output.get("confidence", 0.0)

    return f"Result for {company} ({intent}) — confidence: {confidence:.2f}."
