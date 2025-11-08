"""
Planner Agent — decides what steps to take based on the user query using an LLM.
"""

from backend.services.llm_service import generate_response


def generate_plan(query: str) -> dict:
    """
    Generates a structured investment reasoning plan using an LLM.
    Returns a dict with a clear sequence of reasoning steps.
    """

    prompt = f"""
    You are a financial planning assistant helping to decide whether to invest in a financial instrument.
    The user asked: "{query}"

    Your goal:
    - Think step-by-step.
    - Identify what information is needed.
    - Plan how to evaluate the opportunity.
    - Return your plan as a structured JSON with keys: "objective", "steps", and "expected_outcome".

    Example output format:
    {{
        "objective": "Evaluate investment potential of HDFC Mutual Fund",
        "steps": [
            "Retrieve recent NAV trends and performance metrics",
            "Analyze fund portfolio composition and top holdings",
            "Compare against similar funds in the same category",
            "Assess risk indicators such as volatility and beta",
            "Summarize suitability based on investor risk appetite"
        ],
        "expected_outcome": "A summarized investment recommendation with risk-benefit analysis"
    }}

    Now, generate the plan for the user query above.
    """

    response_text = generate_response(prompt)

    # The LLM might return plain text or JSON-like text
    # Ensure it returns a clean dict
    try:
        import json
        plan = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback if the model returned a non-JSON text
        plan = {"objective": "Investment Plan", "steps": [response_text], "expected_outcome": "Plan generated in text form"}

    return plan
