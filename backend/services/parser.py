# backend/services/parser.py

import re
from typing import Dict, Optional

def extract_ticker_or_name(query: str) -> Dict[str, Optional[str]]:
    """
    Extract potential stock ticker symbols or entity names
    from the user's financial query.
    """

    # Match common ticker patterns like: AAPL, TSLA, HDFC.NS, RELIANCE
    ticker_pattern = re.compile(r"\b[A-Z]{2,6}(?:\.[A-Z]{1,3})?\b")
    tickers = ticker_pattern.findall(query.upper())

    # Filter out false positives
    blacklist = {"ETF", "IPO", "USD", "NSE", "BSE", "BANK"}
    tickers = [t for t in tickers if t not in blacklist]

    # Extract possible company / fund names (capitalized sequences)
    name_pattern = re.compile(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*\b")
    names = name_pattern.findall(query)

    return {
        "ticker": tickers[0] if tickers else None,
        "name": names[0] if names else None,
    }


def detect_intent(query: str) -> str:
    """
    Determine the user's intent from the query.
    This helps the planner_agent decide which sub-agent(s) to trigger.
    """

    q_lower = query.lower()

    if any(word in q_lower for word in ["buy", "invest", "good stock", "worth", "should i", "recommendation"]):
        return "investment_decision"

    elif any(word in q_lower for word in ["price", "today", "quote", "value", "chart", "data", "performance"]):
        return "information_lookup"

    elif any(word in q_lower for word in ["risk", "safe", "danger", "volatile", "exposure"]):
        return "risk_assessment"

    elif any(word in q_lower for word in ["fund", "nav", "mutual", "scheme"]):
        return "mutual_fund_analysis"

    elif any(word in q_lower for word in ["news", "update", "latest", "trend"]):
        return "news_summary"

    elif any(word in q_lower for word in ["predict", "forecast", "future", "expected"]):
        return "prediction_query"

    else:
        return "general_finance_query"


def parse_query(query: str) -> Dict[str, Optional[str]]:
    """
    Complete parsing pipeline:
    Extract entities and detect intent to structure the user's request.
    """

    entity_info = extract_ticker_or_name(query)
    intent = detect_intent(query)

    parsed_output = {
        "query": query,
        "intent": intent,
        "ticker": entity_info.get("ticker"),
        "name": entity_info.get("name"),
    }

    return parsed_output