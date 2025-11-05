# backend/services/data_fetcher.py
"""
Data Fetcher — retrieves structured financial data like stock fundamentals or fund info.
"""

import random

def get_financial_data(query: str):
    """
    Mock structured data fetcher — replace with real API integration later.
    """
    fake_data = {
        "symbol": query.upper(),
        "current_price": round(random.uniform(100, 1500), 2),
        "pe_ratio": round(random.uniform(5, 40), 2),
        "market_cap": f"{round(random.uniform(1, 500), 2)}B",
        "sector": "Technology",
        "52_week_high": round(random.uniform(500, 2000), 2),
        "52_week_low": round(random.uniform(50, 800), 2),
    }

    return fake_data
