# backend/services/news_fetcher.py
"""
News Fetcher — retrieves recent financial or business news.
"""

import datetime

def get_recent_news(query: str):
    """
    Mock financial news fetcher. Replace later with real API calls.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    sample_news = [
        {
            "headline": f"{query} reports strong quarterly earnings",
            "date": today,
            "summary": f"The company behind {query} has shown promising growth with improved revenue margins.",
            "source": "Financial Times"
        },
        {
            "headline": f"Market sentiment on {query} remains optimistic",
            "date": today,
            "summary": f"Investors continue to express confidence in {query}'s long-term potential.",
            "source": "Bloomberg"
        }
    ]
    
    return sample_news
