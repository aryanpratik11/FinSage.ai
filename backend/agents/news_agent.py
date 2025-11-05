# backend/agents/news_agent.py
"""
News agent — handles financial news queries and summaries.
"""

from backend.services import news_fetcher

def handle_news_query(query: str):
    """
    Retrieves and summarizes financial or company news related to the user's query.
    """
    news_results = news_fetcher.get_recent_news(query)
    
    if not news_results:
        return "No recent financial news found."
    
    return {
        "topic": query,
        "articles": news_results
    }
