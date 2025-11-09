"""# backend/routers/news_router.py

from fastapi import APIRouter, Query
from backend.services.news_fetcher import fetch_latest_news, summarize_news

router = APIRouter(prefix="/news", tags=["News"])

@router.get("/latest")
async def get_latest_news(
    query: str = Query(..., description="Search term like 'HDFC Bank' or 'Nifty 50'")
):

    try:
        news_articles = fetch_latest_news(query)
        summaries = [summarize_news(article) for article in news_articles]
        return {
            "query": query,
            "count": len(news_articles),
            "results": summaries
        }
    except Exception as e:
        return {"error": str(e)}
"""