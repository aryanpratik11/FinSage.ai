"""NewsAgent: fetch recent news for a ticker using Google News RSS and basic sentiment scoring.

Note: sentiment is a simple rule-based score (counts positive/negative words). For production
use a proper NLP model or sentiment API.
"""
from typing import Dict, Any, List
try:
    import feedparser
except Exception:  # pragma: no cover - optional dependency
    feedparser = None

# Minimal sentiment word lists (very small, illustrative)
POS_WORDS = {"beat", "gains", "growth", "positive", "upgrade", "record"}
NEG_WORDS = {"miss", "decline", "downgrade", "lawsuit", "recall", "loss"}


class NewsAgent:
    name = "NewsAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        query = f"{ticker} stock"
        if feedparser is None:
            return {"error": "feedparser not installed. Install with 'pip install feedparser'"}

        url = f"https://news.google.com/rss/search?q={query}"
        feed = feedparser.parse(url)
        items: List[Dict[str, Any]] = []
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = (title + " " + summary).lower()
            score = 0
            for w in POS_WORDS:
                if w in text:
                    score += 1
            for w in NEG_WORDS:
                if w in text:
                    score -= 1
            items.append({"title": title, "link": entry.get("link"), "score": score})

        # Aggregate sentiment
        total = sum(i["score"] for i in items) if items else 0
        sentiment = "neutral"
        if total > 1:
            sentiment = "positive"
        elif total < -1:
            sentiment = "negative"

        return {"ticker": ticker, "source": "google_news_rss", "sentiment": sentiment, "items": items}
