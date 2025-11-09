import feedparser
import requests
from newspaper import Article
import json
import uuid
from datetime import datetime
import time
import sys

# -----------------------------
# ⚙️ CONFIG
# -----------------------------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
HEADERS = {"User-Agent": USER_AGENT}

# RSS feeds of major finance news sites
RSS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.reuters.com/rssFeed/businessNews",
]


# -----------------------------
# 📰 Fetch full article content
# -----------------------------
def fetch_full_article(url):
    """Fetch full article content using newspaper3k, fallback to raw HTML text."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        # fallback if article parsing fails
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                text = res.text
                text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<p>", "\n").replace("</p>", "")
                return text[:2000]
            return None
        except Exception:
            return None


# -----------------------------
# 🔎 Fetch and filter news
# -----------------------------
def fetch_stock_news(stock_query: str):
    all_news = []
    for feed_url in RSS_FEEDS:
        print(f"\nFetching from: {feed_url}")
        feed = feedparser.parse(feed_url)
        source_name = feed.feed.get("title", "Unknown Source")

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")
            published = entry.get("published", None)

            # Filter only news containing stock keyword
            text_match = stock_query.lower() in (title + summary).lower()
            if not text_match:
                continue

            print(f" → Found: {title}")
            content = fetch_full_article(link)
            if not content:
                print("    ⚠️ Skipped (no content)")
                continue

            news_item = {
                "id": str(uuid.uuid4()),
                "stock": stock_query.upper(),
                "title": title,
                "link": link,
                "published": published,
                "source": source_name,
                "summary": summary,
                "content": content,
                "fetched_at": datetime.now().isoformat(),
            }
            all_news.append(news_item)
            time.sleep(1)

    return all_news


# -----------------------------
# 💾 Save results
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        stock_query = input("Enter stock name or symbol: ").strip()
    else:
        stock_query = sys.argv[1]

    print(f"\n📰 Fetching news related to '{stock_query}'...\n")
    news_data = fetch_stock_news(stock_query)

    filename = f"news_{stock_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(news_data)} news articles for '{stock_query}' to {filename}")
