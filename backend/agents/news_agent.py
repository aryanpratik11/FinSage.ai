import aiohttp
import asyncio
import feedparser
from newspaper import Article
from datetime import datetime
import re
import uuid
from typing import Dict, Any, List


class NewsAgent:
    """Fetches financial news from multiple RSS sources."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    HEADERS = {"User-Agent": USER_AGENT}

    BASE_FEEDS = [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.reuters.com/rssFeed/businessNews",
    ]

    def __init__(self):
        self.name = "News Agent"

    # -------------------- COMPANY/TICKER EXTRACTION --------------------
    def _extract_company_or_ticker(self, text: str) -> str:
        """Extracts probable company name or stock ticker from user query."""
        if not text:
            return ""

        # Detect uppercase stock-like tokens (TCS, INFY)
        ticker_match = re.search(r"\b[A-Z]{2,6}\b", text)
        if ticker_match:
            return ticker_match.group(0)

        # Clean up phrases
        cleaned = re.sub(r"(?i)\b(should|invest|stock|buy|sell|price|share|in|the|about)\b", "", text)
        cleaned = re.sub(r"[^A-Za-z0-9 ]", "", cleaned).strip()

        # Keep at most 3 words
        words = cleaned.split()[:3]
        return " ".join(words).title()

    # -------------------- COMPANY MATCH FLEXIBILITY --------------------
    def _matches_company(self, text: str, company: str) -> bool:
        """Flexible match for company mentions (handles 'Ltd', spacing, etc.)"""
        text_l, comp_l = text.lower(), company.lower()
        variations = [
            comp_l,
            comp_l.replace(" ", ""),
            f"{comp_l} ltd",
            f"{comp_l} limited",
            f"{comp_l} inc",
            f"{comp_l} corporation",
        ]
        return any(v in text_l for v in variations)

    # -------------------- TICKER MAPPING (for Yahoo Feed) --------------------
    def _infer_ticker(self, company: str) -> str:
        """Very lightweight mapping for known Indian companies."""
        mapping = {
            "tata motors": "TATAMOTORS.NS",
            "tcs": "TCS.NS",
            "infosys": "INFY.NS",
            "reliance": "RELIANCE.NS",
            "hdfc bank": "HDFCBANK.NS",
            "icici bank": "ICICIBANK.NS",
            "wipro": "WIPRO.NS",
            "sbi": "SBIN.NS",
        }
        return mapping.get(company.lower(), "")

    # -------------------- ARTICLE FETCHER --------------------
    async def fetch_full_article(self, url: str) -> str:
        """Fetches full article content via newspaper3k with fallback."""
        try:
            loop = asyncio.get_event_loop()

            def scrape():
                article = Article(url)
                article.download()
                article.parse()
                return article.text.strip()

            content = await asyncio.wait_for(loop.run_in_executor(None, scrape), timeout=10)
            return content
        except Exception:
            # Fallback to simple HTML fetch
            try:
                timeout = aiohttp.ClientTimeout(total=8)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=self.HEADERS) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            return re.sub(r"<[^>]+>", "", text)[:2000]
            except Exception as e:
                print(f"[NewsAgent] Fallback fetch error for {url}: {e}")
        return None

    # -------------------- RSS FETCHER --------------------
    async def fetch_stock_news(self, company: str, limit: int = 10) -> List[Dict]:
        """Fetches and filters news articles mentioning the company."""
        all_news = []
        ticker = self._infer_ticker(company)

        rss_feeds = self.BASE_FEEDS.copy()
        if ticker:
            rss_feeds.insert(0, f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN")

        for feed_url in rss_feeds:
            try:
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
                source_name = feed.feed.get("title", feed_url)

                for entry in feed.entries[:30]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    link = entry.get("link", "")
                    published = entry.get("published", None)

                    # match company
                    if not self._matches_company(title + " " + summary, company):
                        continue

                    try:
                        content = await asyncio.wait_for(self.fetch_full_article(link), timeout=10)
                    except asyncio.TimeoutError:
                        content = summary

                    news_item = {
                        "id": str(uuid.uuid4()),
                        "company": company.upper(),
                        "title": title.strip(),
                        "link": link,
                        "published": published,
                        "source": source_name,
                        "summary": summary.strip(),
                        "content": content or summary,
                        "fetched_at": datetime.utcnow().isoformat(),
                    }

                    all_news.append(news_item)
                    if len(all_news) >= limit:
                        return all_news

                    await asyncio.sleep(0.3)

            except Exception as e:
                print(f"[NewsAgent] RSS feed error: {feed_url} — {e}")
                continue

        return all_news

    # -------------------- MAIN ENTRYPOINT --------------------
    async def get_company_news(self, query: str) -> Dict[str, Any]:
        """Main entrypoint — cleans query, fetches, filters, and summarizes news."""
        try:
            print(f"[NewsAgent] Query received: '{query}'")
            company_or_ticker = self._extract_company_or_ticker(query)
            print(f"[NewsAgent] Extracted company/ticker: '{company_or_ticker}'")

            results = await self.fetch_stock_news(company_or_ticker, limit=10)
            print(f"[NewsAgent] Found {len(results)} matching news items for '{company_or_ticker}'")

            if not results:
                return {
                    "news": [],
                    "summary": f"No recent financial news found for '{company_or_ticker}'.",
                    "count": 0,
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # build summary
            top_titles = [r["title"] for r in results[:5]]
            summary = (
                f"Found {len(results)} recent news articles about '{company_or_ticker}'.\n\n"
                f"Top headlines:\n" + "\n".join([f"• {t}" for t in top_titles])
            )

            return {
                "news": results,
                "summary": summary,
                "count": len(results),
                "top_sources": list(set([r["source"] for r in results])),
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            print(f"[NewsAgent] Error while fetching news: {e}")
            return {
                "news": [],
                "summary": f"Error while fetching news for '{query}': {e}",
                "count": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

news_agent = NewsAgent()