"""Comparison Agent — Compare multiple stocks or asset classes.

This agent is useful for:
- Comparing a set of stocks within a sector or theme
- Providing peer benchmarking
- Handling topical queries like "best companies to invest in"
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.agents.data_agent import data_agent
from backend.agents.news_agent import news_agent


class ComparisonAgent:
    """Fetches and compares financial metrics across multiple stocks."""

    # Curated lists of representative Indian stocks by sector/theme
    SECTOR_SAMPLES = {
        "technology": ["TCS", "INFY", "WIPRO"],
        "banking": ["HDFCBANK", "ICICIBANK", "SBIN"],
        "automobiles": ["TATAMOTORS", "MARUTI", "BAJAJFINSV"],
        "energy": ["RELIANCE", "NTPC", "ONGC"],
        "fmcg": ["NESTLEIND", "BRITANNIA", "ITC"],
        "pharmaceuticals": ["SUNPHARMA", "DIVI", "CIPLA"],
        "real_estate": ["DLF", "OBEROI", "PRESTIGE"],
        "utilities": ["POWER", "NTPC", "ONGC"],
    }

    # Default sample if no sector detected
    DEFAULT_SAMPLES = ["TCS", "INFY", "HDFCBANK", "RELIANCE", "MARUTI"]

    def __init__(self):
        self.name = "Comparison Agent"

    def _select_sample_stocks(self, query: str) -> List[str]:
        """Select representative stocks based on query keywords."""
        query_lower = query.lower()
        # Detect explicit tickers or known stock symbols mentioned in the query.
        # If the user explicitly mentions a ticker/company (e.g., "TCS", "INFY"),
        # prefer returning that ticker first so single-company queries are handled.
        token_candidates = re.findall(r"\b[A-Za-z]{2,8}\b", query)
        pool = set(self.DEFAULT_SAMPLES)
        for stocks in self.SECTOR_SAMPLES.values():
            pool.update(stocks)

        for tok in token_candidates:
            up = tok.upper()
            if up in pool:
                # Return the explicit ticker first, then fill with default samples
                rest = [s for s in self.DEFAULT_SAMPLES if s != up]
                return [up] + rest

        # Try to match sector keywords
        for sector, stocks in self.SECTOR_SAMPLES.items():
            if sector in query_lower or sector.replace("_", " ") in query_lower:
                return stocks

        # Check for specific terms
        if any(word in query_lower for word in ["tech", "software", "it"]):
            return self.SECTOR_SAMPLES["technology"]
        if any(word in query_lower for word in ["bank", "financial"]):
            return self.SECTOR_SAMPLES["banking"]
        if any(word in query_lower for word in ["auto", "car", "vehicle"]):
            return self.SECTOR_SAMPLES["automobiles"]
        if any(word in query_lower for word in ["energy", "oil", "gas"]):
            return self.SECTOR_SAMPLES["energy"]
        if any(word in query_lower for word in ["fmcg", "consumer", "food"]):
            return self.SECTOR_SAMPLES["fmcg"]
        if any(word in query_lower for word in ["pharma", "health", "drug"]):
            return self.SECTOR_SAMPLES["pharmaceuticals"]

        # Default: return a balanced cross-sector sample
        return self.DEFAULT_SAMPLES

    async def compare_stocks(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Main entry point: compare a set of representative stocks."""
        try:
            print(f"[ComparisonAgent] Query: {query}")

            # Select representative stocks
            stocks_to_compare = self._select_sample_stocks(query)[:limit]
            print(f"[ComparisonAgent] Selected stocks: {stocks_to_compare}")

            # Fetch data and news in parallel for each stock
            tasks = [
                self._fetch_stock_profile(ticker) for ticker in stocks_to_compare
            ]
            profiles = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors
            valid_profiles = [p for p in profiles if not isinstance(p, Exception)]

            # Build comparison summary
            comparison_data = {
                "query": query,
                "stocks_compared": stocks_to_compare,
                "count": len(valid_profiles),
                "profiles": valid_profiles,
                "comparison_summary": self._build_comparison_summary(valid_profiles),
                "timestamp": datetime.utcnow().isoformat(),
            }

            print(f"[ComparisonAgent] Comparison complete. {len(valid_profiles)} stocks profiled.")
            return comparison_data

        except Exception as e:
            print(f"[ComparisonAgent] Error: {e}")
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _fetch_stock_profile(self, ticker: str) -> Dict[str, Any]:
        """Fetch financial data and recent news for a single stock."""
        try:
            # Parallel fetch: data + news
            data_task = data_agent.get_financial_data(ticker)
            news_task = news_agent.get_company_news(ticker)

            data, news = await asyncio.gather(data_task, news_task)

            return {
                "ticker": ticker,
                "company_name": data.get("company_name", ticker),
                "sector": data.get("sector", "N/A"),
                "market_cap": data.get("market_cap"),
                "pe_ratio": data.get("pe_ratio"),
                "dividend_yield": data.get("dividend_yield"),
                "52_week_high": data.get("52_week_high"),
                "52_week_low": data.get("52_week_low"),
                "roe": data.get("roe"),
                "debt_to_equity": data.get("debt_to_equity"),
                "recent_news_count": news.get("count", 0),
                "recent_news_sources": news.get("top_sources", [])[:3],
            }
        except Exception as e:
            print(f"[ComparisonAgent] Error fetching {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def _build_comparison_summary(self, profiles: List[Dict[str, Any]]) -> str:
        """Generate a text summary comparing the stocks."""
        if not profiles:
            return "Unable to fetch comparison data."

        # Sort by P/E ratio (lower often indicates better value)
        valid_pe = [p for p in profiles if isinstance(p.get("pe_ratio"), (int, float))]
        sorted_by_pe = sorted(valid_pe, key=lambda x: x["pe_ratio"])

        summary_lines = [
            f"Compared {len(profiles)} stocks across sectors.",
            ""
        ]

        # Highlight top performer by PE
        if sorted_by_pe:
            best = sorted_by_pe[0]
            summary_lines.append(
                f"Best valuation (lowest P/E): {best['ticker']} — "
                f"P/E {best['pe_ratio']:.2f}, "
                f"Sector: {best['sector']}"
            )

        # Highlight highest dividend yield
        valid_div = [p for p in profiles if isinstance(p.get("dividend_yield"), (int, float))]
        if valid_div:
            best_div = max(valid_div, key=lambda x: x["dividend_yield"])
            summary_lines.append(
                f"Highest dividend yield: {best_div['ticker']} — "
                f"{best_div['dividend_yield']:.2f}%"
            )

        # Highlight highest ROE
        valid_roe = [p for p in profiles if isinstance(p.get("roe"), (int, float))]
        if valid_roe:
            best_roe = max(valid_roe, key=lambda x: x["roe"])
            summary_lines.append(
                f"Best ROE (profitability): {best_roe['ticker']} — {best_roe['roe']:.2f}%"
            )

        summary_lines.extend([
            "",
            "Note: This is a snapshot comparison. Always verify metrics and conduct deeper due diligence before investing."
        ])

        return "\n".join(summary_lines)


comparison_agent = ComparisonAgent()
