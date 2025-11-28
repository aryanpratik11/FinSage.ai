import json
import re
import asyncio
from typing import Dict, Any
from functools import lru_cache
from datetime import datetime
import aiohttp
import yfinance as yf
import pandas as pd


class DataAgent:
    """Fetches reliable fundamental & financial data for Indian stocks (NSE/BSE)."""
    
    def __init__(self, exchange: str = "NSE"):
        self.exchange = exchange.upper()
        self.name = "Data Agent"

    # ---------------- SYMBOL RESOLUTION ----------------
    @lru_cache(maxsize=100)
    def resolve_symbol(self, query: str) -> str:
        query = query.strip()
        try:
            import requests
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
            res = requests.get(url, timeout=5).json()

            if "quotes" in res and len(res["quotes"]) > 0:
                for q in res["quotes"]:
                    exch = q.get("exchange", "")
                    symbol = q.get("symbol", "")
                    if self.exchange in exch or exch in ["NSI", "BSE"]:
                        return symbol
                return res["quotes"][0].get("symbol", query)
        except Exception as e:
            print(f"Symbol resolution error: {e}")

        cleaned = query.replace(" ", "").upper()
        return f"{cleaned}.NS" if self.exchange == "NSE" else cleaned

    # ---------------- TEXT CLEANER ----------------
    def _extract_company_or_ticker(self, text: str) -> str:
        """
        Extracts a likely company name or ticker from a free-form query.
        Example: 'Should I invest in TCS?' -> 'TCS'
        """
        # If the query explicitly contains indicators of a generic/topical question,
        # return empty to indicate "no specific company/ticker found".
        generic_indicators = [
            r"\bbest\b",
            r"\btop\b",
            r"\bwhich\b",
            r"\bwho\b",
            r"\bshould I\b",
            r"\bwhat are\b",
            r"\bbest companies\b",
            r"\bbest stocks\b",
            r"\binvest in\b",
            r"\bto invest\b",
        ]

        lowered = text.lower() if text else ""
        for pat in generic_indicators:
            if re.search(pat, lowered):
                return ""  # signal no specific company

        # Detect explicit ticker-like uppercase tokens (e.g., TCS, INFY)
        # Use word boundary to avoid single letters like "I"
        ticker_match = re.search(r"\b([A-Z]{2,5})\b", text)
        if ticker_match:
            return ticker_match.group(1)

        cleaned = re.sub(
            r'(stock|share|price|value|of|in|company|financials|the|today|about|performance|should)',
            '',
            text,
            flags=re.IGNORECASE
        ).strip()

        return cleaned or ""

    # ---------------- DF CONVERTER ----------------
    def _convert_df_to_dict(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}
        try:
            df = df.copy()
            df.columns = df.columns.map(str)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime("%Y-%m-%d")
            else:
                df.index = df.index.map(str)
            result = df.to_dict()
            return json.loads(json.dumps(result, default=str))
        except Exception as e:
            print(f"DataFrame conversion error: {e}")
            return {}

    # ---------------- TICKERTAPE API ----------------
    async def _fetch_from_ticker_tape(self, query: str) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"https://api.tickertape.in/search?text={query}"
                async with session.get(search_url, timeout=10) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
                    items = data.get("data", {}).get("items", [])
                    if not items:
                        return {}
                    symbol = items[0]["ticker"]

                info_url = f"https://api.tickertape.in/stocks/info/{symbol}"
                async with session.get(info_url, timeout=10) as resp:
                    if resp.status != 200:
                        return {}
                    info_data = await resp.json()
                    info = info_data.get("data", {})
                    if not info:
                        return {}

                    return {
                        "company_name": info.get("name"),
                        "symbol": symbol,
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "market_cap": info.get("mcap"),
                        "pe_ratio": info.get("peRatio"),
                        "pb_ratio": info.get("pbRatio"),
                        "dividend_yield": info.get("divYield"),
                        "book_value": info.get("bookValue"),
                        "52_week_high": info.get("high52w"),
                        "52_week_low": info.get("low52w"),
                        "debt_to_equity": info.get("debtToEquity"),
                        "roe": info.get("roe"),
                        "currency": "INR",
                        "website": info.get("website"),
                        "logo_url": info.get("logoUrl"),
                    }
        except Exception as e:
            print(f"TickerTape API error: {e}")
            return {}

    # ---------------- GROWW API ----------------
    async def _fetch_from_groww(self, query: str) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"https://groww.in/v1/api/search/v1/query?query={query}&page=0"
                async with session.get(search_url, timeout=10) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
                    results = data.get("data", [])
                    if not results:
                        return {}
                    stock = next((x for x in results if x.get("entityType") == "STOCK"), results[0])
                    symbol = stock.get("symbol")
                    token = stock.get("meta", {}).get("id") or stock.get("nseScriptCode")
                    if not token:
                        return {}

                stock_url = f"https://groww.in/v1/api/stocks_data/v1/companies/{token}/snapshot"
                async with session.get(stock_url, timeout=10) as resp:
                    if resp.status != 200:
                        return {}
                    detail = await resp.json()
                    return {
                        "company_name": stock.get("title"),
                        "symbol": symbol,
                        "market_cap": detail.get("marketCap"),
                        "pe_ratio": detail.get("peRatio"),
                        "book_value": detail.get("bookValue"),
                        "dividend_yield": detail.get("dividendYield"),
                        "52_week_high": detail.get("high52w"),
                        "52_week_low": detail.get("low52w"),
                        "currency": "INR",
                        "logo_url": stock.get("logoUrl"),
                    }
        except Exception as e:
            print(f"Groww API error: {e}")
            return {}

    # ---------------- FETCH FUNDAMENTALS ----------------
    async def fetch_fundamentals(self, query: str) -> dict:
        loop = asyncio.get_event_loop()
        symbol = await loop.run_in_executor(None, self.resolve_symbol, query)
        fundamentals = {
            "query": query,
            "symbol": symbol,
            "fetched_at": datetime.utcnow().isoformat()
        }

        try:
            def fetch_yf():
                stock = yf.Ticker(symbol)
                return stock.info
            info = await loop.run_in_executor(None, fetch_yf)
            if info and "longName" in info:
                print(f"DataAgent: Yahoo Finance data found for '{query}'")
                fundamentals.update({
                    "exchange": info.get("exchange"),
                    "company_name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "eps": info.get("trailingEps"),
                    "dividend_yield": info.get("dividendYield"),
                    "book_value": info.get("bookValue"),
                    "beta": info.get("beta"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "revenue": info.get("totalRevenue"),
                    "profit_margins": info.get("profitMargins"),
                    "net_income": info.get("netIncomeToCommon"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "currency": info.get("currency"),
                    "website": info.get("website"),
                })
        except Exception as e:
            print(f"Yahoo Finance error: {e}")

        if not fundamentals.get("company_name"):
            print(f"DataAgent: No company name from Yahoo Finance, trying TickerTape...")
            tt_data = await self._fetch_from_ticker_tape(query)
            fundamentals.update(tt_data)

        if not fundamentals.get("company_name"):
            print(f"DataAgent: No company name from TickerTape, trying Groww...")
            groww_data = await self._fetch_from_groww(query)
            fundamentals.update(groww_data)

        try:
            def fetch_financials():
                stock = yf.Ticker(symbol)
                return {
                    "income_statement": stock.financials,
                    "balance_sheet": stock.balance_sheet,
                    "cashflow": stock.cashflow,
                }
            fin_data = await loop.run_in_executor(None, fetch_financials)
            fundamentals["financials"] = {
                "income_statement": self._convert_df_to_dict(fin_data["income_statement"]),
                "balance_sheet": self._convert_df_to_dict(fin_data["balance_sheet"]),
                "cashflow": self._convert_df_to_dict(fin_data["cashflow"]),
            }
        except Exception as e:
            print(f"Financial statements error: {e}")
            fundamentals["financials"] = {}

        return fundamentals

    # ---------------- WRAPPER FUNCTION ----------------
    async def get_financial_data(self, query: str) -> Dict[str, Any]:
        """
        Args:
            query: Either a clean company name/ticker (preferred) or a full user query (fallback)
        Returns:
            Dictionary containing financial data and summary
        """
        try:
            print(f"DataAgent: Query/company received: '{query}'")
            company_or_ticker = self._extract_company_or_ticker(query)
            print(f"DataAgent: Extracted company/ticker: '{company_or_ticker}'")

            # If no specific company/ticker detected, avoid treating the whole user
            # query as a symbol. Return an empty/annotated result so planner can
            # choose a different analysis path.
            if not company_or_ticker:
                return {
                    "query": query,
                    "symbol": None,
                    "company_name": None,
                    "summary": "No specific company or ticker detected in the query.",
                    "note": "DataAgent skipped symbol resolution for a general/topic query.",
                }

            data = await self.fetch_fundamentals(company_or_ticker)

            if data.get("company_name"):
                data["summary"] = (
                    f"{data['company_name']} ({data.get('symbol', 'N/A')}) "
                    f"- Sector: {data.get('sector', 'N/A')}, "
                    f"P/E: {data.get('pe_ratio', 'N/A')}, "
                    f"Market Cap: {data.get('market_cap', 'N/A')}"
                )
            else:
                data["summary"] = f"Limited or no financial data available for '{query}'"

            print(f"DataAgent: Summary for '{company_or_ticker}': {data.get('summary', '')}")
            return data

        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "message": f"Failed to fetch financial data for '{query}'"
            }


data_agent = DataAgent()
