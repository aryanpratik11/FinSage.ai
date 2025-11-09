import json
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime


class DataAgent:
    """Fetches reliable fundamental & financial data for Indian stocks (NSE/BSE)."""

    def __init__(self, exchange="NSE"):
        self.exchange = exchange.upper()

    # ---------------- SYMBOL RESOLVER ----------------
    def resolve_symbol(self, query: str) -> str:
        """Resolve the correct stock symbol using Yahoo search API."""
        query = query.strip()
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
            res = requests.get(url, timeout=5).json()
            if "quotes" in res and len(res["quotes"]) > 0:
                for q in res["quotes"]:
                    exch = q.get("exchange", "")
                    if self.exchange in exch or exch in ["NSI", "BSE"]:
                        return q.get("symbol")
                return res["quotes"][0]["symbol"]
        except Exception:
            pass
        return query.replace(" ", "").upper() + ".NS"

    # ---------------- DATAFRAME CONVERTER ----------------
    def _convert_df_to_dict(self, df: pd.DataFrame):
        """Convert pandas DataFrame to JSON-safe dict."""
        if df is None or df.empty:
            return {}
        df = df.copy()
        df.columns = df.columns.map(str)
        df.index = df.index.map(lambda x: str(x) if not isinstance(x, datetime) else x.strftime("%Y-%m-%d"))
        return df.to_dict()

    # ---------------- TICKERTAPE FETCH ----------------
    def _fetch_from_ticker_tape(self, query: str):
        """Fetch fundamental info using TickerTape API."""
        try:
            search_url = f"https://api.tickertape.in/search?text={query}"
            resp = requests.get(search_url, timeout=10).json()
            items = resp.get("data", {}).get("items", [])
            if not items:
                return {}
            symbol = items[0]["ticker"]

            info_url = f"https://api.tickertape.in/stocks/info/{symbol}"
            info_data = requests.get(info_url, timeout=10).json().get("data", {})
            if not info_data:
                return {}

            return {
                "company_name": info_data.get("name"),
                "symbol": symbol,
                "sector": info_data.get("sector"),
                "industry": info_data.get("industry"),
                "market_cap": info_data.get("mcap"),
                "pe_ratio": info_data.get("peRatio"),
                "pb_ratio": info_data.get("pbRatio"),
                "dividend_yield": info_data.get("divYield"),
                "book_value": info_data.get("bookValue"),
                "52_week_high": info_data.get("high52w"),
                "52_week_low": info_data.get("low52w"),
                "debt_to_equity": info_data.get("debtToEquity"),
                "roe": info_data.get("roe"),
                "currency": "INR",
                "website": info_data.get("website"),
                "logo_url": info_data.get("logoUrl"),
            }
        except Exception:
            return {}

    # ---------------- GROWW FALLBACK ----------------
    def _fetch_from_groww(self, query: str):
        """Fallback to Groww public API (more reliable for NSE/BSE)."""
        try:
            search_url = f"https://groww.in/v1/api/search/v1/query?query={query}&page=0"
            resp = requests.get(search_url, timeout=10).json()
            results = resp.get("data", [])
            if not results:
                return {}

            stock = next((x for x in results if x.get("entityType") == "STOCK"), results[0])
            symbol = stock.get("symbol")
            token = stock.get("meta", {}).get("id") or stock.get("nseScriptCode")

            if not token:
                return {}

            stock_url = f"https://groww.in/v1/api/stocks_data/v1/companies/{token}/snapshot"
            detail_resp = requests.get(stock_url, timeout=10).json()

            return {
                "company_name": stock.get("title"),
                "symbol": symbol,
                "market_cap": detail_resp.get("marketCap"),
                "pe_ratio": detail_resp.get("peRatio"),
                "book_value": detail_resp.get("bookValue"),
                "dividend_yield": detail_resp.get("dividendYield"),
                "52_week_high": detail_resp.get("high52w"),
                "52_week_low": detail_resp.get("low52w"),
                "currency": "INR",
                "logo_url": stock.get("logoUrl"),
            }
        except Exception:
            return {}

    # ---------------- MAIN FUNCTION ----------------
    def fetch_fundamentals(self, query: str) -> dict:
        """Fetch all possible stock data from multiple sources."""
        symbol = self.resolve_symbol(query)
        fundamentals = {"query": query, "symbol": symbol}
        stock = yf.Ticker(symbol)

        # 1️⃣ Try Yahoo Finance
        try:
            info = stock.info
            if info and "longName" in info:
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
                    "logo_url": info.get("logo_url"),
                })
        except Exception:
            pass

        # 2️⃣ Fallback: TickerTape
        if not fundamentals.get("company_name"):
            tt_data = self._fetch_from_ticker_tape(query)
            fundamentals.update(tt_data)

        # 3️⃣ Fallback: Groww
        if not fundamentals.get("company_name"):
            groww_data = self._fetch_from_groww(query)
            fundamentals.update(groww_data)

        # 4️⃣ Financial Statements (Yahoo)
        try:
            fundamentals["financials"] = {
                "income_statement": self._convert_df_to_dict(stock.financials),
                "balance_sheet": self._convert_df_to_dict(stock.balance_sheet),
                "cashflow": self._convert_df_to_dict(stock.cashflow),
            }
        except Exception:
            fundamentals["financials"] = {}

        return fundamentals


# ---------------- RUN TEST ----------------
if __name__ == "__main__":
    agent = DataAgent(exchange="NSE")

    # You can test multiple stocks here
    stock_list = ["HDFC Bank"]
    results = []

    for s in stock_list:
        print(f"\n🔍 Fetching {s} ...")
        data = agent.fetch_fundamentals(s)
        results.append(data)
        print(json.dumps(data, indent=2, ensure_ascii=False))

    # Save all fundamentals to one file
    with open("all_stocks_fundamentals.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n✅ Saved all stock fundamentals → all_stocks_fundamentals.json")
