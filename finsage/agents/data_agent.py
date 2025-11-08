"""DataAgent: fetches structured financial data using yfinance (no API key required).

This agent fetches price history and basic fundamentals for a ticker.
It falls back gracefully if yfinance isn't installed and returns helpful error messages.
"""
from typing import Dict, Any

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None


class DataAgent:
    """Fetch structured financial data for a ticker using yfinance."""

    name = "DataAgent"

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        if yf is None:
            return {"error": "yfinance not installed. Install with 'pip install yfinance'"}

        tk = yf.Ticker(ticker)
        info = tk.info if hasattr(tk, "info") else {}

        # Historical prices (last 90 days)
        hist = tk.history(period="90d", auto_adjust=False)
        hist_records = []
        if not hist.empty:
            hist_records = [
                {"date": str(idx.date()), "close": float(row["Close"]), "volume": int(row["Volume"])}
                for idx, row in hist.iterrows()
            ]

        fundamentals = {
            "shortName": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "previousClose": info.get("previousClose"),
            "regularMarketPrice": info.get("regularMarketPrice"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "epsTrailingTwelveMonths": info.get("epsTrailingTwelveMonths"),
        }

        return {
            "ticker": ticker,
            "source": "yfinance",
            "fundamentals": fundamentals,
            "history": hist_records,
        }
