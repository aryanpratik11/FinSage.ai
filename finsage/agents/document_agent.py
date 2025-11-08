"""DocumentAgent: fetch recent SEC filings for a ticker using SEC's submissions endpoint.

Works without API key. Uses SEC public endpoints and requires a proper User-Agent header.
"""
from typing import Dict, Any, List
import requests
import time

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class DocumentAgent:
    name = "DocumentAgent"

    def __init__(self, user_agent: str = None):
        # SEC requires a descriptive User-Agent
        self.user_agent = user_agent or "finsage-agent (email@example.com)"

    def _get_headers(self):
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    def _load_ticker_map(self) -> Dict[str, Any]:
        resp = requests.get(SEC_TICKER_MAP_URL, headers=self._get_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def run(self, ticker: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        try:
            mapping = self._load_ticker_map()
        except Exception as e:
            return {"error": f"Failed to load SEC ticker map: {e}"}

        # The SEC ticker map keys are numeric strings for entries; find matching entry
        cik = None
        for key, val in mapping.items():
            if val.get("ticker", "").upper() == ticker.upper():
                cik = val.get("cik_str")
                break

        if cik is None:
            return {"error": f"CIK not found for ticker {ticker}"}

        # Zero-pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)
        try:
            url = SUBMISSIONS_URL.format(cik=cik_padded)
            resp = requests.get(url, headers=self._get_headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"error": f"Failed to fetch submissions for CIK {cik}: {e}"}

        filings = data.get("filings", {}).get("recent", {})
        results: List[Dict[str, Any]] = []
        # Collect most recent 5 filings
        for i in range(min(5, len(filings.get("accessionNumber", [])))):
            acc = filings.get("accessionNumber")[i]
            form = filings.get("form")[i]
            filing_date = filings.get("filingDate")[i]
            # Build filing URL (text file)
            accession_nodashes = acc.replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/"
            doc_url = base + acc + "-index.htm"
            results.append({"form": form, "filing_date": filing_date, "url": doc_url})

        return {"ticker": ticker, "source": "sec.submissions", "filings": results}
