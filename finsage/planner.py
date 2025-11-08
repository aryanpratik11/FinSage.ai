"""Planner orchestrator for the FinSage MVP."""
from typing import Dict, Any, List, Optional
from .agents import (
    DataAgent,
    DocumentAgent,
    CalculationAgent,
    ReasoningAgent,
    NewsAgent,
    PredictionAgent,
    ComparisonAgent,
    RiskAgent,
    ValidationAgent,
    RAGAgent,
    LLMAgent,
)


class Planner:
    def __init__(self, sec_user_agent: Optional[str] = None):
        self.data_agent = DataAgent()
        self.doc_agent = DocumentAgent(user_agent=sec_user_agent)
        self.calc_agent = CalculationAgent()
        self.reasoning_agent = ReasoningAgent()
        self.news_agent = NewsAgent()
        self.pred_agent = PredictionAgent()
        self.comparison_agent = ComparisonAgent()
        self.risk_agent = RiskAgent()
        self.validation_agent = ValidationAgent()
    self.rag_agent = RAGAgent()
    self.llm_agent = LLMAgent()

    def _parse_tickers(self, query: str) -> List[str]:
        # Very small heuristic: look for uppercase tokens 1-5 chars long
        tokens = [t.strip(".,?()") for t in query.split()]
        picks = [t for t in tokens if t.isupper() and 1 <= len(t) <= 5]
        return picks

    def run(self, query: str, tickers: Optional[List[str]] = None, peers: Optional[List[str]] = None) -> Dict[str, Any]:
        print(f"[Planner] Running planner for query: {query}")
        tickers = tickers or self._parse_tickers(query)
        if not tickers:
            return {"error": "No ticker provided or detected in query. Pass tickers=[...] to Planner.run"}

        primary = tickers[0]
        context: Dict[str, Any] = {}

        # 1) Data
        data_out = self.data_agent.run(primary)
        if data_out.get("error"):
            context["data_error"] = data_out.get("error")
        else:
            context["fundamentals"] = data_out.get("fundamentals")
            context["history"] = data_out.get("history")

        # 2) Documents
        docs_out = self.doc_agent.run(primary)
        context["filings"] = docs_out.get("filings")

        # 3) News
        news_out = self.news_agent.run(primary)
        context["news"] = news_out

        # 4) Calculations
        calc_out = self.calc_agent.run(primary, {"fundamentals": context.get("fundamentals"), "history": context.get("history")})
        context["metrics"] = calc_out

        # 5) Prediction
        pred_out = self.pred_agent.run(primary, {"history": context.get("history")})
        context["prediction"] = pred_out

        # 6) Comparison (if peers provided)
        if peers:
            peer_data = []
            for p in peers:
                pd = self.data_agent.run(p)
                peer_data.append({"ticker": p, "fundamentals": pd.get("fundamentals")})
            context["peers"] = peer_data
            comp_out = self.comparison_agent.run(primary, {"peers": peer_data, "fundamentals": context.get("fundamentals")})
            context["comparison"] = comp_out

        # 7) Risk
        risk_out = self.risk_agent.run(primary, {"fundamentals": context.get("fundamentals"), "news": news_out})
        context["risk"] = risk_out

        # 8) Validation
        val_out = self.validation_agent.run(primary, {"fundamentals": context.get("fundamentals"), "history": context.get("history")})
        context["validation"] = val_out

        # 9) RAG ingest recent filings (if any) and retrieve top passages
        rag_results = None
        if context.get("filings"):
            docs = []
            for f in context.get("filings"):
                docs.append({"id": f.get("url"), "url": f.get("url"), "source": f.get("url")})
            ingest = self.rag_agent.ingest_urls(docs)
            rag_results = self.rag_agent.retrieve(query, top_k=5)
            context["rag"] = {"ingest": ingest, "retrieve": rag_results}

        # 10) LLM synthesis using retrieved passages (RAG)
        llm_out = self.llm_agent.run(query, passages=(rag_results.get("results") if rag_results else None))
        context["llm"] = llm_out

        # 11) Final reasoning combines LLM output if available
        if llm_out and not llm_out.get("error"):
            context["thesis"] = llm_out.get("text")[:1500]
        else:
            reasoning_out = self.reasoning_agent.run(primary, {"fundamentals": context.get("fundamentals"), "news": news_out, "metrics": calc_out, "prediction": pred_out})
            context["thesis"] = reasoning_out.get("thesis")

        
        response = {
            "query": query,
            "ticker": primary,
            "fundamentals": context.get("fundamentals"),
            "history_len": len(context.get("history") or []),
            "filings": context.get("filings"),
            "news": news_out,
            "metrics": calc_out,
            "prediction": pred_out,
            "comparison": context.get("comparison"),
            "risk": risk_out,
            "validation": val_out,
            "thesis": context.get("thesis"),
        }
        print(f"[Planner] Synthesis complete for {primary}.")
        return response
