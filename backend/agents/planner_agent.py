import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from backend.utils.llm_service import call_llm
from backend.models.chat_models import AgentStep
from backend.agents.data_agent import data_agent
from backend.agents.news_agent import news_agent
from backend.agents.comparison_agent import comparison_agent
# from backend.agents.prediction_agent import prediction_agent
# from backend.agents.reasoning_agent import reasoning_agent
# from backend.agents.document_agent import document_agent
# from backend.agents.calculation_agent import calculation_agent
# from backend.agents.risk_assessment_agent import risk_agent
# from backend.agents.validation_agent import validation_agent


# ───────────────────────── SAFE TEXT LIMITER ─────────────────────────

def truncate(text: Any, limit: int = 1500) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[TRUNCATED]..."


class PlannerAgent:
    """
    LLM-structured multi-step pipeline:
    
    1) Parse & plan using local Llama
    2) News agent → all important recent news about the entity
    3) Data agent → fundamentals, prices, ratios, etc.
    4) Calculation agent → compute key metrics from raw data (P/E, growth, etc.) [future]
    5) Reasoning agent (or LLM fallback) → analyze investment case
    6) Prediction agent (optional) → future outlook if relevant
    7) Validation agent → sanity check and confidence
    """

    AVAILABLE_AGENTS = [
        "news_agent",
        "data_agent",
        "calculation_agent",
        "reasoning_agent",
        "prediction_agent",
        "document_agent",
        "comparison_agent",
        "risk_agent",
        "validation_agent"
    ]

    # Strict order that matches your description
    AGENT_PRIORITY = {
        "news_agent": 1,          # First: related news
        "data_agent": 2,          # Second: market/financial data
        "calculation_agent": 3,   # Third: derived metrics
        "reasoning_agent": 4,     # Fourth: reasoning on all info
        "prediction_agent": 5,    # Fifth: future outlook if needed
        "comparison_agent": 5,    # Peer comparison around same stage
        "risk_agent": 5,          # Risk assessment with prediction
        "document_agent": 6,      # Filings / deeper docs if needed
        "validation_agent": 7     # Last: validation
    }

    def __init__(self):
        self.name = "Planner Agent"
        self.execution_log: List[AgentStep] = []

    async def handle_query(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Returns:
            {
              "response": final LLM answer,
              "company": detected entity,
              "intent": high-level intent,
              "confidence": float,
              "sources": [...],
              "agent_steps": [...],
              "agents_used": [...],
              "processing_time_ms": float
            }
        """
        self.execution_log = []
        start_time = datetime.utcnow()

        try:
            print(f"Planner Agent: Query received - {query}")
            self._log_step("planner", "Query received", f"Processing: {query[:100]}...")

            # STEP 1: LLM planning / parsing
            analysis = await self.analyze_and_plan(query, context)

            company = analysis.get("company")
            intent = analysis.get("intent")
            subtasks = analysis.get("agents", [])

            # Validate and order the agent pipeline
            ordered_agents = self._validate_and_prioritize_plan(subtasks)

            # STEP 2: Execute agents
            results = await self.execute_agents(ordered_agents, query, context=context, company=company)

            # STEP 3: Synthesize final response
            final_text = await self.synthesize_response(query, results, intent or "general")

            # Meta information
            confidence = self._calculate_confidence(results)
            sources = self._extract_sources(results)
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

            return {
                "response": final_text,
                "company": company,
                "intent": intent,
                "confidence": confidence,
                "sources": sources,
                "agent_steps": [s.dict() for s in self.execution_log],
                "agents_used": ordered_agents,
                "processing_time_ms": processing_time_ms
            }

        except Exception as e:
            self._log_step("planner", "Failed", str(e))
            return {
                "response": self._generate_error_response(query, str(e)),
                "company": None,
                "intent": "error",
                "confidence": 0.0,
                "sources": [],
                "agent_steps": [s.dict() for s in self.execution_log],
                "agents_used": [],
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000.0
            }
    # ───────────────────────── LLM PLANNING STEP ─────────────────────────

    async def analyze_and_plan(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Use Llama to parse the query, identify entity, intent,
        and decide which agents to use (but the order is fixed by AGENT_PRIORITY).
        """
        # keep query and context small for the planner
        query_short = truncate(query, 600)

        ctx_str = ""
        if context:
            # last 3 messages for lightweight context
            ctx_str = "\nRecent context:\n" + truncate(
                json.dumps(context[-3:], ensure_ascii=False),
                500
            )

        # Hint: Prefer Indian markets by default unless user specifies otherwise.
        market_hint = "Assume the Indian market (NSE/BSE) by default unless the user explicitly asks about US or other markets."

        prompt = f"""
        You are FinSage's Planner AI.

        Your job:
        1. Understand the user's financial/investment query
        2. Detect the main entity (company / stock / crypto / index / sector) — or NULL if it's a general/topical question
        3. Classify the high-level intent
        4. Decide which analysis stages are needed.

        IMPORTANT: If the query asks for a COMPARISON, LIST, or GENERAL ADVICE (e.g., "best companies", "which stocks", "top performers"), set company to NULL and use comparison_agent.

        Available stages (agents):
        - news_agent: Find recent important news about the entity or topic
        - data_agent: Get market/financial data and fundamentals (ONLY for specific companies)
        - calculation_agent: Compute key metrics (valuations, growth, ratios)
        - reasoning_agent: Combine information and analyze investment case
        - prediction_agent: Provide future outlook or scenarios
        - risk_agent: Assess major risks
        - document_agent: Use filings/reports if needed
        - comparison_agent: Compare multiple stocks, peers, sectors (for topical/list queries)
        - validation_agent: Final consistency check

        RULES:
        - If query contains "best", "top", "compare", "which", "list", "companies", "stocks" → company=null, use comparison_agent
        - If query mentions ONE specific company/ticker → company=that_company, use standard pipeline
        - If unclear, default to company=null with comparison_agent

        User Query:
        """
        {query_short}
        """
        {ctx_str}

        Market hint:
        {market_hint}

        Return a JSON object ONLY, no extra text.

JSON format:
{{
    "company": string or null,
    "intent": string,  // e.g. "buy/sell decision", "comparison", "sector analysis", etc.
    "need_prediction": boolean,
    "need_risk_assessment": boolean,
    "agents": ["news_agent", "comparison_agent", "reasoning_agent", "validation_agent"]
}}
"""

        response = await call_llm(prompt, max_tokens=250)

        # Safe JSON parsing
        result: Optional[Dict[str, Any]] = None
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                try:
                    result = json.loads(match.group())
                except Exception:
                    result = None

        # Fallback if LLM returned junk
        if not result or not isinstance(result, dict):
            self._log_step("planner", "LLM output invalid", f"Raw output: {truncate(response, 500)}")
            result = {
                "company": None,
                "intent": "general_financial_query",
                "need_prediction": False,
                "need_risk_assessment": False,
                "agents": [
                    "news_agent",
                    "data_agent",
                    "calculation_agent",
                    "reasoning_agent",
                    "validation_agent"
                ]
            }

        # Ensure agents list exists
        if "agents" not in result or not isinstance(result["agents"], list):
            result["agents"] = [
                "news_agent",
                "data_agent",
                "calculation_agent",
                "reasoning_agent",
                "validation_agent"
            ]

        # Inject prediction/risk agents if LLM flags them
        if result.get("need_prediction"):
            if "prediction_agent" not in result["agents"]:
                result["agents"].append("prediction_agent")
        if result.get("need_risk_assessment"):
            if "risk_agent" not in result["agents"]:
                result["agents"].append("risk_agent")

        # If the planner could not detect a specific company/entity, avoid
        # running the `data_agent` which expects a ticker or company name.
        # For general/topical queries, prefer news, reasoning and comparison.
        if not result.get("company"):
            agents = [a for a in result.get("agents", []) if a != "data_agent"]

            # Ensure at least news + comparison + reasoning are present for topical queries
            if "news_agent" not in agents:
                agents.insert(0, "news_agent")
            if "comparison_agent" not in agents:
                agents.append("comparison_agent")
            if "reasoning_agent" not in agents:
                agents.append("reasoning_agent")

            # Keep prediction/risk if explicitly requested
            if result.get("need_prediction") and "prediction_agent" not in agents:
                agents.append("prediction_agent")
            if result.get("need_risk_assessment") and "risk_agent" not in agents:
                agents.append("risk_agent")

            if "validation_agent" not in agents:
                agents.append("validation_agent")

            # Deduplicate while preserving order
            seen = set()
            filtered = []
            for a in agents:
                if a not in seen:
                    filtered.append(a)
                    seen.add(a)

            result["agents"] = filtered or ["news_agent", "reasoning_agent", "validation_agent"]

        return result

    # ───────────────────────── PLAN VALIDATION / ORDERING ─────────────────────────

    def _validate_and_prioritize_plan(self, subtasks: List[str]) -> List[str]:
        """
        Keep only known agents, remove duplicates, then sort by AGENT_PRIORITY
        to enforce the step-by-step pipeline structure.
        """
        # Filter to known agents
        valid = [a for a in subtasks if a in self.AVAILABLE_AGENTS]
        # Remove duplicates while preserving order
        unique = list(dict.fromkeys(valid))

        # Sort by priority (lower = earlier)
        sorted_tasks = sorted(unique, key=lambda x: self.AGENT_PRIORITY.get(x, 999))

        self._log_step("planner", "Final agent pipeline", f" → ".join(sorted_tasks) or "No agents selected")
        return sorted_tasks

    # ───────────────────────── EXECUTION LAYER ─────────────────────────

    async def execute_agents(
        self,
        subtasks: List[str],
        query: str,
        context: Optional[List[Dict[str, str]]] = None,
        company: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute agents in priority order.
        Agents with the same numeric priority can run in parallel.
        """
        results: Dict[str, Any] = {}

        # Group agents by priority
        priority_groups: Dict[int, List[str]] = {}
        for agent in subtasks:
            priority = self.AGENT_PRIORITY.get(agent, 999)
            priority_groups.setdefault(priority, []).append(agent)

        # Run groups in ascending priority order
        for priority in sorted(priority_groups.keys()):
            agents_in_group = priority_groups[priority]

            tasks = [
                self._run_agent(agent, query, results, context, company)
                for agent in agents_in_group
            ]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent, result in zip(agents_in_group, group_results):
                if isinstance(result, Exception):
                    self._log_step(agent, "Execution failed", str(result))
                    results[agent] = {"error": str(result)}
                else:
                    results[agent] = result

        return results

    async def _run_agent(
        self,
        agent_name: str,
        query: str,
        previous_results: Dict[str, Any],
        context: Optional[List[Dict[str, str]]] = None,
        company: Optional[str] = None
    ) -> Any:
        """
        Run a single agent, with logging.
        """
        self._log_step(agent_name, "Executing", "Processing...")

        try:
            # 📰 News stage
            if agent_name == "news_agent":
                print(f"Planner Agent: Running news_agent for entity: {company or query}")
                result = await news_agent.get_company_news(company or query)

            # 📊 Data stage
            elif agent_name == "data_agent":
                print(f"Planner Agent: Running data_agent for entity: {company or query}")
                result = await data_agent.get_financial_data(company or query)

            # 🧮 Calculation stage (placeholder)
            elif agent_name == "calculation_agent":
                # Example placeholder: in future, use calculation_agent and pass data_agent results
                # result = await calculation_agent.compute_metrics(query, previous_results)
                data = previous_results.get("data_agent")
                result = {
                    "mock": True,
                    "note": "calculation_agent not implemented yet",
                    "based_on_data": bool(data)
                }

            # 🧠 Reasoning stage
            elif agent_name == "reasoning_agent":
                # If you implement a dedicated reasoning_agent, call it here
                # result = await reasoning_agent.analyze_context(query, previous_results, context)
                result = await self._reasoning_fallback(query, previous_results)

            # 📈 Prediction stage
            elif agent_name == "prediction_agent":
                # result = await prediction_agent.forecast_trends(query, previous_results)
                # For now, use LLM to provide a cautious qualitative outlook
                result = await self._prediction_fallback(query, previous_results)

            # 📑 Document / risk / validation
            elif agent_name == "document_agent":
                result = {"mock": "document_agent not implemented yet"}

            # 🔄 Comparison stage
            elif agent_name == "comparison_agent":
                print(f"Planner Agent: Running comparison_agent for query: {query}")
                result = await comparison_agent.compare_stocks(query, limit=5)

            elif agent_name == "risk_agent":
                result = await self._risk_fallback(query, previous_results)

            elif agent_name == "validation_agent":
                # If you have a real validation_agent, plug it here
                # result = await validation_agent.verify_results(previous_results)
                result = {
                    "validated": True,
                    "confidence": 0.85,
                    "notes": "Simple validation placeholder"
                }

            else:
                result = {"error": f"Unknown agent: {agent_name}"}

            self._log_step(agent_name, "Completed", "Success")
            return result

        except Exception as e:
            self._log_step(agent_name, "Failed", str(e))
            raise

    # ───────────────────────── LLM FALLBACK HELPERS ─────────────────────────

    async def _reasoning_fallback(
        self,
        query: str,
        previous_results: Dict[str, Any]
    ) -> str:
        """Use Llama to reason over aggregated agent outputs."""
        context_lines = []
        for k, v in previous_results.items():
            try:
                serialized = truncate(json.dumps(v, indent=2, default=str), 1200)
            except Exception:
                serialized = truncate(v, 1200)
            context_lines.append(f"{k}:\n{serialized}")

        context_str = "\n\n".join(context_lines)
        context_str = truncate(context_str, 2000)

        market_hint = "Assume the Indian market (NSE/BSE) by default unless user explicitly requests another market."

        prompt = f"""
    You are FinSage's Reasoning AI.

    User query:
    {truncate(query, 400)}

    Data from previous analysis stages:
    {context_str}

    Task:
    - Summarize the current situation of the entity or topic.
    - Analyze investment implications (upside, downside, key drivers).
    - Keep it focused, practical, and investor-friendly.
    - Avoid making absolute guarantees; talk in terms of scenarios and probabilities.

    Write 2–4 concise paragraphs.

    Market hint: {market_hint}
    """

        return await call_llm(prompt, max_tokens=350)

    async def _prediction_fallback(
        self,
        query: str,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Qualitative future outlook using Llama (since no numeric model is wired yet)."""
        context_str = truncate(json.dumps(previous_results, default=str), 1500)

        market_hint = "Assume the Indian market (NSE/BSE) by default unless user explicitly requests another market."

        prompt = f"""
    You are FinSage's Forecast AI.

    User query:
    {truncate(query, 400)}

    Background analysis so far:
    {context_str}

    Task:
    - Provide a cautious, scenario-based future outlook (short-term and medium-term if relevant).
    - Explicitly mention uncertainty and factors that could change the outlook.
    - DO NOT claim precise price targets or guaranteed returns.

    Return your answer as plain text (no JSON needed).

    Market hint: {market_hint}
    """

        text = await call_llm(prompt, max_tokens=300)
        return {"qualitative_outlook": text.strip()}

    async def _risk_fallback(
        self,
        query: str,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM-based risk assessment placeholder."""
        context_str = truncate(json.dumps(previous_results, default=str), 1500)

        prompt = f"""
You are FinSage's Risk Analysis AI.

User query:
{truncate(query, 400)}

Existing analysis:
{context_str}

List the main categories of risk (e.g., market risk, company-specific risk, regulatory risk, valuation risk)
and briefly explain each in 1–2 sentences.

Return a short bullet-style text description.
"""

        text = await call_llm(prompt, max_tokens=250)
        return {"risk_summary": text.strip()}

    # ───────────────────────── RESPONSE SYNTHESIS ─────────────────────────

    async def synthesize_response(
        self,
        query: str,
        results: Dict[str, Any],
        intent: str
    ) -> str:
        """
        Combines all agent outputs into a polished final summary using Llama.
        """
        # Build context from results
        context_parts = []
        for agent, data in results.items():
            # Only ignore explicit errors
            if isinstance(data, dict) and "error" in data:
                continue

            try:
                serialized = truncate(json.dumps(data, indent=2, default=str), 1200)
            except Exception:
                serialized = truncate(data, 1200)

            context_parts.append(f"{agent}:\n{serialized}")

        context = truncate("\n\n".join(context_parts), 3000)

        market_hint = "Assume the Indian market (NSE/BSE) by default unless user explicitly requests another market."

        prompt = f"""
    You are FinSage AI, an expert financial analyst assistant.

    User Query:
    {truncate(query, 500)}

    Detected Intent:
    {intent}

    Below is the multi-stage analysis from different agents (news, data, calculations, reasoning, prediction, risk, validation):

    {context}

    TASK:
    - Start with a clear, direct answer to the user's question.
    - Use data, news, metrics, and reasoning to justify the answer.
    - If there is a forecast/outlook, present it as scenarios with uncertainty.
    - Highlight key risks and what could invalidate the view.
    - End with 2–4 concrete, actionable suggestions (e.g., what to monitor, what to compare, what to consider).

    Tone:
    - Professional but conversational
    - No hype, no absolute guarantees
    - Suitable for a retail investor with basic knowledge

    Write 3–6 short paragraphs.

    Market hint: {market_hint}
    """

        try:
            response = await call_llm(prompt, max_tokens=500)
            return response.strip()
        except Exception as e:
            return self._generate_error_response(query, str(e))

    # ───────────────────────── META UTILITIES ─────────────────────────

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Simple heuristic confidence: based on how many agents succeeded
        and whether validation exists.
        """
        if not results:
            return 0.0

        total = len(results)
        valid_count = 0

        for r in results.values():
            if isinstance(r, dict) and "error" not in r:
                valid_count += 1
            elif isinstance(r, str) and r.strip():
                valid_count += 1

        base = 0.6 + (valid_count / (total or 1)) * 0.3

        if "validation_agent" in results:
            val_conf = results["validation_agent"].get("confidence", 0.75)
            base = (base + val_conf) / 2

        return min(base, 1.0)

    def _extract_sources(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract simple source metadata based on which agents ran."""
        sources = []

        if "data_agent" in results:
            sources.append({
                "type": "Market Data",
                "name": "Financial APIs (e.g., Yahoo Finance / Alpha Vantage)",
                "date": datetime.utcnow().strftime("%Y-%m-%d")
            })

        if "news_agent" in results:
            sources.append({
                "type": "Financial News",
                "name": "News APIs / major financial news outlets",
                "date": datetime.utcnow().strftime("%Y-%m-%d")
            })

        if "document_agent" in results:
            sources.append({
                "type": "Filings / Reports",
                "name": "Regulatory/filing databases (e.g., SEC EDGAR)",
                "date": datetime.utcnow().strftime("%Y-%m-%d")
            })

        return sources

    def _log_step(self, agent_name: str, action: str, result: str):
        """Log a step in the execution trace."""
        self.execution_log.append(
            AgentStep(
                agent_name=agent_name,
                action=action,
                result=result,
                timestamp=datetime.utcnow()
            )
        )

    def _generate_error_response(self, query: str, error: str) -> str:
        """Generate a user-friendly error response."""
        return f"""I’m sorry, but I ran into a problem while processing your query:

\"{query}\"

Error details (for debugging):
{error}

You can try:
- Rephrasing or simplifying the question
- Being specific about the company, ticker, or asset
- Asking for a particular kind of analysis (news impact, valuation, comparison, etc.)

I’m designed to help with financial analysis, stock information, market insights, and investment-oriented questions."""


planner_agent = PlannerAgent()
