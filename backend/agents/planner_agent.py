import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from backend.services.llm_service import call_llm
from backend.models.chat_models import AgentStep

# Import agents as they become available
# from backend.agents.data_agent import data_agent
# from backend.agents.news_agent import news_agent
# from backend.agents.prediction_agent import prediction_agent
# from backend.agents.reasoning_agent import reasoning_agent
# from backend.agents.document_agent import document_agent
# from backend.agents.calculation_agent import calculation_agent
# from backend.agents.comparison_agent import comparison_agent
# from backend.agents.risk_assessment_agent import risk_agent
# from backend.agents.validation_agent import validation_agent


class PlannerAgent:
    """
    Central orchestrator for FinSage AI multi-agent system.
    
    Responsibilities:
    1. Analyze user intent
    2. Create dynamic execution plan
    3. Orchestrate agent execution (parallel when possible)
    4. Validate results
    5. Synthesize final response
    """
    
    # Available agents registry
    AVAILABLE_AGENTS = [
        "data_agent",
        "news_agent", 
        "prediction_agent",
        "reasoning_agent",
        "document_agent",
        "calculation_agent",
        "comparison_agent",
        "risk_agent",
        "validation_agent"
    ]
    
    # Agent execution priority (lower = higher priority)
    AGENT_PRIORITY = {
        "document_agent": 1,  # Get documents first
        "data_agent": 2,      # Then fetch data
        "news_agent": 2,      # News in parallel with data
        "calculation_agent": 3, # Calculate after data available
        "comparison_agent": 3,  # Compare after data available
        "prediction_agent": 4,  # Predict based on data
        "risk_agent": 4,       # Assess risk based on data
        "reasoning_agent": 5,  # Reason about everything
        "validation_agent": 6  # Always validate last
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
        Central orchestrator — plans, executes, and synthesizes agent outputs.
        
        Args:
            query: User's financial query
            context: Conversation history (optional)
            
        Returns:
            Dictionary containing:
                - response: Final synthesized answer
                - confidence: Confidence score (0-1)
                - sources: List of data sources used
                - agent_steps: Execution log
                - agents_used: List of agent names invoked
                - intent: Detected query intent
        """
        self.execution_log = []
        start_time = datetime.utcnow()
        
        try:
            self._log_step("planner", "Query received", f"Processing: {query[:100]}...")
            
            # Step 1: Intent analysis
            intent = await self.analyze_intent(query, context)
            self._log_step("planner", "Intent analysis", f"Detected: {intent}")
            
            # Step 2: Dynamic task planning
            subtasks = await self.dynamic_plan(intent, query, context)
            self._log_step("planner", "Task planning", f"Planned: {', '.join(subtasks)}")
            
            # Step 3: Validate plan
            subtasks = self._validate_and_prioritize_plan(subtasks)
            
            # Step 4: Execute agents in optimized order
            results = await self.execute_agents(subtasks, query, context)
            
            # Step 5: Always run validation agent if available
            if "validation_agent" not in subtasks:
                self._log_step("validation", "Final validation", "Verifying results...")
                # results["validation_agent"] = await self._run_validation(results)
            
            # Step 6: Calculate confidence based on results
            confidence = self._calculate_confidence(results)
            
            # Step 7: Extract sources
            sources = self._extract_sources(results)
            
            # Step 8: Final synthesis
            final_answer = await self.synthesize_response(query, results, intent)
            self._log_step("planner", "Response synthesis", "Complete")
            
            return {
                "response": final_answer,
                "confidence": confidence,
                "sources": sources,
                "agent_steps": self.execution_log,
                "agents_used": list(results.keys()),
                "intent": intent,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            self._log_step("planner", "Error occurred", str(e))
            return {
                "response": self._generate_error_response(query, str(e)),
                "confidence": 0.0,
                "sources": [],
                "agent_steps": self.execution_log,
                "agents_used": [],
                "intent": "error",
                "error": str(e)
            }
    
    async def analyze_intent(
        self, 
        query: str, 
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Uses LLM to classify the financial query intent.
        
        Args:
            query: User query
            context: Previous conversation (optional)
            
        Returns:
            Intent classification string
        """
        context_text = ""
        if context:
            recent = context[-3:]  # Last 3 messages
            context_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        
        prompt = f"""You are the FinSage Intent Analyzer.

Analyze this financial query and classify its primary intent in ONE precise phrase.

Query: "{query}"

{f'Recent Conversation Context:\n{context_text}\n' if context_text else ''}

Intent Categories:
- Stock price inquiry
- Investment recommendation
- Company analysis
- Market prediction
- Financial document retrieval
- Portfolio comparison
- Risk assessment
- News and sentiment analysis
- Financial calculation
- General financial advice

Return ONLY the intent phrase, nothing else.

Examples:
Query: "What is Tesla's stock price?"
Intent: Stock price inquiry

Query: "Should I invest in Apple?"
Intent: Investment recommendation

Query: "Compare AAPL and MSFT"
Intent: Portfolio comparison

Query: "Show me Tesla's latest 10-K"
Intent: Financial document retrieval

Now analyze the query above:"""
        
        try:
            intent = await call_llm(prompt, max_tokens=50)
            return intent.strip()
        except Exception as e:
            self._log_step("planner", "Intent analysis failed", str(e))
            return "general_query"
    
    async def dynamic_plan(
        self, 
        intent: str, 
        query: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Dynamically determines which agents to invoke based on intent and query.
        
        Args:
            intent: Detected intent
            query: User query
            context: Conversation context
            
        Returns:
            Ordered list of agent names to execute
        """
        planning_prompt = f"""You are the FinSage Task Planner.

Based on the user's query and intent, determine which specialized agents should be invoked.

Available Agents:
- data_agent: Fetches real-time stock prices, market data, company fundamentals
- news_agent: Retrieves latest financial news and sentiment
- prediction_agent: Makes forecasts and predictions
- reasoning_agent: Analyzes data and provides insights
- document_agent: Retrieves SEC filings, reports, documents
- calculation_agent: Performs financial calculations (P/E, DCF, ratios)
- comparison_agent: Compares multiple stocks/companies
- risk_agent: Assesses investment risks
- validation_agent: Validates outputs for accuracy

User Intent: "{intent}"
User Query: "{query}"

Instructions:
1. Select ONLY the agents needed (don't invoke all unnecessarily)
2. Return as a JSON array of agent names
3. Order agents logically (e.g., get data before calculating)
4. Always include reasoning_agent for synthesis

Examples:

Query: "What is AAPL stock price?"
Response: ["data_agent", "reasoning_agent"]

Query: "Should I invest in Tesla? Compare with Ford."
Response: ["data_agent", "news_agent", "comparison_agent", "risk_agent", "reasoning_agent"]

Query: "Show me Microsoft's latest 10-K and calculate their P/E"
Response: ["document_agent", "data_agent", "calculation_agent", "reasoning_agent"]

Query: "Predict next week's NIFTY trend"
Response: ["data_agent", "news_agent", "prediction_agent", "reasoning_agent"]

Now plan for the query above. Return ONLY the JSON array:"""
        
        try:
            response = await call_llm(planning_prompt, max_tokens=200)
            
            # Parse JSON response
            agents = self._parse_agent_list(response)
            
            # Ensure reasoning_agent is always included
            if "reasoning_agent" not in agents:
                agents.append("reasoning_agent")
            
            # Filter to only available agents
            agents = [a for a in agents if a in self.AVAILABLE_AGENTS]
            
            return agents if agents else ["reasoning_agent"]
            
        except Exception as e:
            self._log_step("planner", "Planning failed, using fallback", str(e))
            # Fallback to basic agents
            return ["data_agent", "reasoning_agent"]
    
    def _parse_agent_list(self, response: str) -> List[str]:
        """
        Parse agent list from LLM response (handles JSON or text).
        
        Args:
            response: LLM response text
            
        Returns:
            List of agent names
        """
        try:
            # Try parsing as JSON
            agents = json.loads(response)
            if isinstance(agents, list):
                return agents
        except:
            pass
        
        # Fallback: regex extract agent names
        agents = re.findall(r'\b(\w+_agent)\b', response)
        return list(set(agents))  # Remove duplicates
    
    def _validate_and_prioritize_plan(self, subtasks: List[str]) -> List[str]:
        """
        Validate plan and sort agents by execution priority.
        
        Args:
            subtasks: List of agent names
            
        Returns:
            Prioritized list of valid agents
        """
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for task in subtasks:
            if task not in seen and task in self.AVAILABLE_AGENTS:
                seen.add(task)
                unique.append(task)
        
        # Sort by priority (validation_agent always last)
        sorted_tasks = sorted(
            unique, 
            key=lambda x: (
                self.AGENT_PRIORITY.get(x, 999),
                unique.index(x)  # Preserve original order for same priority
            )
        )
        
        return sorted_tasks
    
    async def execute_agents(
        self, 
        subtasks: List[str], 
        query: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes selected agents with optimized parallelization.
        
        Agents at the same priority level run in parallel.
        
        Args:
            subtasks: List of agent names to execute
            query: User query
            context: Conversation context
            
        Returns:
            Dictionary of agent results
        """
        results = {}
        
        # Group agents by priority for parallel execution
        priority_groups = {}
        for agent in subtasks:
            priority = self.AGENT_PRIORITY.get(agent, 999)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(agent)
        
        # Execute each priority group sequentially, agents within group in parallel
        for priority in sorted(priority_groups.keys()):
            agents_in_group = priority_groups[priority]
            
            # Run all agents in this priority group in parallel
            tasks = [self._run_agent(agent, query, results, context) for agent in agents_in_group]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
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
        context: Optional[List[Dict[str, str]]] = None
    ) -> Any:
        """
        Run a single agent and log execution.
        
        Args:
            agent_name: Name of agent to execute
            query: User query
            previous_results: Results from previously executed agents
            context: Conversation context
            
        Returns:
            Agent execution result
        """
        self._log_step(agent_name, "Executing", f"Processing query...")
        
        try:
            # TODO: Replace with actual agent calls when implemented
            # For now, return mock data
            
            if agent_name == "data_agent":
                # result = await data_agent.get_financial_data(query)
                result = {"mock": "data_agent not implemented yet"}
                
            elif agent_name == "news_agent":
                # result = await news_agent.get_latest_news(query)
                result = {"mock": "news_agent not implemented yet"}
                
            elif agent_name == "prediction_agent":
                # result = await prediction_agent.forecast_trends(query, previous_results)
                result = {"mock": "prediction_agent not implemented yet"}
                
            elif agent_name == "document_agent":
                # result = await document_agent.extract_information(query)
                result = {"mock": "document_agent not implemented yet"}
                
            elif agent_name == "calculation_agent":
                # result = await calculation_agent.compute_metrics(query, previous_results)
                result = {"mock": "calculation_agent not implemented yet"}
                
            elif agent_name == "comparison_agent":
                # result = await comparison_agent.compare_peers(query, previous_results)
                result = {"mock": "comparison_agent not implemented yet"}
                
            elif agent_name == "risk_agent":
                # result = await risk_agent.assess_risks(query, previous_results)
                result = {"mock": "risk_agent not implemented yet"}
                
            elif agent_name == "reasoning_agent":
                # result = await reasoning_agent.analyze_context(query, previous_results, context)
                result = await self._reasoning_fallback(query, previous_results)
                
            elif agent_name == "validation_agent":
                # result = await validation_agent.verify_results(previous_results)
                result = {"validated": True, "confidence": 0.85}
                
            else:
                result = {"error": f"Unknown agent: {agent_name}"}
            
            self._log_step(agent_name, "Completed", "Success")
            return result
            
        except Exception as e:
            self._log_step(agent_name, "Failed", str(e))
            raise
    
    async def _reasoning_fallback(self, query: str, previous_results: Dict[str, Any]) -> str:
        """Fallback reasoning when reasoning_agent not implemented"""
        context = "\n".join([f"{k}: {v}" for k, v in previous_results.items()])
        
        prompt = f"""Analyze this financial query and provide insights:

Query: {query}

Available Data:
{context}

Provide a brief analysis:"""
        
        return await call_llm(prompt)
    
    async def synthesize_response(
        self, 
        query: str, 
        results: Dict[str, Any],
        intent: str
    ) -> str:
        """
        Combines all agent outputs into a polished final summary.
        
        Args:
            query: Original user query
            results: Dictionary of agent results
            intent: Detected intent
            
        Returns:
            Final synthesized response
        """
        # Build context from results
        context_parts = []
        for agent, data in results.items():
            if data and not isinstance(data, dict) or (isinstance(data, dict) and "error" not in data):
                context_parts.append(f"**{agent.replace('_', ' ').title()}**:\n{data}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are FinSage AI, an expert financial analyst assistant.

User Query: "{query}"
Detected Intent: {intent}

Based on the multi-agent analysis below, synthesize a comprehensive, professional response.

Requirements:
1. Start with a direct answer to the user's question
2. Support with data and facts from the analysis
3. Include relevant metrics, calculations, or comparisons
4. Mention news/sentiment if relevant
5. Provide risk assessment if available
6. End with actionable insights or recommendations
7. Be concise but thorough (3-5 paragraphs max)
8. Use professional but conversational tone

Agent Analysis:
{context}

Generate the final response:"""
        
        try:
            response = await call_llm(prompt, max_tokens=800)
            return response.strip()
        except Exception as e:
            return self._generate_error_response(query, str(e))
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on agent results.
        
        Args:
            results: Dictionary of agent results
            
        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Boost for successful agent executions
        successful = sum(1 for r in results.values() if not isinstance(r, dict) or "error" not in r)
        total = len(results)
        success_rate = successful / total if total > 0 else 0
        
        confidence += success_rate * 0.2
        
        # Boost for validation
        if "validation_agent" in results:
            val_conf = results["validation_agent"].get("confidence", 0.8)
            confidence = (confidence + val_conf) / 2
        
        return min(confidence, 1.0)
    
    def _extract_sources(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract data sources from agent results"""
        sources = []
        
        # Add sources based on which agents ran
        if "data_agent" in results:
            sources.append({
                "type": "Market Data",
                "name": "Yahoo Finance / Alpha Vantage",
                "date": datetime.utcnow().strftime("%Y-%m-%d")
            })
        
        if "news_agent" in results:
            sources.append({
                "type": "Financial News",
                "name": "NewsAPI / Financial Times",
                "date": datetime.utcnow().strftime("%Y-%m-%d")
            })
        
        if "document_agent" in results:
            sources.append({
                "type": "SEC Filings",
                "name": "SEC EDGAR Database",
                "date": datetime.utcnow().strftime("%Y-%m-%d")
            })
        
        return sources
    
    def _log_step(self, agent_name: str, action: str, result: str):
        """Log an agent execution step"""
        self.execution_log.append(
            AgentStep(
                agent_name=agent_name,
                action=action,
                result=result,
                timestamp=datetime.utcnow()
            )
        )
    
    def _generate_error_response(self, query: str, error: str) -> str:
        """Generate user-friendly error response"""
        return f"""I apologize, but I encountered an issue processing your query: "{query}"

Error: {error}

Please try:
- Rephrasing your question
- Being more specific about what information you need
- Checking if you mentioned valid stock tickers

I'm here to help with financial analysis, stock information, market insights, and investment guidance."""


# Create singleton instance
planner_agent = PlannerAgent()