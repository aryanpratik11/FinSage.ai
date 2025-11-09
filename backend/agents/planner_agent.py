# backend/agents/planner_agent.py
"""
Planner Agent — orchestrates multi-agent workflow for FinSage AI.
"""

from backend.services.llm_service import call_llm
"""from backend.agents import (
    data_agent,
    document_agent,
    calculation_agent,
    reasoning_agent,
)"""
import asyncio


class PlannerAgent:
    def __init__(self):
        self.name = "Planner Agent"

    async def handle_query(self, query: str) -> dict:
        """
        Main entry point for orchestrating agents.
        """
        try:
            # Step 1: Understand query
            intent = await self.analyze_intent(query)
            print(f"Determined intent: {intent}")

            # Step 2: Determine agent flow
            subtasks = self.plan_subtasks(intent)
            print(f"Planned subtasks: {subtasks}")

            # Step 3: Execute agents in order (or parallel)
            results = await self.execute_agents(subtasks, query)

            # Step 4: Synthesize final response
            final_answer = await self.synthesize_response(query, results)

            return {"response": final_answer, "agents_used": list(results.keys())}

        except Exception as e:
            return {"error": str(e)}

    async def analyze_intent(self, query: str) -> str:
        """
        Uses LLM to analyze what the user wants.
        """
        prompt = f"""
        Analyze this financial query and describe its intent in one short phrase.
        Query: "{query}"
        """
        intent = await call_llm(prompt)
        return intent.strip()

    def plan_subtasks(self, intent: str):
        """
        Based on the intent, decide which agents to call.
        """
        subtasks = []

        if "financial data" in intent or "price" in intent:
            subtasks.append("data_agent")

        if "document" in intent or "filing" in intent or "report" in intent:
            subtasks.append("document_agent")

        if "calculate" in intent or "ratio" in intent or "valuation" in intent:
            subtasks.append("calculation_agent")

        # Default: reasoning_agent for analysis
        subtasks.append("reasoning_agent")

        return subtasks

    async def execute_agents(self, subtasks, query):
        """
        Executes selected agents asynchronously.
        """
        results = {}

        async def run_agent(name):
            if name == "data_agent":
                print("Data Agent called")
                results[name] = await data_agent.get_financial_data(query)
                print(f"Data Agent result: {results[name]}")
            elif name == "document_agent":
                print("Document Agent called")
                results[name] = await document_agent.extract_information(query)
                print(f"Document Agent result: {results[name]}")
            elif name == "calculation_agent":
                print("Calculation Agent called")
                results[name] = await calculation_agent.compute_metrics(query)
                print(f"Calculation Agent result: {results[name]}")
            elif name == "reasoning_agent":
                print("Reasoning Agent called")
                results[name] = await reasoning_agent.analyze_context(query, results)
                print(f"Reasoning Agent result: {results[name]}")

        await asyncio.gather(*(run_agent(t) for t in subtasks))
        return results

    async def synthesize_response(self, query, results):
        """
        Combines all agent outputs into a single LLM-generated summary.
        """
        context = "\n\n".join(
            [f"{agent.upper()}:\n{data}" for agent, data in results.items()]
        )
        prompt = f"""
        You are FinSage AI — a multi-agent financial assistant.
        User Query: "{query}"

        Use the following agent outputs to synthesize a professional, factual, and sourced response:

        {context}

        Make sure to include:
        - Key insights
        - Cited data points
        - Summary conclusion
        """
        result= await call_llm(prompt)
        print(f"Synthesized response: {result}")
        return result


planner_agent = PlannerAgent()