# backend/routers/chat_router.py

from fastapi import APIRouter
from backend.services.parser import extract_ticker_or_name, detect_intent
from backend.agents import planner_agent, data_agent, news_agent, prediction_agent, reasoning_agent

router = APIRouter()

@router.post("/")
async def chat_with_agent(user_query: dict):
    """
    Main chatbot endpoint for intelligent financial query processing.
    Expects JSON body:
    {
        "query": "Should I invest in HDFC Bank?"
    }
    """

    query = user_query.get("query", "")
    if not query:
        return {"error": "Query cannot be empty"}

    # Step 1: Parse query → extract name/ticker and intent
    parsed = extract_ticker_or_name(query)
    intent = detect_intent(query)

    # Step 2: Generate high-level plan using LLM (planner agent)
    plan = planner_agent.generate_plan(query, parsed, intent)

    # Step 3: Execute plan
    data = None
    news = None
    prediction = None

    # 3a. Fetch company/fund data if needed
    if any(k in plan.lower() for k in ["fundamental", "details", "financials"]):
        data = data_agent.fetch_data(parsed)

    # 3b. Get related news summaries
    if "news" in plan.lower():
        news = news_agent.get_recent_news(parsed)

    # 3c. Predict risk or price
    if "predict" in plan.lower() or "risk" in plan.lower():
        prediction = prediction_agent.predict_future(parsed)

    # Step 4: Reasoning agent composes final answer
    final_answer = reasoning_agent.compose_answer(
        query=query,
        plan=plan,
        data=data,
        news=news,
        prediction=prediction,
    )

    # Step 5: Return combined output
    return {
        "user_query": query,
        "intent_detected": intent,
        "parsed_entities": parsed,
        "plan_generated": plan,
        "retrieved_data": data,
        "summarized_news": news,
        "prediction": prediction,
        "final_answer": final_answer,
    }
