from fastapi import APIRouter, HTTPException
from backend.agents.planner_agent import PlannerAgent

# Initialize the planner agent
planner = PlannerAgent()

router = APIRouter()

@router.post("/")
async def chat_with_agent(request: dict):
    """
    Main chatbot endpoint.
    Expects JSON body:
    {
        "query": "Tell me about HDFC Mutual Fund"
    }
    """
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        print(f"Chat: User query received - {query}")
        print("Passing query to Planner Agent...")

        response = await planner.handle_query(query)

        print(f"Chat: Planner Agent response received: {response}")
        return {"response": response}

    except Exception as e:
        print(f"Chat: Error in chat_with_agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
