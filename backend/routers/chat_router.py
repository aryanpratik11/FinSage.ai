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
        print("User query received")
        print(f"User Query: {query}")
        print("Passing query to Planner Agent...")

        # Call planner agent asynchronously
        response = await planner.handle_query(query)

        print("Planner Agent response received successfully.")
        return {"response": response}

    except Exception as e:
        print(f"Error in chat_with_agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
