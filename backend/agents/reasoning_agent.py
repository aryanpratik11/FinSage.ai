# backend/agents/reasoning_agent.py
"""
Reasoning Agent — combines RAG and LLM reasoning to produce the final answer.
"""

from backend.services import rag_service, llm_service

def handle_reasoning(query: str, context: str = ""):
    """
    Retrieves context via RAG and refines it using LLM reasoning.
    """
    # Step 1: Retrieve documents
    retrieved_docs = rag_service.retrieve_relevant_docs(query)
    
    # Step 2: Build combined reasoning prompt
    prompt = f"""
    User Query: {query}
    Retrieved Context:
    {retrieved_docs}

    Using the above information, generate a clear and well-structured answer.
    """

    response = llm_service.generate_response(prompt)
    
    return {
        "query": query,
        "response": response,
        "sources": retrieved_docs
    }
