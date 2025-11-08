# backend/agents/data_agent.py
"""
Data agent — handles structured/unstructured data retrieval and RAG logic.
"""

from backend.services import rag_service

def handle_data_query(query: str):
    """
    Uses RAG (Retrieval-Augmented Generation) to search across local or online data.
    """
    response = rag_service.query_documents(query)
    return response
