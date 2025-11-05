# backend/services/rag_service.py
"""
RAG Service — retrieves relevant documents or context for reasoning.
"""

def retrieve_relevant_docs(query: str):
    """
    Mock RAG retriever — replace with vector database search later.
    """
    sample_docs = [
        f"Report: {query} has seen steady performance over the last quarter.",
        f"Analysis: Experts predict moderate growth for {query} in the coming fiscal year.",
        f"Insight: The market trend for {query} aligns with overall sector stability."
    ]

    return "\n".join(sample_docs)
