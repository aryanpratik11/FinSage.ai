# backend/agents/prediction_agent.py
"""
Prediction Agent — uses ML models to predict financial trends or metrics.
"""

from backend.utils import prediction_service

def handle_prediction_query(query: str):
    """
    Calls the trained ML model or forecasting service to predict outcomes.
    """
    prediction = prediction_service.predict_financial_outcome(query)
    
    if not prediction:
        return "Unable to generate prediction at the moment."
    
    return {
        "query": query,
        "predicted_outcome": prediction
    }
