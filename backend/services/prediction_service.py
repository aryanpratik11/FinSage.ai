# backend/services/prediction_service.py
"""
Prediction Service — connects with ML models to predict financial outcomes.
"""

import random

def predict_financial_outcome(query: str):
    """
    Mock prediction function — replace with real model inference later.
    """
    scenarios = ["Positive Trend", "Neutral Trend", "Negative Trend"]
    prediction = random.choice(scenarios)
    
    return {
        "symbol": query.upper(),
        "predicted_trend": prediction,
        "confidence": f"{round(random.uniform(70, 99), 2)}%"
    }
