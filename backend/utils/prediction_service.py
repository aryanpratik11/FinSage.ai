"""Compatibility shim: expose `predict_financial_outcome` under `backend.utils`.

This delegates to `backend.services.prediction_service` so code importing
from `backend.utils import prediction_service` continues to work.
"""
from backend.services import prediction_service as _svc


def predict_financial_outcome(query: str, horizon: str = None):
    return _svc.predict_financial_outcome(query, horizon=horizon)
