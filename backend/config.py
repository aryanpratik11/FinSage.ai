"""Simple configuration module for FinSage backend.

Uses environment variables with sensible defaults for local development.
"""
import os

from typing import List

# LLM endpoint (expects a local Llama / proxy). Can be overridden via environment.
LLAMA_API_URL = os.environ.get("LLAMA_API_URL", "http://localhost:8001/v1/completions")

# Allowed CORS origins for the FastAPI app (comma-separated)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# HTTP client settings
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HTTP_TIMEOUT_SECONDS", "10"))
HTTP_RETRY_COUNT = int(os.environ.get("HTTP_RETRY_COUNT", "2"))

# Other feature flags / toggles
ENABLE_PREDICTION_SERVICE = os.environ.get("ENABLE_PREDICTION_SERVICE", "false").lower() in ("1", "true", "yes")

def get_allowed_origins() -> List[str]:
	return ALLOWED_ORIGINS or ["*"]
