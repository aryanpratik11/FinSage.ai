"""
LLM Service — connects FinSage AI to Groq’s free LLMs (e.g. Llama-3.1-70B).
"""

import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file")


async def call_llm(prompt: str, model: str = None, temperature: float = 0.3, max_tokens: int = 800) -> str:
    """
    Calls Groq API asynchronously and returns the model's response text.
    """
    model = model or DEFAULT_MODEL

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Groq API error ({response.status}): {error_text}")

            res = await response.json()
            try:
                return res["choices"][0]["message"]["content"]
            except Exception:
                raise Exception(f"Invalid response format: {res}")
