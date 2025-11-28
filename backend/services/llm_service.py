import aiohttp
import json
from backend import config


async def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call the configured LLM endpoint and return text content.

    Uses `backend.config.LLAMA_API_URL` and respects `HTTP_TIMEOUT_SECONDS`.
    """
    LLAMA_API_URL = getattr(config, "LLAMA_API_URL", "http://localhost:8001/v1/completions")

    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["</s>"]
    }

    timeout = aiohttp.ClientTimeout(total=getattr(config, "HTTP_TIMEOUT_SECONDS", 10))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(LLAMA_API_URL, json=payload) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise Exception(f"LLM server error {resp.status}: {text}")

            try:
                data = json.loads(text)
                if isinstance(data, dict) and "choices" in data:
                    return data["choices"][0].get("text", "")
                if isinstance(data, dict) and "text" in data:
                    return data["text"]
            except Exception:
                return text

            return str(data)
