import aiohttp
import json

LLAMA_API_URL = "http://localhost:8001/v1/completions"

async def call_llm(prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["</s>"]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(LLAMA_API_URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"LLM server error {resp.status}: {text}")

            data = await resp.json()
            return data["choices"][0]["text"]
