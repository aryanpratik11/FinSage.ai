"""
LLM Service — connects FinSage AI to Groq's free LLMs (e.g. Llama-3.1-70B).

FIXES:
- Added async context manager cleanup
- Better error handling
- Added timeout
- Response validation
"""

import os
import aiohttp
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found in .env file")


async def call_llm(
    prompt: str, 
    model: str = None, 
    temperature: float = 0.3, 
    max_tokens: int = 800,
    system_message: Optional[str] = None
) -> str:
    """
    Calls Groq API asynchronously and returns the model's response text.
    
    Args:
        prompt: User prompt
        model: Model name (default: llama-3.1-8b-instant)
        temperature: Creativity (0.0-1.0)
        max_tokens: Max response length
        system_message: Optional system prompt
        
    Returns:
        Model response text
        
    Raises:
        Exception: If API call fails
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured")
    
    model = model or DEFAULT_MODEL
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Build messages array
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error ({response.status}): {error_text}")
                
                res = await response.json()
                
                # Validate response structure
                if "choices" not in res or not res["choices"]:
                    raise Exception(f"Invalid response format: {res}")
                
                content = res["choices"][0]["message"]["content"]
                
                if not content:
                    raise Exception("Empty response from Groq API")
                
                return content.strip()
                
    except aiohttp.ClientError as e:
        raise Exception(f"Network error calling Groq API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error calling LLM: {str(e)}")
