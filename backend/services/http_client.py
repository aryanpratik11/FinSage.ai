import asyncio
from typing import Any, Optional
import aiohttp
from . import llm_service
from backend import config

async def async_get(url: str, params: Optional[dict] = None, timeout: Optional[int] = None) -> Any:
    timeout = timeout or config.HTTP_TIMEOUT_SECONDS
    retry = config.HTTP_RETRY_COUNT

    for attempt in range(retry + 1):
        try:
            t = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=t) as session:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await resp.json()
                    return await resp.text()
        except Exception as e:
            if attempt >= retry:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
