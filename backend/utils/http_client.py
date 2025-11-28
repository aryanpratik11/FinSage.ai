# backend/utils/http_client.py
import aiohttp
import asyncio
from typing import Any, Dict, Optional

class HttpClient:
    def __init__(self, timeout: int = 10, retries: int = 2, backoff: float = 0.5):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_json(self, url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
        session = await self._get_session()
        attempt = 0
        while True:
            try:
                async with session.get(url, params=params, headers=headers, timeout=self.timeout) as resp:
                    text = await resp.text()
                    try:
                        return {"status": resp.status, "data": await resp.json()}
                    except Exception:
                        return {"status": resp.status, "text": text}
            except Exception as e:
                attempt += 1
                if attempt > self.retries:
                    return {"error": str(e)}
                await asyncio.sleep(self.backoff * attempt)

    async def post_json(self, url: str, json_body: dict, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
        session = await self._get_session()
        attempt = 0
        while True:
            try:
                async with session.post(url, json=json_body, headers=headers, timeout=self.timeout) as resp:
                    text = await resp.text()
                    try:
                        return {"status": resp.status, "data": await resp.json()}
                    except Exception:
                        return {"status": resp.status, "text": text}
            except Exception as e:
                attempt += 1
                if attempt > self.retries:
                    return {"error": str(e)}
                await asyncio.sleep(self.backoff * attempt)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

http_client = HttpClient()
