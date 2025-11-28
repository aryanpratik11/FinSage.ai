# backend/utils/cache.py
import time
from typing import Any, Dict

class SimpleCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.store: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str):
        rec = self.store.get(key)
        if not rec:
            return None
        value, expiry = rec
        if expiry < time.time():
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: Any):
        self.store[key] = (value, time.time() + self.ttl)

cache = SimpleCache()
