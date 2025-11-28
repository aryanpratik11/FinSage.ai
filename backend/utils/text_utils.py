# backend/utils/text_utils.py
import json
import re
from typing import Any, Optional

def truncate(text: Any, limit: int = 1500) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[TRUNCATED]..."

def safe_json_extract(text: str) -> Optional[dict]:
    if not text:
        return None
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None

def clean_text(t: Any) -> str:
    if t is None:
        return ""
    return str(t).replace("\r", " ").replace("\t", " ").strip()
