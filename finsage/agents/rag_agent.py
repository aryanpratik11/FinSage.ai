"""RAGAgent: simple retrieval-augmented generation support.

Implements document ingestion (fetch filing HTML/text), embedding via
sentence-transformers, and an in-memory vector store with cosine similarity.

Notes:
- Requires `sentence-transformers` for embeddings. If not installed, the agent
  returns an error message instead of raising at import time.
- This is a minimal, dependency-light RAG implementation suitable for experiments.
"""
from typing import Dict, Any, List, Tuple
import requests
import os

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    np = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


class RAGAgent:
    name = "RAGAgent"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index: List[Dict[str, Any]] = []
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def _fetch_text(self, url: str) -> str:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            text = r.text
            if BeautifulSoup is not None:
                soup = BeautifulSoup(text, "html.parser")
                # remove scripts/styles
                for s in soup(["script", "style"]):
                    s.decompose()
                return soup.get_text(separator=" ", strip=True)
            return text
        except Exception:
            return ""

    def ingest_urls(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest a list of documents with keys {'id','url','source'} and index their embeddings."""
        if SentenceTransformer is None or np is None:
            return {"error": "sentence-transformers or numpy not installed. Install with 'pip install sentence-transformers numpy'"}

        if self.model is None:
            # attempt to load
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                return {"error": f"failed to load embedding model: {e}"}

        new_items = 0
        for d in docs:
            url = d.get("url")
            doc_id = d.get("id") or d.get("url")
            text = self._fetch_text(url)
            if not text:
                continue
            # simple chunking: split by paragraphs into smaller passages
            passages = [p.strip() for p in text.split("\n") if p.strip()][:30]
            for i, p in enumerate(passages):
                emb = self.model.encode(p)
                self.index.append({"id": f"{doc_id}#p{i}", "text": p, "embedding": emb, "source": url})
                new_items += 1

        return {"ingested": new_items, "index_size": len(self.index)}

    def _cosine_sim(self, a: Any, b: Any) -> float:
        if np is None:
            return 0.0
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve top_k passages relevant to query."""
        if SentenceTransformer is None or np is None:
            return {"error": "sentence-transformers or numpy not installed. Install with 'pip install sentence-transformers numpy'"}
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                return {"error": f"failed to load embedding model: {e}"}

        q_emb = self.model.encode(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.index:
            score = self._cosine_sim(q_emb, item["embedding"])
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ {"score": s, "id": it["id"], "text": it["text"], "source": it.get("source")} for s, it in scored[:top_k] ]
        return {"query": query, "top_k": top_k, "results": top}
