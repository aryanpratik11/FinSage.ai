"""LLMAgent: wrapper for calling a language model to synthesize answers.

Primary: OpenAI ChatCompletion (requires OPENAI_API_KEY and `openai` package).
Fallback: Hugging Face `transformers` text-generation pipeline (best-effort).

The agent accepts a prompt and optional context passages; it returns the generated text
and any metadata about the call.
"""
from typing import Dict, Any, List, Optional
import os

try:
    import openai
except Exception:
    openai = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


class LLMAgent:
    name = "LLMAgent"

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.openai_available = openai is not None and os.environ.get("OPENAI_API_KEY")
        self.hf_available = pipeline is not None

    def _compose_prompt(self, user_query: str, passages: Optional[List[Dict[str, Any]]] = None) -> str:
        SYSTEM = (
            "You are FinSage, an assistant that answers financial queries using provided factual excerpts. "
            "Cite sources by url when appropriate and be concise."
        )
        ctx = SYSTEM + "\n\n"
        if passages:
            ctx += "Relevant excerpts:\n"
            for i, p in enumerate(passages, 1):
                ctx += f"[{i}] {p.get('text')[:500]} ... (source: {p.get('source')})\n\n"
        ctx += "User question: " + user_query + "\n\nAnswer succinctly and include citations like [1]."
        return ctx

    def run(self, user_query: str, passages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        prompt = self._compose_prompt(user_query, passages)

        # Prefer OpenAI Chat API
        if self.openai_available:
            try:
                openai.api_key = os.environ.get("OPENAI_API_KEY")
                # Use chat format
                messages = [{"role": "system", "content": "You are a helpful financial assistant."}, {"role": "user", "content": prompt}]
                res = openai.ChatCompletion.create(model=self.model, messages=messages, temperature=0.2, max_tokens=512)
                txt = res["choices"][0]["message"]["content"].strip()
                return {"model": self.model, "text": txt, "source": "openai"}
            except Exception as e:
                return {"error": f"OpenAI call failed: {e}"}

        # Fallback to Hugging Face text-generation
        if self.hf_available:
            try:
                gen = pipeline("text-generation", model="gpt2", device=-1)
                out = gen(prompt, max_length=512, do_sample=False)
                txt = out[0]["generated_text"]
                return {"model": "hf-gpt2", "text": txt, "source": "huggingface"}
            except Exception as e:
                return {"error": f"HF pipeline failed: {e}"}

        return {"error": "No LLM available. Set OPENAI_API_KEY and install openai, or install transformers for local fallback."}
