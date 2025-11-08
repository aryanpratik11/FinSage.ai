"""
LLM Service — Handles all interactions with OpenAI (or any LLM API).
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env (if not already loaded)
load_dotenv()

# Get API key safely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_response(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.5, max_tokens: int = 600) -> str:
    """
    Sends a prompt to the LLM and returns the model's text response.
    This is used by agents (e.g., planner_agent) to reason and plan.
    """

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a highly intelligent financial reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response_text = completion.choices[0].message.content.strip()
        return response_text

    except Exception as e:
        print(f"LLM Service Error: {e}")
        return "Error: Failed to generate response from LLM."


def generate_structured_json(prompt: str, model: str = "gpt-4o-mini", schema_description: str = "") -> dict:
    """
    Generates a structured JSON response from the LLM.
    Useful when we want data in a specific schema.
    """

    structured_prompt = f"""
    You are an assistant that outputs ONLY valid JSON (no explanations, no comments).
    Schema description: {schema_description}
    
    Task:
    {prompt}
    """

    response_text = generate_response(structured_prompt, model=model)

    # Attempt to parse into JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Handle partial or malformed JSON
        return {"raw_output": response_text, "error": "Could not parse LLM output as JSON."}
