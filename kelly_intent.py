"""LLM-based intent checker for Kelly.

Provides a function `intent_is_technical(prompt)` which queries the configured
Groq endpoint and asks the model to classify whether the prompt is AI/technical.
Returns True/False on clear classification, or None if the check failed.

This intentionally uses a minimal, safe protocol: we ask the model to reply
with one word: TECHNICAL or NON-TECHNICAL to simplify parsing.
"""
import os
import json
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


def _get_config():
    model = os.getenv("GROQ_MODEL", "mistral-7b")
    api_key = os.getenv("GROQ_API_KEY")
    api_url = os.getenv("GROQ_API_URL")
    if not api_url:
        api_url = f"https://api.groq.ai/v1/models/{model}/chat/completions"
    return api_key, api_url


SYSTEM = (
    "You are a concise classifier. Decide whether the user's prompt is about AI, "
    "machine learning, data, models, programming, deployment, or other technical topics.\n"
    "Return exactly one word: TECHNICAL or NON-TECHNICAL. Do not add any explanation."
)


def intent_is_technical(prompt: str, timeout: float = 6.0) -> Optional[bool]:
    """Return True if model classifies prompt as technical, False if non-technical,
    or None if the check failed or was ambiguous.
    """
    if not prompt:
        return False

    api_key, api_url = _get_config()
    if not api_key:
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    try:
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        body = resp.json()
    except Exception:
        return None

    choices = body.get("choices") or []
    if not choices:
        return None
    first = choices[0]
    msg = (first.get("message") or {})
    text = (msg.get("content") or first.get("text") or "").strip().upper()
    if text.startswith("TECHNICAL"):
        return True
    if text.startswith("NON-TECHNICAL") or text.startswith("NONTECHNICAL"):
        return False
    # accept simple Y/N variants
    if text.startswith("YES") or text.startswith("Y"):
        return True
    if text.startswith("NO") or text.startswith("N"):
        return False
    return None
