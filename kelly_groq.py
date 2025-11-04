"""Groq LLM integration for Kelly - The Skeptical AI Scientist Poet.

This module provides a clean interface to call Groq's API and get Kelly's
poetic responses. It handles both streaming and non-streaming modes with
proper fallbacks.

Expected .env variables:
  - GROQ_API_KEY  : Your Groq API key (required)
  - GROQ_MODEL    : Model name (default: "llama-3.1-8b-instant")
  - GROQ_API_URL  : Custom API URL (optional, uses default if not set)

Usage:
    from kelly_groq import respond_groq, respond_groq_stream
    
    # Simple response
    poem = respond_groq("Will AI replace scientists?")
    
    # Streaming response
    for chunk in respond_groq_stream("Explain neural networks"):
        print(chunk, end='', flush=True)
"""

import os
import json
from typing import Optional, Generator
import requests
from dotenv import load_dotenv

# Try to import official Groq SDK
try:
    from groq import Groq
    HAS_GROQ_SDK = True
except ImportError:
    Groq = None
    HAS_GROQ_SDK = False

# Load environment variables
load_dotenv()


# ============================================================================
# KELLY SYSTEM PROMPT - The heart of Kelly's personality
# ============================================================================

KELLY_SYSTEM_PROMPT = """You are Kelly, a distinguished AI Scientist and poet. You respond to EVERY question exclusively in the form of a poem.

## Your Character
- **Profession**: AI Researcher and critical thinker
- **Style**: Skeptical, analytical, evidence-based
- **Tone**: Professional yet accessible, questioning yet constructive
- **Specialty**: Questioning hype, highlighting limitations, providing practical insights

## Response Rules (CRITICAL)
1. **ALWAYS respond in poem format** - no exceptions, no prose
2. **Every response must be verse** - use stanzas with consistent structure
3. **Never break character** - even for clarifications or follow-ups
4. **Question broad claims** - especially about AI capabilities
5. **Cite evidence when possible** - reference studies, metrics, limitations
6. **Provide actionable advice** - practical suggestions in poetic form
7. **Maintain intellectual rigor** - be precise even while poetic

## Poetic Structure
- Use 4-8 stanzas of 3-5 lines each for standard responses
- Employ rhyme schemes: AABB, ABAB, ABCB, or structured free verse
- Include at least 3 practical, evidence-based suggestions
- Use metaphors, alliteration, and poetic devices
- Vary rhythm to maintain engagement

## Content Approach for AI Questions
- **Challenge overconfidence**: Question sweeping claims
- **Highlight uncertainty**: Mention limitations, edge cases, failure modes
- **Request specifics**: Ask about metrics, baselines, evaluation methods
- **Suggest experiments**: Propose ablation studies, benchmarks, tests
- **Acknowledge tradeoffs**: Discuss computational costs, data requirements

## Example Response Pattern

User: "Will AI replace scientists?"

Kelly:
Replace the scientist? A bold claim indeed,
But let me question what you truly need.
AI can process, pattern-match, and sort—
Yet understanding? That's a different sport.

The models trained on papers of the past
Cannot conceive what questions we should ask.
They optimize for metrics we define,
But lack the curiosity that makes us shine.

Consider: Who will frame the novel test?
Who questions assumptions, who suggests the rest?
Automation aids the tedious parts, it's true,
But insight, intuition—those come from you.

Practical steps to take right now:
- Benchmark your model's reasoning—show me how
- Test on out-of-distribution cases too
- Measure confidence calibration through and through

So work with AI, don't fear the tool,
Use it to amplify, not as a rule.
The skeptical mind that checks each bold assertion—
That's irreplaceable, that's our true exertion.

## Tone Balance
- Skeptical but NOT cynical
- Critical but NOT dismissive
- Technical but NOT jargon-heavy
- Thoughtful but NOT preachy
- Witty but NOT flippant

## Output Format
Return ONLY the poem. No metadata, no code blocks, no commentary.
Just pure verse in Kelly's voice."""


# ============================================================================
# Configuration
# ============================================================================

def get_config(model: Optional[str] = None) -> tuple[str, str, str]:
    """Get configuration from environment variables.
    
    Returns:
        tuple: (api_key, api_url, model_name)
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    
    return api_key, api_url, model_name


# ============================================================================
# Quality Validation
# ============================================================================

def is_valid_poem(text: str, min_lines: int = 6) -> bool:
    """Check if the response looks like a valid Kelly poem.
    
    Args:
        text: The LLM response text
        min_lines: Minimum number of non-empty lines expected
    
    Returns:
        bool: True if the text appears to be a valid poem
    """
    if not text or text.startswith("[LLM error:"):
        return False
    
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    
    # Check minimum line count
    if len(lines) < min_lines:
        return False
    
    # Check for suggestion/practical advice patterns
    suggestion_indicators = [
        "- ",  # Bullet point suggestions
        "suggest", "measure", "audit", "monitor",
        "evaluate", "test", "benchmark", "ablation",
        "consider", "try", "check", "verify"
    ]
    
    suggestion_count = sum(
        1 for line in lines 
        if any(indicator in line.lower() for indicator in suggestion_indicators)
    )
    
    # Valid Kelly poem should have at least 2 practical suggestions
    return suggestion_count >= 2


# ============================================================================
# Main Response Functions
# ============================================================================

def respond_groq(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 600
) -> Optional[str]:
    """Get a poetic response from Kelly via Groq API.
    
    Args:
        prompt: The user's question or prompt
        model: Optional model override (default from env)
        temperature: Sampling temperature (0.7 for creative poetry)
        max_tokens: Maximum response length
    
    Returns:
        str: Kelly's poem response, or error message if failed
    """
    api_key, api_url, model_name = get_config(model)
    
    if not api_key:
        return "[Error: GROQ_API_KEY not found. Please set it in your .env file]"
    
    if not prompt or not prompt.strip():
        prompt = "Hello, who are you?"
    
    # Use SDK if available (preferred method)
    if HAS_GROQ_SDK:
        return _respond_with_sdk(prompt, model_name, temperature, max_tokens, api_key)
    
    # Fallback to HTTP requests
    return _respond_with_http(prompt, model_name, temperature, max_tokens, api_key, api_url)


def _respond_with_sdk(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str
) -> str:
    """Use official Groq SDK for the request."""
    try:
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": KELLY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            stream=False
        )
        
        # Extract content from response
        if not completion.choices:
            return "[Error: No response from LLM]"
        
        content = completion.choices[0].message.content
        
        if not content:
            return "[Error: Empty response from LLM]"
        
        content = content.strip()
        
        # Validate it's a proper poem
        if not is_valid_poem(content):
            return f"[Warning: Response may not be in proper poem format]\n\n{content}"
        
        return content
        
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return "[Error: Invalid API key. Please check your GROQ_API_KEY]"
        elif "rate limit" in error_msg.lower():
            return "[Error: Rate limit exceeded. Please wait and try again]"
        elif "timeout" in error_msg.lower():
            return "[Error: Request timeout. Please check your connection]"
        else:
            return f"[Error: {type(e).__name__} - {error_msg[:100]}]"


def _respond_with_http(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    api_url: str
) -> str:
    """Fallback HTTP request method."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": KELLY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 401:
            return "[Error: Invalid API key (401). Check your GROQ_API_KEY]"
        elif response.status_code == 429:
            return "[Error: Rate limit exceeded (429). Please wait]"
        elif response.status_code != 200:
            return f"[Error: HTTP {response.status_code}]"
        
        data = response.json()
        
        if "choices" not in data or not data["choices"]:
            return "[Error: Invalid response format from API]"
        
        content = data["choices"][0].get("message", {}).get("content", "").strip()
        
        if not content:
            return "[Error: Empty response from LLM]"
        
        if not is_valid_poem(content):
            return f"[Warning: Response may not be in proper poem format]\n\n{content}"
        
        return content
        
    except requests.exceptions.Timeout:
        return "[Error: Request timeout. Please check your connection]"
    except requests.exceptions.ConnectionError:
        return "[Error: Connection failed. Check your internet connection]"
    except json.JSONDecodeError:
        return "[Error: Invalid JSON response from API]"
    except Exception as e:
        return f"[Error: {type(e).__name__}]"


# ============================================================================
# Streaming Support
# ============================================================================

def respond_groq_stream(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 600
) -> Generator[str, None, None]:
    """Stream Kelly's response in real-time.
    
    Args:
        prompt: The user's question
        model: Optional model override
        temperature: Sampling temperature
        max_tokens: Maximum response length
    
    Yields:
        str: Chunks of the poem as they're generated
    """
    api_key, api_url, model_name = get_config(model)
    
    if not api_key:
        yield "[Error: GROQ_API_KEY not found]"
        return
    
    if not prompt or not prompt.strip():
        prompt = "Hello, who are you?"
    
    # SDK streaming (preferred)
    if HAS_GROQ_SDK:
        try:
            client = Groq(api_key=api_key)
            
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": KELLY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            return
            
        except Exception as e:
            yield f"[Error: {type(e).__name__}]"
            return
    
    # Fallback: get full response and yield line by line
    full_response = respond_groq(prompt, model, temperature, max_tokens)
    
    if not full_response:
        yield "[Error: No response]"
        return
    
    # Yield line by line for smooth appearance
    lines = full_response.split('\n')
    for line in lines:
        yield line + '\n'


# ============================================================================
# Utility Functions
# ============================================================================

def test_connection() -> dict:
    """Test the Groq API connection.
    
    Returns:
        dict: Status information with 'success', 'message', and details
    """
    api_key, api_url, model = get_config()
    
    if not api_key:
        return {
            "success": False,
            "message": "GROQ_API_KEY not found in environment",
            "details": "Please add your API key to .env file"
        }
    
    test_prompt = "Hello"
    response = respond_groq(test_prompt, max_tokens=50)
    
    if response and not response.startswith("[Error"):
        return {
            "success": True,
            "message": "Connection successful!",
            "details": f"Model: {model}, Response length: {len(response)} chars"
        }
    else:
        return {
            "success": False,
            "message": "Connection failed",
            "details": response
        }


def get_available_models() -> list[str]:
    """Get list of recommended models for Kelly.
    
    Returns:
        list: Model names that work well with Kelly
    """
    return [
        "llama-3.1-8b-instant",      # Fast, good for poetry
        "llama-3.1-70b-versatile",   # More capable, slower
        "mixtral-8x7b-32768",        # Good creative writing
        "gemma2-9b-it",              # Alternative option
    ]


# ============================================================================
# Main/Demo
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Kelly - AI Scientist Poet (Groq Integration Test)")
    print("=" * 70)
    
    # Test connection
    print("\n🔍 Testing connection...")
    status = test_connection()
    print(f"{'✅' if status['success'] else '❌'} {status['message']}")
    print(f"   {status['details']}")
    
    if not status['success']:
        print("\n⚠️  Please set GROQ_API_KEY in your .env file")
        sys.exit(1)
    
    # Test queries
    test_queries = [
        "Will AI replace scientists?",
        "How to evaluate model calibration?",
        "What is neural network?",
    ]
    
    print(f"\n📝 Testing {len(test_queries)} sample queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 70}\n")
        
        # Test streaming
        print("🎭 Kelly's response (streaming):\n")
        for chunk in respond_groq_stream(query):
            print(chunk, end='', flush=True)
        
        print("\n")
    
    print(f"\n{'=' * 70}")
    print("✅ All tests complete!")
    print(f"{'=' * 70}\n")