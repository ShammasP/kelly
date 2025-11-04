"""Small CLI wrapper to run Kelly from the command line.

Usage:
  python run_kelly.py "Is AI an existential threat?"

If no argument provided, the script will prompt for input.
"""
import sys
import argparse
import os
from kelly_bot import respond as respond_template

try:
    # optional LLM modules
    from kelly_groq import respond_groq
except Exception:
    respond_groq = None


def main(argv):
    parser = argparse.ArgumentParser(description="Run Kelly, the AI Scientist poet.")
    parser.add_argument("prompt", nargs="*", help="Question to ask Kelly")
    parser.add_argument("--llm", choices=("groq", "template"), default=None,
                        help="Use LLM provider (groq) or local template (template). If omitted, auto-detects from environment.")
    parser.add_argument("--model", default=None, help="LLM model name (provider-specific)")
    args = parser.parse_args(argv[1:])

    prompt = " ".join(args.prompt).strip() if args.prompt else ""

    # Auto-detect provider: prefer groq when GROQ_API_KEY present
    provider = args.llm
    if provider is None:
        provider = "groq" if os.getenv("GROQ_API_KEY") else "template"

    if provider == "groq" and respond_groq is not None:
        poem = respond_groq(prompt, model=args.model)
        if poem is None:
            poem = "[LLM unavailable: set GROQ_API_KEY and ensure network access]"
        # If the groq module returned an error marker, show it directly
        if isinstance(poem, str) and poem.startswith("[LLM error:"):
            print(poem)
            return
    else:
        poem = "[Template mode disabled: use --llm template to enable]"
        print(poem)
        return

    print(poem)


if __name__ == "__main__":
    main(sys.argv)
