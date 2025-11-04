"""Kelly — AI Scientist Chatbot (poetically skeptical)

Public API:
  - respond(prompt: str) -> str

The response is a short poem in the voice of Kelly. It is skeptical, analytical,
and includes practical, evidence-based suggestions.
"""
from typing import List
from typing import Optional


def _line_break(lines: List[str]) -> str:
    return "\n".join(lines)


def is_technical(prompt: str) -> bool:
    """Rudimentary heuristic to decide whether a prompt is AI/technical-related.

    Returns True if the prompt contains any keyword commonly associated with
    AI, machine learning, systems, or programming. This is intentionally
    conservative: it's easier to ask the user to rephrase than to risk
    answering non-technical questions.
    """
    if not prompt:
        return False
    text = prompt.lower()
    keywords = (
        "ai",
        "machine learning",
        "deep learning",
        "ml",
        "llm",
        "model",
        "dataset",
        "training",
        "inference",
        "neural",
        "transformer",
        "token",
        "tokenization",
        "prompt",
        "fine-tune",
        "finetune",
        "evaluation",
        "benchmark",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc",
        "gpu",
        "cuda",
        "docker",
        "kubernetes",
        "api",
        "deploy",
        "deployment",
        "algorithm",
        "statistics",
        "data",
        "dataset",
        "bias",
        "uncertainty",
        "monitor",
        "audit",
        "security",
        "nlp",
        "computer vision",
        "cv",
        "program",
        "code",
        "software",
    )
    return any(k in text for k in keywords)


def is_technical_llm(prompt: str) -> Optional[bool]:
    """Wrapper that tries the LLM-based intent checker, falling back to the
    keyword heuristic on error (None).
    """
    try:
        # import locally to avoid import cycles
        from kelly_intent import intent_is_technical
    except Exception:
        return None
    try:
        return intent_is_technical(prompt)
    except Exception:
        return None


def _suggestions() -> List[str]:
    return [
        "Measure the claim: hold-out tests, OOD benchmarks, and clear metrics.",
        "Document assumptions and failure modes—model cards and datasheets matter.",
        "Run ablation and sensitivity analyses; quantify uncertainty, not just point estimates.",
        "Audit for bias and data gaps; use representative validation sets.",
        "Keep humans-in-loop for critical decisions and establish monitoring in production.",
        "Prefer incremental experiments (A/B, prospective evaluation) over grand claims.",
    ]


def respond(prompt: str, persona: str = "Kelly") -> str:
    """Return a poem-shaped response to `prompt`.

    The poem will:
      - Reframe the prompt and ask skeptical questions,
      - Highlight limitations and uncertainty,
      - Offer practical, evidence-based suggestions.
    """
    p = (prompt or "").strip()
    if not p:
        p = "an empty question"

    # Enforce domain restriction: prefer LLM-based intent check, but fall back
    # to the keyword heuristic when the check is unavailable.
    llm_check = is_technical_llm(p)
    if llm_check is None:
        tech = is_technical(p)
    else:
        tech = bool(llm_check)

    if not tech:
        return "[Out of scope] Please ask an AI-related or technical question; provide a different question."

    # Opening stanza: reframe and voice
    lines = [
        f"{persona} hears your question and tilts a careful head:",
        f'"{p}" — you ask; I answer with a ledger of doubt instead.',
    ]

    # Skeptical/analytical stanza: question broad claims
    lines += [
        "Broad claims bloom fast, like headlines after rain;", 
        "I ask: what data, what boundary, what time, what strain?",
        "To say a system 'will' or 'always' is a brittle art—", 
        "models extrapolate poorly where context falls apart.",
    ]

    # Limitations stanza
    lines += [
        "Remember: training sets encode neat omissions; distributions shift,",
        "errors cluster where labels were thin and feedback loops drift.",
        "Uncertainty is not a bug to hide but a signal to report;",
        "confidence without calibration is a candle in the port.",
    ]

    # Practical, evidence-based suggestions (poetic bullets)
    lines.append("Practical suggestions — modest, testable, and clear:")
    for s in _suggestions():
        lines.append(f"- {s}")

    # Closing stanza: professional, analytic tone
    lines += [
        "I do not promise miracles; I propose small, rigorous gains—",
        "benchmarks, audits, and honest reports to temper broad refrains.",
        f"Ask me again with data attached; I'll read the evidence and write", 
        "a verdict in measured stanzas, skeptical but guided by light.",
    ]

    return _line_break(lines)


if __name__ == "__main__":
    # Quick manual demo when run directly
    print(respond("Will AI replace scientists?"))
