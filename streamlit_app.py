"""Streamlit chat-style UI for Kelly — a skeptical, analytical AI Scientist poet.

This app provides a GPT-like chat interface using Streamlit's chat components.
It keeps a small, configurable session memory (last N exchanges). The app
supports a local template responder and an optional Groq (Mistral 7B) provider
— the Groq call is used only when configured and successful; otherwise the
template responder is used as a fallback.

Run:
  streamlit run streamlit_app.py
"""
from typing import List, Dict
import streamlit as st
from datetime import datetime, timezone
import json

from kelly_bot import respond as respond_template

try:
    from kelly_groq import respond_groq, respond_groq_stream
except Exception:
    # Keep names present even if module import fails
    respond_groq = None
    respond_groq_stream = None


DEFAULT_MEMORY = 5


def init_state():
    if "history" not in st.session_state:
        # history: list of {role: 'user'|'assistant', content: str, ts: iso str}
        st.session_state.history = []
    if "memory_size" not in st.session_state:
        st.session_state.memory_size = DEFAULT_MEMORY
    if "use_llm_intent" not in st.session_state:
        # Use LLM-based intent checker by default (costly); user can toggle in settings
        st.session_state.use_llm_intent = True


def push_message(role: str, content: str):
    # Use timezone-aware UTC timestamp
    st.session_state.history.append({"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()})
    # Trim to last N exchanges (counted by messages)
    n = st.session_state.memory_size * 2  # each exchange is two messages
    if len(st.session_state.history) > n:
        st.session_state.history = st.session_state.history[-n:]


def render_chat():
    # Render chat messages in order using Streamlit chat components
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.code(msg["content"])


def main():
    st.set_page_config(page_title="Kelly — Chat", layout="wide")
    init_state()

    # Main title and description
    st.title("Kelly — AI Scientist (Poet)")
    st.markdown("_Skeptical, analytical, professional. Answers in poem form._")

    # Settings moved to sidebar for a cleaner main UI
    import os
    sidebar = st.sidebar
    sidebar.title("Settings")
    # Auto-detect: prefer groq when GROQ_API_KEY present in environment; otherwise template
    default_index = 1 if os.getenv("GROQ_API_KEY") else 0
    provider = sidebar.selectbox("Provider", options=["template", "groq"], index=default_index)
    st.session_state.memory_size = sidebar.number_input("Memory (last N exchanges)", min_value=1, max_value=50, value=st.session_state.memory_size)
    # Provider status
    if os.getenv("GROQ_API_KEY"):
        sidebar.success("Groq: API key detected")
    else:
        sidebar.info("Groq: no API key found — responses will use the template provider")
    # Intent-check toggle (LLM-based intent check costs one API call per prompt)
    st.session_state.use_llm_intent = sidebar.checkbox(
        "Use LLM-based intent checker (may incur extra cost)",
        value=st.session_state.get("use_llm_intent", True),
    )
    if sidebar.button("Clear memory"):
        st.session_state.history = []
        sidebar.success("Memory cleared.")
    # Diagnostics button: run a connectivity test to the Groq API
    if sidebar.button("Run diagnostics"):
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        api_key, api_url = None, os.getenv("GROQ_API_URL")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sidebar.warning("GROQ_API_KEY not set in environment (.env not loaded or missing)")
        else:
            sidebar.info("Running connectivity test to Groq endpoint...")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            body = {"messages": [{"role": "system", "content": "Connectivity test"}, {"role": "user", "content": "ping"}], "max_tokens": 5}
            try:
                r = requests.post(api_url, headers=headers, json=body, timeout=10)
                if r.status_code == 200:
                    sidebar.success(f"Connected: HTTP 200 OK — Groq endpoint reachable")
                else:
                    sidebar.error(f"Groq HTTP {r.status_code} — check key and endpoint")
            except Exception as e:
                sidebar.error(f"Request failed: {type(e).__name__}")
    # Samples and export in sidebar
    sidebar.markdown("---")
    sidebar.subheader("Quick prompts")
    sample = sidebar.selectbox(
        "Choose a sample prompt",
        options=[
            "Will AI replace scientists?",
            "How to evaluate model calibration?",
            "Explain prompt engineering best practices.",
        ],
    )
    if sidebar.button("Use sample"):
        st.session_state._sample_to_send = sample
    if st.session_state.history:
        sidebar.download_button(
            "Export conversation (JSON)",
            data=json.dumps(st.session_state.history, indent=2),
            file_name="kelly_conversation.json",
            mime="application/json",
        )

    st.markdown("---")

    # Chat area (clean main column). We'll render after any generation so
    # assistant replies appended during this run are visible immediately.

    # Bottom-fixed input: approximate fixed footer using CSS (Streamlit DOM classes)
    st.markdown(
        """
        <style>
        /* Ensure main content has enough bottom padding so chat is not hidden behind the fixed input */
        .main .block-container { padding-bottom: 160px; }
        /* Attempt to fix text inputs/areas to the bottom of the viewport */
        .stTextArea, .stTextInput { position: fixed !important; bottom: 16px; left: 16px; right: 160px; z-index: 9999; background: white; }
        /* Place the send button on the bottom-right */
        .stButton > button { position: fixed !important; bottom: 16px; right: 16px; z-index: 10000; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input widgets (rendered at the end but visually fixed by the CSS above)
    # Use a persistent key so we can clear after sending
    try:
        # Try to use chat_input when available (it renders as a text field)
        user_input = st.chat_input("Ask Kelly a question", key="kelly_chat_input")
    except Exception:
        user_input = st.text_area("Ask Kelly a question", key="kelly_text_area", height=80)

    # If a sample prompt was chosen in the sidebar, inject it into the input value
    if st.session_state.get("_sample_to_send"):
        # If using chat_input, push immediately; if using text_area, set its value via session_state
        sample = st.session_state.pop("_sample_to_send")
        push_message("user", sample)

    if user_input:
        push_message("user", user_input)

    # If last message is from user and no assistant reply yet, generate reply
    if st.session_state.history and st.session_state.history[-1]["role"] == "user":
        latest = st.session_state.history[-1]["content"]
        # Stream output while generating when possible
        if provider == "groq" and respond_groq_stream is not None:
            stream_placeholder = st.empty()
            accumulated = ""
            with st.spinner("Kelly is composing a poem..."):
                try:
                    for chunk in respond_groq_stream(latest):
                        # chunk may be a str; append and render incrementally
                        accumulated += chunk
                        stream_placeholder.code(accumulated)
                except Exception as e:
                    stream_placeholder.error(f"Streaming failed: {type(e).__name__}")
                    accumulated = f"[LLM error: streaming failed: {type(e).__name__}]"
            # Finalize: push assistant message and clear the placeholder
            push_message("assistant", accumulated)
            stream_placeholder.empty()
        else:
            with st.spinner("Kelly is composing a poem..."):
                poem = None
                if provider == "groq" and respond_groq is not None:
                    poem = respond_groq(latest)
                    if poem is None:
                        poem = "[LLM unavailable: check GROQ_API_KEY and network]"
                else:
                    poem = "[Template mode disabled: switch provider to 'template' to enable]"
            # Only push assistant message when a poem has been generated
            push_message("assistant", poem)

        # Render chat AFTER generation so newly appended assistant messages
        # appear in the current run instead of the next rerun.
        if not st.session_state.history:
            st.info("No messages yet. Use the sidebar to change settings or pick a quick prompt.")
        render_chat()


if __name__ == "__main__":
    main()
