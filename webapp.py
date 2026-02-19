"""
Simple web chat UI for `backend.core.run_llm` using Streamlit.

How it works:
1) User enters a question in the chat input.
2) UI calls `run_llm(question)` from backend/core.py.
3) Answer is shown in chat.
4) Retrieved document sources are shown in an expandable section.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def extract_sources(context_docs: List[Any]) -> List[str]:
    """
    Extract unique source URLs/labels from returned context documents.

    `run_llm` returns `context` as a list of Document-like objects.
    Each Document usually has `metadata["source"]`.
    """
    unique_sources: List[str] = []
    seen = set()

    for doc in context_docs or []:
        source = "Unknown source"
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = str(doc.metadata.get("source", source))

        if source not in seen:
            seen.add(source)
            unique_sources.append(source)

    return unique_sources


def ask_backend(query: str) -> Dict[str, Any]:
    """
    Lazy import backend core so import-time setup errors can be surfaced in UI.
    """
    from backend.core import run_llm

    return run_llm(query)


def render_chat_message(message: Dict[str, Any]) -> None:
    """Render one message and its optional sources."""
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"- {source}")


def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="LangChain Docs Chat", page_icon="💬", layout="wide")
    st.title("LangChain Docs Chat")
    st.caption("RAG chat using backend/core.py")

    with st.sidebar:
        st.subheader("Runtime Config")
        st.write(f"`OLLAMA_BASE_URL`: `{os.getenv('OLLAMA_BASE_URL', 'not set')}`")
        st.write(f"`PINECONE_INDEX_NAME`: `{os.getenv('PINECONE_INDEX_NAME', 'not set')}`")
        st.write(f"`INDEX_NAME`: `{os.getenv('INDEX_NAME', 'not set')}`")

    # Session state keeps chat history across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about LangChain docs.",
                "sources": [],
            }
        ]

    for msg in st.session_state.messages:
        render_chat_message(msg)

    prompt = st.chat_input("Ask your question...")
    if not prompt:
        return

    # 1) Show/store user message.
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_chat_message(user_msg)

    # 2) Get assistant answer from backend.
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = ask_backend(prompt)
                answer = str(result.get("answer", "")).strip() or "No answer generated."
                sources = extract_sources(result.get("context", []))
            except Exception as exc:
                answer = (
                    "Error running backend RAG pipeline.\n\n"
                    "Check `.env` (Ollama/Pinecone settings) and backend connectivity.\n\n"
                    f"Details: `{exc}`"
                )
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"- {source}")

    # 3) Persist assistant message for future reruns.
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def _is_running_under_streamlit() -> bool:
    """
    Detect whether this file is executed by `streamlit run`.
    """
    # `st.runtime.exists()` is True when Streamlit runtime is active.
    # Unlike `get_script_run_ctx()`, this check avoids noisy bare-mode warnings.
    return bool(st.runtime.exists())


if __name__ == "__main__":
    # Prevent noisy warnings if someone runs `python webapp.py` directly.
    # Streamlit apps should be launched with `streamlit run`.
    if _is_running_under_streamlit():
        main()
    else:
        script_name = Path(__file__).name
        print(
            "This is a Streamlit app.\n\n"
            "Run it with:\n"
            f"  streamlit run {script_name}\n\n"
            "or:\n"
            f"  {sys.executable} -m streamlit run {script_name}"
        )
