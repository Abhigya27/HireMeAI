import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def _get_backend_url() -> str:
    # Streamlit Community Cloud injects secrets via st.secrets, not plain env vars.
    # Falls back to .env / OS env var for local development.
    try:
        return st.secrets["BACKEND_URL"]
    except Exception:
        return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


BACKEND_URL = _get_backend_url()

# ---- Edit these with your real links ----
SOCIAL_LINKS = {
    "Email": ("✉️", "mailto:your_email@example.com"),
    "GitHub": ("💻", "https://github.com/your-username"),
    "LinkedIn": ("🔗", "https://linkedin.com/in/your-profile"),
    "Instagram": ("📸", "https://instagram.com/your-handle"),
}

st.set_page_config(page_title="Ask Abhigya", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("Abhigya Narain")
        st.caption("AI/ML Engineer")  # edit as needed
        st.divider()

        st.subheader("Connect")
        for label, (icon, url) in SOCIAL_LINKS.items():
            st.markdown(f"{icon} [{label}]({url})")

        st.divider()
        st.caption(f"Backend: {BACKEND_URL}")


# ---------------------------------------------------------------------------
# Chat tab (history-aware conversational RAG)
# ---------------------------------------------------------------------------
def init_chat_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def stream_chat_response(query: str):
    payload = {"session_id": st.session_state.session_id, "query": query}
    with httpx.stream("POST", f"{BACKEND_URL}/chat", json=payload, timeout=120.0) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                yield chunk


def render_chat_tab():
    """Renders the header, clear button, and scrollable message history.
    Returns the container so main() can stream new messages into the same
    visual box (st.chat_input has to live outside the tab to stay pinned to
    the bottom of the page — see main()).
    """
    header_col, clear_col = st.columns([5, 1])
    with header_col:
        st.subheader("Chat with Abhigya's AI")
    with clear_col:
        if st.button("Clear chat"):
            try:
                httpx.post(
                    f"{BACKEND_URL}/chat/clear",
                    params={"session_id": st.session_state.session_id},
                    timeout=10.0,
                )
            except httpx.HTTPError:
                pass
            st.session_state.messages = []
            st.rerun()

    # fixed-height container -> scrolls internally instead of growing the page
    chat_box = st.container(height=500)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    return chat_box


# ---------------------------------------------------------------------------
# JD Matcher tab
# ---------------------------------------------------------------------------
def score_band(score: float):
    if score >= 75:
        return "success", "Strong fit"
    elif score >= 50:
        return "warning", "Moderate fit"
    else:
        return "error", "Weak fit"


def render_jd_tab():
    st.subheader("Job Description Matcher")
    st.write("Upload a job description and see how well Abhigya's profile matches it.")

    uploaded = st.file_uploader("Upload job description", type=["txt", "pdf", "docx"])

    if uploaded and st.button("Check fit", type="primary"):
        with st.spinner("Analyzing match..."):
            try:
                files = {
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                }
                response = httpx.post(f"{BACKEND_URL}/match", files=files, timeout=60.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as e:
                st.error(f"Match request failed: {e}")
                return

        score = result.get("final_score") or 0
        level, label = score_band(score)

        st.metric("Fit Score", f"{score}/100")
        st.progress(min(max(score / 100, 0.0), 1.0))
        getattr(st, level)(f"**{label}** — {result.get('verdict', '')}")

        match_col, gap_col = st.columns(2)
        with match_col:
            st.markdown("**Matching skills**")
            for skill in result.get("matching_skills", []):
                st.markdown(f"- ✅ {skill}")
        with gap_col:
            st.markdown("**Missing skills**")
            for skill in result.get("missing_skills", []):
                st.markdown(f"- ❌ {skill}")

        with st.expander("Reasoning & score breakdown"):
            st.write(result.get("reasoning", ""))
            st.caption(
                f"Embedding similarity: {result.get('embedding_score')} | "
                f"LLM score: {result.get('llm_score')}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_chat_state()
    render_sidebar()
    st.title("Ask Abhigya 🤖")

    tab_jd, tab_chat = st.tabs(["📄 Job Description Matcher", "💬 Chat"])
    with tab_jd:
        render_jd_tab()
    with tab_chat:
        chat_box = render_chat_tab()

    # IMPORTANT: chat_input must be called at the root level (not nested inside
    # st.tabs) for Streamlit to pin it to the true bottom of the page. It will
    # stay visible across both tabs -- a known tradeoff of this pattern.
    query = st.chat_input("Ask a question about Abhigya...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    full_response = st.write_stream(stream_chat_response(query))
                except httpx.HTTPError as e:
                    full_response = f"Sorry, something went wrong talking to the backend: {e}"
                    st.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()