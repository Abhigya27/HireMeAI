import json
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
RESUME_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "Data", "abhigya.pdf")

# ---- Edit these with your real links ----
SOCIAL_LINKS = {
    "GitHub": ("💻", "https://github.com/Abhigya27"),
    "LinkedIn": ("🔗", "www.linkedin.com/in/abhigya-narain-11643b2b5"),
    "X": ("🐤", "https://x.com/AbhigyaNarain"),
    "LeetCode": ("👨‍💻", "https://leetcode.com/u/abhigya_27/"),
}

st.set_page_config(page_title="Ask Abhigya", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("Abhigya Narain")
        st.caption("AI Engineer")  # edit as needed
        if os.path.exists(RESUME_PATH):
            with open(RESUME_PATH, "rb") as resume_file:
                resume_data = resume_file.read()
            st.download_button(
                "Download Resume",
                data=resume_data,
                file_name="abhigya.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("Resume unavailable")
        st.divider()
        st.subheader("Email : narainabhigya27@gmail.com")
        st.subheader("Connect")
        for label, (icon, url) in SOCIAL_LINKS.items():
            st.markdown(f"{icon} [{label}]({url})")

        st.divider()


def render_nav():
    """Two buttons standing in for tabs. Not a layout container like st.tabs,
    so it doesn't block st.chat_input from pinning to the bottom of the page
    when the chat view is active -- and lets us skip rendering chat_input
    entirely when the JD Matcher view is active, instead of it floating over
    every view.
    """
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "💬 Chat",
            use_container_width=True,
            type="primary" if st.session_state.active_view == "chat" else "secondary",
        ):
            st.session_state.active_view = "chat"
            st.rerun()
    with col2:
        if st.button(
            "📄 Job Description Matcher",
            use_container_width=True,
            type="primary" if st.session_state.active_view == "jd" else "secondary",
        ):
            st.session_state.active_view = "jd"
            st.rerun()


# ---------------------------------------------------------------------------
# Chat tab (history-aware conversational RAG)
# ---------------------------------------------------------------------------
def init_app_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_view" not in st.session_state:
        st.session_state.active_view = "chat"  # which of the two sections is showing
    if "jd_result" not in st.session_state:
        st.session_state.jd_result = None
        st.session_state.jd_filename = None


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
    visual box -- chat_input itself is called separately in main(), directly
    in the script flow (not inside this function), so it stays pinned to the
    bottom of the page.
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


# must match backend/ats.py's RESULT_MARKER -- the two apps are separate
# processes talking over HTTP, so this is a small wire-format contract
# rather than a shared import.
JD_RESULT_MARKER = "<<<JD_MATCH_RESULT>>>"


def render_jd_tab():
    st.subheader("Job Description Matcher")
    st.write(
        "Upload a job description and see a field-by-field comparison against Abhigya's resume.")

    uploaded = st.file_uploader(
        "Upload job description", type=["txt", "pdf", "docx"])

    if uploaded and st.button("Check fit", type="primary"):
        status = st.status(
            "Comparing resume against the job description...", expanded=True)
        preview_box = status.empty()
        buffer = ""
        result = None
        try:
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                )
            }
            with httpx.stream("POST", f"{BACKEND_URL}/match", files=files, timeout=120.0) as response:
                response.raise_for_status()
                for chunk in response.iter_text():
                    if not chunk:
                        continue
                    buffer += chunk
                    if JD_RESULT_MARKER in buffer:
                        raw_part, _, result_part = buffer.partition(
                            JD_RESULT_MARKER)
                        result = json.loads(result_part)
                        buffer = raw_part
                        break
                    preview_box.code(buffer, language="json")
        except httpx.HTTPError as e:
            status.update(label="Match request failed", state="error")
            st.error(f"Match request failed: {e}")
            return
        except json.JSONDecodeError:
            status.update(label="Couldn't parse the result", state="error")
            st.error(
                "Something went wrong parsing the match result — please try again.")
            return

        if result is None:
            status.update(label="No result received", state="error")
            st.error("Didn't receive a complete result — please try again.")
            return

        if result.get("error"):
            # The backend couldn't recover a usable analysis from the model's
            # response (e.g. it got cut off mid-generation). Show that
            # plainly rather than a score quietly built from an unrelated
            # fallback signal -- and don't overwrite any previous good result.
            status.update(label="Analysis failed", state="error")
            st.error(result["error"])
            return

        status.update(label="Analysis complete",
                      state="complete", expanded=False)
        # persist in session_state -- a plain local variable would be lost
        # the moment the user switches views and comes back, since
        # st.button() only evaluates True on the exact rerun it was
        # clicked, not on every later rerun
        st.session_state.jd_result = result
        st.session_state.jd_filename = uploaded.name

    result = st.session_state.jd_result
    if not result:
        return  # nothing checked yet this session

    if st.session_state.jd_filename:
        st.caption(f"Showing results for: {st.session_state.jd_filename}")

    score = result.get("final_score") or 0
    level, label = score_band(score)

    st.metric("Fit Score", f"{score}/100")
    st.progress(min(max(score / 100, 0.0), 1.0))
    getattr(st, level)(f"**{label}** — {result.get('final_verdict', '')}")

    topic_verdicts = result.get("topic_verdicts", [])
    if topic_verdicts:
        st.markdown("**Field-by-field breakdown**")
        for item in topic_verdicts:
            st.markdown(
                f"- **{item.get('topic', '')}:** {item.get('verdict', '')}")

    transferable = result.get("transferable_skills", [])
    if transferable:
        st.markdown("**Transferable skills**")
        st.caption(
            "Skills the job asks for that don't appear by name, but have a genuine equivalent in the resume.")
        for t in transferable:
            st.markdown(
                f"- Job wants **{t.get('required', '')}** → resume has **{t.get('have_instead', '')}** "
                f"— {t.get('why_similar', '')}"
            )

    gaps = result.get("genuine_gaps", [])
    if gaps:
        st.markdown("**Genuine gaps**")
        for gap in gaps:
            severity = gap.get("severity", "") if isinstance(gap, dict) else ""
            gap_text = gap.get("gap", "") if isinstance(gap, dict) else gap
            icon = "❌" if severity == "must-have" else "⚠️"
            label = f"[{severity.replace('-', ' ').title()}] " if severity else ""
            st.markdown(f"- {icon} {label}{gap_text}")

    with st.expander("Score breakdown"):
        st.caption(
            f"LLM assessment score (drives the fit score above): {result.get('llm_score')} | "
            f"Whole-document embedding similarity (reference only, not used in the score): "
            f"{result.get('embedding_score')}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_app_state()
    render_sidebar()
    st.title("Ask Abhigya 🤖")
    st.caption("A personal AI chatbot and job-fit matcher — chat with Abhigya's resume, or check how well a job description matches his skills.")

    render_nav()
    st.divider()

    if st.session_state.active_view == "jd":
        render_jd_tab()
        return  # no chat_input on this view -- nothing left to do this run

    # --- Chat view ---
    chat_box = render_chat_tab()

    # Calling chat_input directly in the main script flow (not nested inside
    # st.tabs/st.columns/etc.) is what lets Streamlit pin it to the true
    # bottom of the page. Because it's only reached when active_view == "chat",
    # it also simply doesn't exist while the JD Matcher view is showing.
    query = st.chat_input("Ask a question about Abhigya...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    full_response = st.write_stream(
                        stream_chat_response(query))
                except httpx.HTTPError as e:
                    full_response = f"Sorry, something went wrong talking to the backend: {e}"
                    st.error(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
