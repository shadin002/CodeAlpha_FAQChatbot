from pathlib import Path

import streamlit as st

from src.chatbot import FAQChatbot


BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "data" / "faqs.json"

st.set_page_config(
    page_title="CodeAlpha FAQ Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        opacity: 0.75;
        margin-bottom: 1.4rem;
    }
    .info-box {
        padding: 0.8rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_chatbot() -> FAQChatbot:
    return FAQChatbot(FAQ_FILE)


try:
    chatbot = load_chatbot()
except Exception as exc:
    st.error(f"Could not start the chatbot: {exc}")
    st.stop()


st.markdown('<div class="main-title">🤖 CodeAlpha FAQ Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">NLP + TF-IDF + Cosine Similarity</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project Controls")
    threshold = st.slider(
        "Minimum match confidence",
        min_value=0.10,
        max_value=0.60,
        value=0.24,
        step=0.01,
        help="Higher values make the chatbot more cautious.",
    )

    show_debug = st.checkbox(
        "Show NLP match details",
        value=True,
        help="Useful while demonstrating how the chatbot chooses an FAQ.",
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Built for CodeAlpha AI Internship Task 2 using NLTK, "
        "scikit-learn, TF-IDF, cosine similarity, and Streamlit."
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    st.markdown(
        """
        <div class="info-box">
        <b>Try asking:</b><br>
        • How many tasks do I need to complete?<br>
        • Where should I upload my source code?<br>
        • What should my GitHub repository be named?<br>
        • Do I need to post a project video?<br>
        • What happens if I submit only one task?
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("details") and show_debug:
            details = message["details"]
            with st.expander("How this answer was matched"):
                confidence = details["confidence"]
                st.write(f"**Similarity score:** {confidence:.3f}")
                if details.get("matched_question"):
                    st.write(f"**Closest FAQ:** {details['matched_question']}")
                st.write(
                    "**Match status:** "
                    + ("Accepted" if details["is_match"] else "Below threshold")
                )


user_question = st.chat_input("Ask a question about the internship...")

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    result = chatbot.get_response(
        user_question,
        threshold=threshold,
        suggestion_count=3,
    )

    response_text = result["answer"]

    if not result["is_match"] and result.get("suggestions"):
        response_text += "\n\n**You could ask:**"
        for suggestion in result["suggestions"]:
            response_text += f"\n- {suggestion}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "details": result,
        }
    )

    st.rerun()
