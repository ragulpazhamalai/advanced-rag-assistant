import streamlit as st
import uuid
import rag

st.set_page_config(page_title="Advanced RAG Assistant", page_icon="🤖", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Advanced RAG")
    st.caption("Hybrid Search • Multi-Query • Parent Retrieval • Compression • Memory • Structured Output")
    st.divider()
    st.write("Session:", st.session_state.session_id[:8])

    if st.button("New chat",use_container_width=True,):
        # Create completely new RAG session
        st.session_state.session_id = str(uuid.uuid4())

         # Clear visible Streamlit chat
        st.session_state.messages = []

        st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        rag.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Advanced RAG Assistant")
st.caption("Ask questions from the indexed PDF knowledge base.")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hi! Ask me a question from the indexed documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        metadata = message.get("metadata")
        if metadata:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Confidence")
                st.write(metadata["confidence"].upper())
            with c2:
                st.caption("Sources")
                if metadata["sources"]:
                    for source in metadata["sources"]:
                        st.write(f"• {source}")
                else:
                    st.write("No supporting source")

            if metadata["key_quotes"]:
                with st.expander("Supporting evidence"):
                    for quote in metadata["key_quotes"]:
                        st.markdown(f"> {quote}")

            if metadata["follow_up_questions"]:
                with st.expander("Suggested follow-up questions"):
                    for item in metadata["follow_up_questions"]:
                        st.write(f"• {item}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Searching documents...", expanded=False):
            response = rag.ask(question, st.session_state.session_id)

        st.markdown(response.answer)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Confidence")
            if response.confidence == "high":
                st.success("HIGH")
            elif response.confidence == "medium":
                st.warning("MEDIUM")
            else:
                st.error("LOW")

        with c2:
            st.caption("Sources")
            if response.sources:
                for source in response.sources:
                    st.write(f"• {source}")
            else:
                st.write("No supporting source")

        if response.key_quotes:
            with st.expander("Supporting evidence"):
                for quote in response.key_quotes:
                    st.markdown(f"> {quote}")

        if response.follow_up_questions:
            st.subheader("Suggested questions")
            for follow_up in response.follow_up_questions:
                st.write(f"• {follow_up}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "metadata": {
            "confidence": response.confidence,
            "sources": response.sources,
            "key_quotes": response.key_quotes,
            "follow_up_questions": response.follow_up_questions,
        }
    })
