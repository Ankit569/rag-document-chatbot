import streamlit as st
import os
import tempfile
from rag import build_vectorstore, load_vectorstore, get_answer, index_exists

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #7F77DD;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 6px;
        font-size: 13px;
        color: #444;
    }
    .stChatMessage { padding: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 DocChat")
st.caption("Upload a PDF and ask anything about it — powered by Mistral AI + RAG")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    show_sources = st.toggle("Show source chunks", value=True)
    st.caption("Shows which part of the document the answer came from.")

    st.divider()
    st.markdown("**Model:** `mistral-small-latest`")
    st.markdown("**Embeddings:** `mistral-embed`")
    st.markdown("**Vector store:** FAISS")
    st.markdown("**Framework:** LangChain")

    st.divider()

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset document"):
        for key in ["vectorstore", "messages", "doc_name"]:
            if key in st.session_state:
                del st.session_state[key]
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        st.rerun()

# ── Document upload ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type="pdf",
    help="Upload any PDF — research paper, resume, report, manual, etc."
)

if uploaded_file:
    doc_changed = st.session_state.get("doc_name") != uploaded_file.name

    if doc_changed:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📎 **{uploaded_file.name}** ready to process")
        with col2:
            process = st.button("Process", type="primary", use_container_width=True)

        if process:
            with st.spinner("Reading, splitting and indexing your document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    st.session_state.vectorstore = build_vectorstore(tmp_path)
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"✅ Document indexed! Ask your questions below.")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    os.unlink(tmp_path)
    else:
        st.success(f"✅ **{uploaded_file.name}** is loaded and ready.")

        if "vectorstore" not in st.session_state and index_exists():
            with st.spinner("Loading index..."):
                st.session_state.vectorstore = load_vectorstore()

# ── Chat interface ────────────────────────────────────────────────────────────
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("💬 Ask your questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("📎 Sources from document"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Page {src["page"]}</strong><br>{src["snippet"]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # New question input
    if question := st.chat_input("Ask something about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = get_answer(question, st.session_state.vectorstore)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.write(answer)

                    if show_sources and sources:
                        with st.expander("📎 Sources from document"):
                            for src in sources:
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<strong>Page {src["page"]}</strong><br>{src["snippet"]}...'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    err_msg = f"Error getting answer: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": []
                    })

else:
    st.info("👆 Upload a PDF above and click **Process** to get started.")

    st.divider()
    st.markdown("#### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**\nUpload any PDF document")
    with col2:
        st.markdown("**2. Process**\nDoc is split, embedded and indexed")
    with col3:
        st.markdown("**3. Ask**\nAsk questions, get grounded answers")
