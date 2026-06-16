import streamlit as st
import os
from rag_pipeline import load_and_index_pdf, load_existing_index, get_qa_chain

st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #0f1117;
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #1a1d27;
            border-right: 1px solid #2e3250;
        }
        [data-testid="stChatMessage"] {
            background-color: #1e2130;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 8px;
        }
        [data-testid="stChatInput"] {
            background-color: #1e2130;
            border-radius: 12px;
            border: 1px solid #2e3250;
        }
        [data-testid="stFileUploader"] {
            background-color: #1e2130;
            border-radius: 12px;
            border: 1px dashed #4a4f7a;
            padding: 10px;
        }
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        h1 {
            color: #7c83fd;
            font-size: 2rem;
            font-weight: 700;
        }
        hr {
            border-color: #2e3250;
        }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 PDF ChatBot")
    st.markdown("---")
    st.markdown("### 📂 Upload your PDF")

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        os.makedirs("docs", exist_ok=True)
        pdf_path = os.path.join("docs", uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("🔍 Indexing your PDF..."):
            # Clear old vectorstore and chain from memory
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.messages = []

            # Build fresh index
            vectorstore = load_and_index_pdf(pdf_path)
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = get_qa_chain(vectorstore)
            st.session_state.pdf_name = uploaded_file.name

        st.success(f"✅ Ready! Chatting with **{uploaded_file.name}**")

    if "pdf_name" in st.session_state:
        st.markdown("---")
        st.markdown(f"📄 **Active PDF:**\n\n`{st.session_state.pdf_name}`")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Built with LangChain + Groq + ChromaDB")

# ── Session State Init ────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main Area ─────────────────────────────────────────────
st.markdown("# 💬 Chat with your PDF")
st.markdown("Upload a PDF from the sidebar, then ask anything about it.")
st.markdown("---")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Welcome hint
if not st.session_state.messages:
    st.info("👈 Upload a PDF from the sidebar to get started!")

# ── Chat Input ────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your PDF..."):
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                source_docs = result["source_documents"]

                st.markdown(answer)

                # Show source pages
                if source_docs:
                    with st.expander("📄 Source Pages"):
                        seen = set()
                        for doc in source_docs:
                            page = doc.metadata.get("page", None)
                            if page is not None and page not in seen:
                                seen.add(page)
                                st.markdown(f"- **Page {page + 1}**")
                                st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})