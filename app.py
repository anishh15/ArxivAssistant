"""
Streamlit UI for ArxivAssistant.
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from loaders import load_arxiv_docs
from rag_chain import build_vectorstore, get_rag_chain, ask_question

# Load environment variables (HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="ArxivAssistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Header gradient (fallback when SVG banner missing) */
    .main-header {
        background: linear-gradient(135deg, #0f0a2e 0%, #1a0a38 50%, #0f0a2e 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(124, 77, 255, 0.3);
        border: 1px solid rgba(224, 64, 251, 0.15);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(124, 77, 255, 0.12) 0%, transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .main-header p {
        color: #CE93D8;
        font-size: 1.15rem;
        font-weight: 300;
        margin: 0.8rem 0 0 0;
        position: relative;
        z-index: 1;
    }

    /* Source badges */
    .source-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-arxiv {
        background: rgba(124, 77, 255, 0.15);
        color: #B388FF;
        border: 1px solid rgba(124, 77, 255, 0.3);
        box-shadow: 0 0 10px rgba(124, 77, 255, 0.1);
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(145deg, #ffffff, #f5f0ff);
        border: 1px solid #e2d9f3;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(124, 77, 255, 0.08);
        color: #1e293b;
    }
    @media (prefers-color-scheme: dark) {
        .answer-card {
            background: linear-gradient(145deg, #1a1030, #0f0a2e);
            border: 1px solid rgba(124, 77, 255, 0.2);
            color: #f8fafc;
            box-shadow: 0 4px 20px rgba(124, 77, 255, 0.15);
        }
    }

    /* Chat message styling */
    .user-msg {
        background: rgba(124, 77, 255, 0.08);
        border-left: 4px solid #7C4DFF;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.8rem 0;
        font-weight: 500;
    }
    .ai-msg {
        background: rgba(224, 64, 251, 0.06);
        border-left: 4px solid #E040FB;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.8rem 0;
        line-height: 1.6;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(124, 77, 255, 0.04);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(124, 77, 255, 0.1);
    }
    @media (prefers-color-scheme: dark) {
        .sidebar-section {
            background: rgba(124, 77, 255, 0.06);
            border: 1px solid rgba(124, 77, 255, 0.12);
        }
    }

    /* Stats pill — purple theme */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(124, 77, 255, 0.10);
        color: #7C4DFF;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(124, 77, 255, 0.20);
        margin: 6px 4px;
        box-shadow: 0 2px 10px rgba(124, 77, 255, 0.08);
    }

    /* Custom Input fields styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid rgba(124, 77, 255, 0.3);
    }
    .stTextInput > div > div > input:focus {
        border-color: #7C4DFF;
        box-shadow: 0 0 8px rgba(124, 77, 255, 0.2);
    }

    /* Button hover animation */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(124, 77, 255, 0.3);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_topic" not in st.session_state:
    st.session_state.indexed_topic = ""

# ─── Header ───────────────────────────────────────────────────────────

import os, base64
_banner_path = os.path.join(os.path.dirname(__file__), "assets", "banner.svg")
try:
    with open(_banner_path, "rb") as f:
        _b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="margin-bottom:1.5rem;border-radius:16px;overflow:hidden;'
        f'box-shadow:0 8px 32px rgba(124,77,255,0.25);">'
        f'<img src="data:image/svg+xml;base64,{_b64}" style="width:100%;display:block;" />'
        f'</div>',
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ArxivAssistant</h1>
        <p>AI-Powered Research Paper Q&A • Fetch papers dynamically, read smartly, and research faster.</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Search Controls")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # Using markdown to show we are searching Arxiv exclusively
    st.markdown("**Source:** `Arxiv.org` (Live Search)")

    topic = st.text_input(
        "🔍 Research Topic or Paper ID",
        placeholder="e.g., Attention is all you need, or 1706.03762",
        help="Enter a paper title for title-specific search, or paste an Arxiv ID (e.g. 1706.03762) for exact fetch",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Fetch & Index button
    fetch_clicked = st.button(
        "🚀 Fetch & Index Papers",
        use_container_width=True,
        type="primary",
        disabled=not topic.strip(),
    )

    if fetch_clicked and topic.strip():
        with st.status("🔄 Fetching & indexing papers...", expanded=True) as status:
            # Step 1: Load documents
            st.write(f"📥 Fetching from **Arxiv** for: *{topic}*")
            docs = load_arxiv_docs(topic)

            if not docs:
                status.update(label="❌ No papers found", state="error")
                st.error("No papers were found on Arxiv for that topic. Try a broader search term.")
            else:
                st.write(f"✅ Fetched **{len(docs)}** paper(s)")

                # Step 2: Build vector store
                st.write("🧩 Splitting & embedding content...")
                vectorstore = build_vectorstore(docs)
                st.write(f"✅ Created vector store with embedded chunks")

                # Step 3: Build RAG chain
                st.write("🔗 Building RAG chain...")
                chain = get_rag_chain(vectorstore)

                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = chain
                st.session_state.documents = docs
                st.session_state.indexed_topic = topic
                st.session_state.chat_history = []  # Reset chat for new topic

                status.update(label="✅ Papers indexed — Ready to answer!", state="complete")

    # Show indexed documents info
    if st.session_state.documents:
        st.markdown("---")
        st.markdown("## 📄 Current Context")
        st.markdown(
            f'<span class="stat-pill">📌 Topic: {st.session_state.indexed_topic}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="stat-pill">📃 {len(st.session_state.documents)} paper(s) loaded</span>',
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        for i, doc in enumerate(st.session_state.documents):
            source_type = doc.metadata.get("source_type", "Arxiv")
            title = doc.metadata.get("Title", doc.metadata.get("title", f"Paper {i+1}"))
            
            st.markdown(
                f'<div style="margin-bottom: 8px;"><span class="source-badge badge-arxiv">{source_type}</span> <strong>{title}</strong></div>',
                unsafe_allow_html=True,
            )

# ─── Main Area: Q&A ──────────────────────────────────────────────────

if st.session_state.chain:
    # Chat history display
    if st.session_state.chat_history:
        st.markdown("### 💬 Discussion")
        for entry in st.session_state.chat_history:
            st.markdown(
                f'<div class="user-msg"><strong>🧑 You:</strong> {entry["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="ai-msg"><strong>🤖 Assistant:</strong> {entry["answer"]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"📑 Sources used ({len(entry['sources'])} chunks)"):
                for j, src in enumerate(entry["sources"]):
                    src_type = src.metadata.get("source_type", "Arxiv")
                    title = src.metadata.get("Title", src.metadata.get("title", "Untitled"))
                    st.markdown(f"**Chunk {j+1}** — *{src_type}: {title}*")
                    st.caption(f"> {src.page_content}")
        st.markdown("---")

    # Question input
    question = st.text_input(
        "💡 Ask a question about the loaded papers",
        placeholder="e.g., What is the main contribution of these papers? How is the architecture designed?",
        key="question_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("🔎 Ask", type="primary", use_container_width=True)

    if ask_clicked and question.strip():
        with st.spinner("🧠 Analyzing papers..."):
            result = ask_question(st.session_state.chain, question)

        # Display answer
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.markdown(f"**🤖 Answer:**\n\n{result['answer']}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display sources
        if result["source_documents"]:
            with st.expander(f"📑 View source chunks ({len(result['source_documents'])} retrieved)"):
                for j, src in enumerate(result["source_documents"]):
                    src_type = src.metadata.get("source_type", "Arxiv")
                    title = src.metadata.get("Title", src.metadata.get("title", "Untitled"))
                    st.markdown(f"**Chunk {j+1}** — *{src_type}: {title}*")
                    st.caption(f"> {src.page_content}")

        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["source_documents"],
        })

        # Rerun to update chat history display
        st.rerun()

else:
    # No documents indexed yet — show welcome with logo
    st.markdown("---")
    _logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")
    try:
        with open(_logo_path, "rb") as f:
            _logo_b64 = base64.b64encode(f.read()).decode()
        _logo_html = f'<img src="data:image/svg+xml;base64,{_logo_b64}" style="width:120px;height:120px;margin-bottom:1.5rem;" />'
    except FileNotFoundError:
        _logo_html = '<div style="font-size:5rem;margin-bottom:1.5rem;">🔬</div>'

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 4rem 1rem;">
            {_logo_html}
            <h2 style="color: #7C4DFF; margin-bottom: 0.8rem; font-weight: 700;">Welcome to ArxivAssistant</h2>
            <p style="color: #64748b; font-size: 1.1rem; line-height: 1.6;">
                Navigate the sea of research papers smoothly. Enter a topic in the sidebar and click
                <strong>"Fetch & Index Papers"</strong> to launch your AI co-pilot.
            </p>
            <div style="margin-top: 2.5rem; display: flex; flex-wrap: wrap; justify-content: center; gap: 12px;">
                <span class="stat-pill">🔬 Dynamically Fetch Arxiv Papers</span>
                <span class="stat-pill">🧠 FAISS Vector Search</span>
                <span class="stat-pill">🤖 Free HuggingFace LLM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
