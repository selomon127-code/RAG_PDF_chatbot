# app.py
# Streamlit Chat UI for RAG PDF Chatbot — Professional Design
# Run with: streamlit run app.py

import os
import sys
import tempfile
import streamlit as st

# ============================================================================
# ✅ FIXED: Reliable .env loading for Windows
# ============================================================================
def load_env_reliably():
    """Load .env file with explicit path resolution for Windows compatibility."""
    try:
        from dotenv import load_dotenv
        
        # Get the directory where app.py lives (project root)
        project_root = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(project_root, ".env")
        
        # Try loading from explicit path first
        if os.path.exists(env_path):
            result = load_dotenv(dotenv_path=env_path, override=True)
            print(f"[app] ✅ Loaded .env from: {env_path} | Success: {result}")
            return True
        
        # Fallback: try current working directory
        cwd_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(cwd_env):
            result = load_dotenv(dotenv_path=cwd_env, override=True)
            print(f"[app] ✅ Loaded .env from CWD: {cwd_env} | Success: {result}")
            return True
        
        # Last resort: try default location
        result = load_dotenv(override=True)
        print(f"[app] ⚠️ Loaded .env from default location | Success: {result}")
        return result
        
    except Exception as e:
        print(f"[app] ❌ Failed to load .env: {e}")
        return False

# Load environment variables BEFORE any imports that depend on them
load_env_reliably()

# Add src/ to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules (now that env vars are loaded)
from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embedder import build_vectorstore, VectorStore
from src.rag_engine import answer_question

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Study Buddy • RAG PDF Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "# Study Buddy\nA RAG-powered PDF chatbot for students."
    }
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== GLOBAL RESETS & TYPOGRAPHY ===== */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
        color: #f1f5f9;
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stText,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        margin: 8px 0;
    }
    .status-ready {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
    }
    .status-waiting {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    /* ===== MAIN CHAT AREA ===== */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }
    
    /* Chat message bubbles */
    .chat-message {
        display: flex;
        flex-direction: column;
        margin: 12px 0;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        align-items: flex-end;
    }
    
    .assistant-message {
        align-items: flex-start;
    }
    
    .message-bubble {
        max-width: 85%;
        padding: 14px 18px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        position: relative;
    }
    
    .user-message .message-bubble {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message .message-bubble {
        background: white;
        color: #1e293b;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
    }
    
    /* Source citation badges */
    .source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 8px 0 0 18px;
    }
    
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        background: linear-gradient(135deg, #818cf8, #6366f1);
        color: white;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        transition: transform 0.15s ease;
    }
    .source-badge:hover {
        transform: scale(1.05);
        cursor: pointer;
    }
    
    /* ===== INPUT AREA ===== */
    .stChatInputContainer {
        background: white;
        border-radius: 16px;
        padding: 8px 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    
    .stChatInputContainer input {
        font-size: 15px !important;
    }
    
    /* ===== UPLOAD AREA ===== */
    .upload-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 2px dashed #cbd5e1;
        transition: all 0.2s ease;
        margin: 1rem 0;
    }
    .upload-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
    }
    
    /* ===== STATS CARD ===== */
    .stats-card {
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        border-left: 4px solid #3b82f6;
    }
    .stats-card h4 {
        margin: 0 0 8px 0;
        color: #1e293b;
        font-size: 14px;
        font-weight: 600;
    }
    .stats-card p {
        margin: 4px 0;
        color: #475569;
        font-size: 13px;
    }
    
    /* ===== LOADING SPINNER ===== */
    .thinking {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #64748b;
        font-size: 14px;
        padding: 12px 18px;
    }
    .thinking-dot {
        width: 8px;
        height: 8px;
        background: #3b82f6;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* ===== ERROR MESSAGE ===== */
    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #991b1b;
        font-size: 14px;
        margin: 12px 0;
    }
    
    /* ===== RESPONSIVE ADJUSTMENTS ===== */
    @media (max-width: 768px) {
        .message-bubble { max-width: 95%; }
        .main .block-container { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "embed_model": None,
        "vector_store": None,
        "chunks": None,
        "pdf_name": None,
        "pdf_ready": False,
        "llm_backend": "gemini",
        "llm_model": "gemini-2.0-flash-lite",
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# SIDEBAR: UPLOAD & SETTINGS
# ============================================================================
with st.sidebar:
    st.markdown("## 📚 Study Buddy")
    st.markdown("*Your AI-powered PDF study assistant*")
    st.markdown("---")
    
    # PDF Upload Card
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Textbooks, lecture notes, research papers — any text-based PDF",
        label_visibility="collapsed",
        key="pdf_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process Button
    if uploaded_file is not None:
        st.markdown(f"**File:** `{uploaded_file.name}`")
        
        if st.button("⚡ Process PDF", type="primary", use_container_width=True, disabled=st.session_state.processing):
            st.session_state.processing = True
            st.rerun()
        
        # Processing logic
        if st.session_state.processing:
            with st.status("🔄 Processing your document...", expanded=True) as status:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    st.write("📄 Extracting text from PDF...")
                    pages = load_pdf(tmp_path)
                    st.write(f"   ✅ Extracted **{len(pages)}** pages")
                    
                    st.write("✂️ Splitting into semantic chunks...")
                    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
                    st.write(f"   ✅ Created **{len(chunks)}** chunks")
                    
                    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
                    lengths = [len(t) for t in texts if t.strip()]
                    if lengths:
                        stats_text = f"Total: {len(lengths)} chunks\nAvg: {sum(lengths)//len(lengths)} chars\nMin: {min(lengths)} | Max: {max(lengths)}"
                        st.code(stats_text, language="text")
                    
                    st.write("🧠 Building semantic index...")
                    try:
                        embed_model, vector_store, saved_chunks = build_vectorstore(chunks)
                    except Exception as ve:
                        st.warning("⚠️ Indexing issue. Attempting recovery...")
                        vs_temp = VectorStore()
                        vs_temp.reset()
                        embed_model, vector_store, saved_chunks = build_vectorstore(chunks)
                        st.success("✅ Recovery successful!")
                    
                    st.write(f"   ✅ Indexed **{len(saved_chunks)}** chunks")
                    
                    st.session_state.embed_model = embed_model
                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = saved_chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_ready = True
                    st.session_state.messages = []
                    
                    status.update(label="✅ Document ready! Start asking questions.", state="complete", expanded=False)
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.info("💡 Try: 1) Delete `data/vectorstore/` folder 2) Restart app")
                    st.session_state.pdf_ready = False
                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    st.session_state.processing = False
                    st.rerun()
    
    st.markdown("---")
    
    # LLM Settings
    st.markdown("### ⚙️ AI Settings")
    
    llm_backend = st.selectbox(
        label="LLM Backend",
        options=["gemini", "ollama", "openai"],
        index=0,
        help="Gemini: Free & fast • Ollama: Local & private • OpenAI: Most capable (paid)"
    )
    st.session_state.llm_backend = llm_backend
    
    if llm_backend == "gemini":
        llm_model = st.selectbox(
            label="Gemini Model",
            options=["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest"],
            index=0,
            help="gemini-2.0-flash-lite is highly recommended to minimize rate limiting."
        )
    elif llm_backend == "ollama":
        llm_model = st.selectbox(
            label="Ollama Model",
            options=["llama3", "gemma2", "mistral", "llama3.1", "phi3"],
            index=0
        )
    else:  # openai
        llm_model = st.selectbox(
            label="OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        api_key = st.text_input(
            label="OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Get your key at platform.openai.com"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    st.session_state.llm_model = llm_model
    
    st.markdown("---")
    
    # Status Indicator
    if st.session_state.pdf_ready:
        st.markdown(f'<div class="status-badge status-ready">✓ Ready • {st.session_state.pdf_name}</div>', unsafe_allow_html=True)
        
        with st.expander("📊 API Quota Info", expanded=False):
            st.markdown("""
            **Gemini Free Tier Limits:**
            - 🔄 ~15 requests / minute
            - 📅 ~1,500 requests / day
            
            **If you see "quota exceeded":**
            1. Wait 60 seconds ⏱️
            2. Switch backend to **Ollama** for zero limits! 🚀
            """)
    else:
        st.markdown('<div class="status-badge status-waiting">⏳ No document loaded</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Utility Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            try:
                vs = VectorStore()
                vs.reset()
            except:
                pass
            for key in ["messages", "embed_model", "vector_store", "chunks", "pdf_name", "pdf_ready"]:
                st.session_state[key] = None if key != "messages" else []
            st.rerun()

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
st.markdown("## 💬 Ask Your Document Anything")

if not st.session_state.pdf_ready:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
        <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Ready to study smarter?</h3>
        <p style="margin: 0 0 1.5rem 0; font-size: 1.1rem;">
            Upload a PDF in the sidebar to get instant, citation-backed answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render all message logs up to current state
    for msg in st.session_state.messages:
        align_class = "user-message" if msg["role"] == "user" else "assistant-message"
        st.markdown(f"""
        <div class="chat-message {align_class}">
            <div class="message-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if msg.get("sources"):
            pages = sorted(set(s["page"] for s in msg["sources"] if s.get("page")))
            if pages:
                badges = "".join(f'<span class="source-badge">📄 p.{p}</span>' for p in pages[:5])
                if len(pages) > 5:
                    badges += f'<span class="source-badge">+{len(pages)-5} more</span>'
                st.markdown(f'<div class="source-container">{badges}</div>', unsafe_allow_html=True)

    # Simple Prompt Suggestions
    if len(st.session_state.messages) == 0:
        st.markdown('<div style="color: #64748b; font-size: 13px; margin: 12px 0 8px;">💡 Try asking:</div>', unsafe_allow_html=True)
        suggestions = ["Summarize key concepts", "What are the important definitions?", "What should I remember for exams?"]
        cols = st.columns(3)
        for idx, (col, suggestion) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion, "sources": []})
                    st.rerun()

    # Chat Input Box Handling
    if user_input := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_input.strip(), "sources": []})
        st.rerun()

    # Contextual evaluation loop when a new question is appended
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]
        
        with st.container():
            st.markdown("""
            <div class="thinking">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <span>Thinking...</span>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                result = answer_question(
                    question=user_msg,
                    model=st.session_state.embed_model,
                    index=st.session_state.vector_store,  # passing vector_store mapping object
                    chunks=st.session_state.chunks,
                    llm_backend=st.session_state.llm_backend,
                    llm_model=st.session_state.llm_model,
                )
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
            except Exception as e:
                answer = f"An execution error occurred: {str(e)}"
                sources = []
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 13px; border-top: 1px solid #e2e8f0; margin-top: 2rem;">
    Study Buddy • RAG PDF Assistant • Built for performance
</div>
""", unsafe_allow_html=True)