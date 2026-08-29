import json
import requests
import streamlit as st

import db_helper
import ingest

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "phi4-mini"
TOP_K = 3

SYSTEM_PROMPT = """You are a local RAG assistant. Answer ONLY based on the provided CONTEXT (retrieved chunks).
Rules:
- If the answer is not in the context, say: "This information is not available in the local knowledge base."
- Never invent facts or add information outside the context.
- Always answer in English, short and clear.
- When you use information from a chunk, cite its source at the end, e.g. (Source: filename.txt)."""

# 1. Page settings
st.set_page_config(page_title="Local RAG Assistant", page_icon="🧠", layout="wide")

# Custom theme / design
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .header {
        background: linear-gradient(135deg, #4f46e5, #9333ea);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .header h1 { color: white; margin: 0; font-size: 1.9rem; }
    .header p { color: #e9d5ff; margin: 0.2rem 0 0; font-size: 1rem; }
    .stChatMessage { border-radius: 12px; }
    [data-testid="stChatInput"] { border-radius: 12px; }
    .source-card {
        background: #f5f3ff;
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    .source-card b { color: #6d28d9; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🧠 Local RAG Assistant</h1>
    <p>Microsoft Phi-4 Mini + SQLite vector search — fully on-device, no cloud, no API keys.</p>
</div>
""", unsafe_allow_html=True)

# 2. Load embedding model (cached)
@st.cache_resource
def get_embedder():
    return ingest.get_embedder()

get_embedder()

# 3. Generate LLM answer via Ollama (streaming)
def strip_think(text):
    """Removes <think>...</think> (reasoning) blocks produced by the model."""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()

def generate_answer(user_query, chat_history):
    query_vector = get_embedder().encode(user_query).tolist()
    retrieved = db_helper.search_similar_chunks(query_vector, top_k=TOP_K)

    if not retrieved:
        return "No relevant information found in the database.", [], query_vector

    context_text = "\n".join([f"[Source: {src}]\n• {chunk}" for chunk, src, _ in retrieved])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {user_query}\n\nAnswer ONLY based on the context above."
    })

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.2},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
        resp.raise_for_status()
        answer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if data.get("message", {}).get("content"):
                answer += data["message"]["content"]
        return strip_think(answer), retrieved, query_vector
    except Exception as e:
        return f"Error: could not reach the Ollama server ({e})", retrieved, query_vector

# 4. Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 5. Screens
tab_chat, tab_upload = st.tabs(["💭 Chat", "📂 Upload Document"])

with tab_chat:
    # Message history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.chat_history and st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with tab_upload:
    st.subheader("📂 Upload a New Document")
    st.caption("Upload a `.txt` or `.md` file. It is immediately chunked, embedded and added to SQLite.")

    uploaded = st.file_uploader("Choose a file", type=["txt", "md"])

    if uploaded is not None:
        if st.button("Add to Database", type="primary"):
            content = uploaded.getvalue().decode("utf-8", errors="ignore")
            with st.spinner("Processing file..."):
                count = ingest.ingest_text(uploaded.name, content)
            st.success(f"✅ `{uploaded.name}` added successfully. {count} chunks saved.")

    st.markdown("---")
    st.subheader("🗄️ Indexed Sources")

    try:
        sources = db_helper.list_sources()
        if sources:
            for src, cnt in sources:
                st.write(f"📄 **{src}** — {cnt} chunks")
            st.metric("Total Chunks", db_helper.count_chunks())
        else:
            st.warning("No data yet. Run `python ingest.py` first or add a file above.")
    except Exception:
        st.warning("Database not found. Run `python ingest.py` first.")

# 6. Chat input bar (top-level → docks to the bottom of the screen)
user_input = st.chat_input("Ask a question about your knowledge base...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the vector database and generating the answer..."):
            answer, retrieved, _ = generate_answer(user_input, st.session_state.chat_history[:-1])

        st.markdown(answer)

        if retrieved:
            with st.expander(f"🔍 Sources Used ({len(retrieved)} retrieved chunks)", expanded=True):
                for chunk, src, score in retrieved:
                    st.markdown(
                        f"<div class='source-card'><b>📄 {src}</b> — similarity <b>{score:.3f}</b><br>"
                        f"<span style='color:#4b5563'>{chunk}</span></div>",
                        unsafe_allow_html=True
                    )

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
