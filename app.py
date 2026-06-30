import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

load_dotenv()

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Chatbot")
st.caption("Upload a PDF and ask questions about it.")

# -----------------------
# Session state init
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# -----------------------
# Sidebar — PDF upload & settings
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    st.divider()
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 200, 4000, 2000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)

    st.subheader("Retrieval")
    top_k = st.slider("Top-k chunks", 1, 10, 3)

    st.subheader("Model")
    llm_model = st.selectbox(
        "LLM",
        ["mistral-small-2506", "mistral-medium-2505", "mistral-large-2411"],
    )

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# -----------------------
# Load & index PDF
# -----------------------
PERSIST_DIR = "chroma_db"

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes, filename, _chunk_size, _chunk_overlap):
    """Write uploaded bytes to a temp file, load, chunk, embed, store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    embedding = MistralAIEmbeddings(model="mistral-embed")

    persist_path = os.path.join(PERSIST_DIR, filename.replace(" ", "_"))
    if os.path.exists(persist_path) and os.listdir(persist_path):
        vs = Chroma(
            persist_directory=persist_path,
            embedding_function=embedding,
        )
    else:
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_path,
        )

    return vs, len(documents), len(chunks)


if uploaded_file:
    file_key = f"{uploaded_file.name}_{chunk_size}_{chunk_overlap}"

    if st.session_state.pdf_name != file_key:
        with st.spinner("Reading and indexing PDF…"):
            vs, n_pages, n_chunks = build_vectorstore(
                uploaded_file.read(),
                uploaded_file.name,
                chunk_size,
                chunk_overlap,
            )
        st.session_state.vectorstore = vs
        st.session_state.pdf_name = file_key
        st.session_state.chat_history = []

    with st.sidebar:
        st.success(f"✅ **{uploaded_file.name}** indexed")
        col1, col2 = st.columns(2)
        col1.metric("Pages", "—")
        col2.metric("Chunks", "—")
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")

# -----------------------
# Chat UI
# -----------------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                if hasattr(msg, "sources") and msg.sources:
                    with st.expander("📎 Sources"):
                        for src in msg.sources:
                            st.caption(src)

# -----------------------
# Chat input
# -----------------------
if question := st.chat_input(
    "Ask something about your PDF…",
    disabled=st.session_state.vectorstore is None,
):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat_history.append(HumanMessage(content=question))

    # Retrieve relevant chunks
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Source page references
    sources = sorted({
        f"Page {doc.metadata.get('page', 0) + 1}  —  {uploaded_file.name}"
        for doc in docs
    })

    # Build history string (last 3 turns)
    history_msgs = st.session_state.chat_history[:-1][-6:]
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history_msgs
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer ONLY using the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Chat history:
{history}

Question:
{question}
""")

    llm = ChatMistralAI(model=llm_model)
    chain = prompt | llm

    # Stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({
            "context": context,
            "question": question,
            "history": history_text,
        }):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        with st.expander("📎 Sources"):
            for src in sources:
                st.caption(src)

    # Save assistant message (attach sources for re-render)
    ai_msg = AIMessage(content=full_response)
    ai_msg.sources = sources
    st.session_state.chat_history.append(ai_msg)