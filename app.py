import streamlit as st
from dotenv import load_dotenv

from src.ingestion import load_document
from src.chunking import split_text_into_chunks
from src.embeddings import load_embedding_model
from src.retriever import create_vector_store, retrieve_documents
from src.generator import load_llm

load_dotenv(override=True)

st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Chatbot")
st.markdown("### RAG-based Document Question Answering using Groq, FAISS & HuggingFace")

with st.sidebar:
    st.header("📌 About")
    st.write("""
    This AI Document Chatbot uses:
    - Multi-format document ingestion
    - RAG Architecture
    - FAISS Vector Database
    - HuggingFace Embeddings
    - Groq LLM
    - Streamlit Interface
    """)

embeddings = load_embedding_model()
llm = load_llm()

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt", "html"]
)

if uploaded_file is not None:

    st.success("Document uploaded successfully!")
    st.info(f"Uploaded file: {uploaded_file.name}")

    with st.spinner("Processing document and creating embeddings..."):
        document_text = load_document(uploaded_file)
        chunks = split_text_into_chunks(document_text)
        vector_store = create_vector_store(chunks, embeddings)

    st.subheader("Text Processing")
    st.write(f"Total chunks created: {len(chunks)}")
    st.success("Vector database created successfully!")

    st.divider()
    st.subheader("Ask a Question")

    question = st.text_input(
        "Ask anything from the uploaded document:",
        placeholder="Example: What is the main topic of this document?"
    )

    st.caption("""
    Example Questions:
    - Summarize this document
    - What are the key points mentioned?
    - What projects are included?
    - Explain the main topic
    """)

    if question:
        docs = retrieve_documents(vector_store, question, k=3)

        context = ""

        for i, doc in enumerate(docs):
            context += f"\n[Chunk {i + 1}]\n{doc.page_content}\n"

        prompt = f"""
        You are a document question-answering assistant.

        Answer the question using ONLY the provided context.
        If the answer is not present in the context, say:
        "The information is not found in the uploaded document."

        Mention the relevant chunk number if possible.

        Context:
        {context}

        Question:
        {question}
        """

        with st.spinner("Generating AI response..."):
            response = llm.invoke(prompt)

        st.success("Answer Generated Successfully!")

        st.subheader("AI Response")

        st.markdown(
            f"""
            <div style="
                background-color:#1e293b;
                padding:20px;
                border-radius:10px;
                border:1px solid #334155;
                line-height:1.6;
            ">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("🔍 Show Retrieved Chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"### Chunk {i + 1}")
                st.write(doc.page_content)

        st.divider()
        st.caption("Built with Streamlit, LangChain, HuggingFace, FAISS & Groq")