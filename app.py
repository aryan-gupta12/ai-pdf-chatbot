import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv(override=True)

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")
st.markdown("### RAG-based PDF Question Answering using Groq, FAISS & HuggingFace")

with st.sidebar:
    st.header("📌 About")
    st.write("""
    This AI PDF Chatbot uses:
    - RAG Architecture
    - FAISS Vector Database
    - HuggingFace Embeddings
    - Groq LLM
    - Streamlit Interface
    """)


def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)


uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf"
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")
    st.info(f"Uploaded file: {uploaded_file.name}")

    with st.spinner("Processing PDF and creating embeddings..."):
        pdf_text = extract_pdf_text(uploaded_file)
        chunks = split_text_into_chunks(pdf_text)
        vector_store = create_vector_store(chunks)

    st.subheader("Text Processing")
    st.write(f"Total chunks created: {len(chunks)}")
    st.success("Vector database created successfully!")

    st.divider()
    st.subheader("Ask a Question")

    question = st.text_input(
        "Ask anything from the uploaded PDF:",
        placeholder="Example: What is the main topic of this document?"
    )

    st.caption("""
    Example Questions:
    - Summarize this document
    - What are the key skills mentioned?
    - What projects are included?
    - Explain the main topic
    """)

    if question:

        docs = vector_store.similarity_search(question, k=3)

        context = ""

        for doc in docs:
            context += doc.page_content + "\n"

        prompt = f"""
        Answer the question based only on the provided context.
        If the answer is not available in the context, say that the information is not found in the uploaded PDF.

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

        st.divider()
        st.caption("Built with Streamlit, LangChain, HuggingFace & Groq")