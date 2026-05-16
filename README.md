# 📄 AI PDF Chatbot

A RAG-based PDF question-answering chatbot built using Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Groq LLM.

## 🚀 Live Demo

[Click Here](https://aryan-ai-pdf-chatbot.streamlit.app/)
## 📌 Features

- Upload and process PDF documents
- Extract text from PDF files
- Split large documents into chunks
- Generate embeddings using HuggingFace
- Store embeddings in FAISS vector database
- Retrieve relevant document chunks using similarity search
- Generate AI answers using Groq LLM
- Interactive Streamlit web interface

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- PyPDF
- RAG

## 🧠 Project Concept

This project uses Retrieval-Augmented Generation (RAG).  
Instead of asking an LLM directly, the app first retrieves relevant information from the uploaded PDF and then generates an answer based on that retrieved context.

## ⚙️ How It Works

1. User uploads a PDF.
2. Text is extracted from the PDF.
3. Text is split into smaller chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in FAISS vector database.
6. User asks a question.
7. Relevant chunks are retrieved using similarity search.
8. Groq LLM generates an answer based on the retrieved context.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py

```

## 🔮 Future Improvements

- Add chat history
- Support multiple PDFs
- Add source citation from PDF
- Improve PDF formatting extraction
- Add FastAPI backend
