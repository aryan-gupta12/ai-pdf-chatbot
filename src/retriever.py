from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    return FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )


def retrieve_documents(vector_store, query, k=3):
    return vector_store.similarity_search(query, k=k)