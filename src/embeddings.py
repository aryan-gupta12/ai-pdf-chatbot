from langchain_community.embeddings import HuggingFaceEmbeddings


def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )