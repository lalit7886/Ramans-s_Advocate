from langchain_huggingface import HuggingFaceEmbeddings
import os
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL_NAME"))
    return _embeddings


