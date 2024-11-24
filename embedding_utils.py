from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Returns an embedding function using Hugging Face's all-MiniLM-L6-v2 model.
    """
    # Initialize Hugging Face embeddings using a pre-trained sentence-transformers model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

