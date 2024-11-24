from langchain.embeddings import OpenAIEmbeddings
import os

def get_embedding_function():
    # Ensure your OpenAI API key is set as an environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings

