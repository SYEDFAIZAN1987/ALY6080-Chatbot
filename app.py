import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from embedding_utils import get_embedding_function
import os

# Define paths
CHROMA_PATH = "chroma"
PROJECT_REPORT_PATH = r"C:\Users\sfaiz\OneDrive\Desktop\Syed_Faizan_Resume.pdf"

# Preprocess and store the report in Chroma (only runs once)
@st.cache_resource
def preprocess_and_load_chroma():
    if not os.path.exists(CHROMA_PATH):
        st.info("Processing the project report for the first time. Please wait...")
        loader = PyPDFLoader(PROJECT_REPORT_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )
        db.add_documents(chunks)
        db.persist()
        st.success(f"Processed and stored {len(chunks)} document chunks!")
    else:
        st.info("Loading existing Chroma database...")
    
    return Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

# Load Chroma database
db = preprocess_and_load_chroma()

# Streamlit UI
st.title("Project Report Search App")
st.subheader("Search the ALY 6080 Group 1 project report with ease!")

# Search input
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a search query!")
    else:
        st.info("Searching the project report...")
        results = db.similarity_search(query, k=5)  # Retrieve top 5 results
        if results:
            st.success(f"Found {len(results)} results!")
            for result in results:
                st.write(f"**Source:** {result.metadata.get('source')}")
                st.write(f"**Page:** {result.metadata.get('page')}")
                st.write(f"**Text:** {result.page_content}")
                st.write("---")
        else:
            st.warning("No results found.")

# Footer
st.markdown("Developed by Syed Faizan using Streamlit")
