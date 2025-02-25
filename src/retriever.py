import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import CHROMA_DB_DIR, EMBEDDING_MODEL,PDF_FILE_PATH


def initialize_vector_db(reset=False):
    """Initializes ChromaDB with embeddings, optionally resetting it."""
    if reset:
        print("Deleting existing ChromaDB storage...")
        shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
        print("ChromaDB storage deleted successfully!")

    if not os.path.exists(CHROMA_DB_DIR) or reset:
        print("Creating a fresh ChromaDB instance...")
        from preprocess import chunk_documents, load_pdfs  # Import here to avoid circular imports
        pdf_docs = load_pdfs(PDF_FILE_PATH)
        chunks = chunk_documents(pdf_docs)
        print(len(chunks))

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            persist_directory=CHROMA_DB_DIR
        )
        vector_db.persist()
        print(f"ChromaDB created with {len(chunks)} document chunks!")
    else:
        print("Loading existing ChromaDB instance...")
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    return vector_db


def retrieve_answer(user_query, hotel_name, top_k=5):
    """Retrieves relevant hotel policies from ChromaDB."""
    vector_db = initialize_vector_db(reset=False)  # Load existing DB

    results = vector_db.similarity_search(
        query=user_query,
        k=top_k,  # Retrieve more candidates
        filter={"hotel": hotel_name}
    )

    if not results:
        return ["No relevant information found."]
    
    return [doc.page_content for doc in results]

