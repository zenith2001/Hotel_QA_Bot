import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document  # Corrected import
from config import CHROMA_DB_DIR, EMBEDDING_MODEL, PDF_FILE_PATH

def clean_text(text):
    """Cleans extracted text by removing bullets, excess spaces, and converting to lowercase."""
    text = text.lower()
    text = re.sub(r'^[●•*-]\s*', '', text, flags=re.MULTILINE)  # Remove bullets at start of lines
    text = re.sub(r'●|•|-|\*', '', text)  # Remove any remaining standalone bullets
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

import os
from config import PDF_FILE_PATH  # Use updated config path

def load_pdfs(pdf_folder):
    """Loads all PDFs from the given folder, extracts text, and returns documents."""
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"❌ Error: The directory '{pdf_folder}' does not exist. Please create it and add PDFs.")

    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            hotel_name = os.path.splitext(file)[0]  # Extract hotel name from filename
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs = loader.load()
        
            # Merge all pages into a single text block
            full_text = " ".join([clean_text(doc.page_content) for doc in docs])

            # Create a single document for the entire PDF
            cleaned_doc = Document(
                page_content=full_text, 
                metadata={"hotel": hotel_name}
            )
            documents.append(cleaned_doc)

    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=200,  # Keeps some context between chunks
        separators=["\n\n", "\n", " ", "."],
    )
    return text_splitter.split_documents(documents)
