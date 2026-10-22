from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt
from RAG.embeddings import get_embeddings


Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)

vectorstore = Chroma(
    collection_name = "agentic_chaybot_docs",
    embedding_function = get_embeddings(),
    persist_directory = "chroma_db"
)

def read_file_text(file_path: str)->str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
        return text
    if suffix == ".docx":
        return docx2txt.process(file_path)
    
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    if suffix == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    raise ValueError("Unsupported file type, Upload docx, pdf,.txt,.md only")

def add_document_to_rag(file_path: str, thread_id: str):
    text = read_file_text(file_path)
    if not text.strip():
        raise ValueError("The document is empty or contains only whitespace.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap=150
    )
    
    chunks = splitter.split_text(text)
    docs: List[Document] = [
        Document(
page_content=chunk, 
metadata={"source": file_path,
          "thread_id": thread_id}) for chunk in chunks
        
    ]
    
    vectorstore.add_documents(docs)
    return {
        "filename" : Path(file_path),
        "chunks": len(docs)
    }
    
    
def retrieve_from_rag(query: str, thread_id: str,K:int=4):
    docs = vectorstore.similarity_search(
        query,
        k=K,
        filter={"thread_id": thread_id}
    )
    
    if not docs:
        return "No relevant uploaded document content found."
    
    results = []
    
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source","uploaded_document")
        results.append(
            f"[Source {i}: {source} \n {doc.page_content}]"
        )
    return "\n\n".join(results)

