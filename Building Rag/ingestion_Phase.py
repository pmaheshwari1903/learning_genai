from dotenv import load_dotenv
import os
from pathlib import Path # for path fetching
from langchain_community.document_loaders import PyPDFLoader # for pdf load
from langchain_text_splitters import RecursiveCharacterTextSplitter # for chunking
from langchain_google_genai import GoogleGenerativeAIEmbeddings #for embeddings I am using google gemini model
from langchain_qdrant import QdrantVectorStore # for qdrant vector store ---> by langchain

load_dotenv()

# yha pe apan ne sabse pehle pdf ke path ko set kiya h aur fhir pdf ko load kraya h

pdf_path = "D:/WorkXP/GenAII/Building Rag/ASSETS/dental_students_book.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()


# yha apan ne file ke content ko chunks me daal diya h!
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, # yha pe apan ne context yaad rhe esliye hr para ki 200 line agle para me bhi jaye esliye kiya h
)
chunks = text_splitter.split_documents(documents=docs)


# Embeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_documents(
    collection_name="learning_rag_genai",
    url="http://localhost:6333",
    documents=chunks,
    embedding=embedding_model
)

print("Indexing of document done...")