import os
import re
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ===============================
# CONFIGURATION
# ===============================
EMBEDDING_MODEL = "mxbai-embed-large"
DOCS_FOLDER = "docs"
DB_PATH = f"./multi_pdf_db_{EMBEDDING_MODEL}"

def load_and_split_all_pdfs():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return []

    all_chunks = []
    # Increased chunk size for better context/longer answers
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_FOLDER, filename)
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for i, doc in enumerate(docs):
                    doc.metadata["source"] = filename
                    doc.metadata["page"] = doc.metadata.get("page", i + 1)
                
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return all_chunks

def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    chunks = load_and_split_all_pdfs()
    if not chunks: return None
        
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
s
def count_word_across_pdfs(word):
    total_count = 0
    if not os.path.exists(DOCS_FOLDER): return 0
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_FOLDER, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            full_text = " ".join([doc.page_content for doc in docs])
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            matches = re.findall(pattern, full_text.lower())
            total_count += len(matches)
    return total_count

def get_llm():
    # Temperature 0.1 allows for slightly more natural phrasing while staying factual
    return ChatOllama(model="llama3.2", temperature=0.1, streaming=True)

SYSTEM_PROMPT = (
    "You are V.E.R.A., a helpful and detailed AI assistant. Your task is to answer questions based on the provided context.\n"
    "Provide a comprehensive and detailed answer, synthesizing information from all relevant parts of the context. Use bullet points or lists if it helps clarify the information.\n"
    "Answer ONLY using the provided context. Do not make up information.\n"
    "If the answer is not found in the context, say: 'I don't know based on the documents.'\n\n"
    "Context: {context}"
)