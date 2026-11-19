import hashlib
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ===============================
# CONFIGURATION
# ===============================
EMBEDDING_MODEL = "mxbai-embed-large"
# mxbai-embed-large is trained to expect this prefix on search queries (never on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOCS_FOLDER = "docs"
DB_PATH = f"./multi_pdf_db_{EMBEDDING_MODEL}"

LLM_MODEL = "llama3.2"
LLM_CONTEXT_TOKENS = 8192  # Ollama's default window is small and silently truncates long prompts

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
FETCH_K = 20           # candidates pulled from each of keyword (BM25) and vector search
TOP_N = 6              # chunks kept after reranking and sent to the LLM
# FlashRank's default (ms-marco-MultiBERT-L-12) scores almost everything ~0.99, so it can't tell
# relevant from unrelated. MiniLM-L-12-v2 is calibrated: off-topic questions score ~0.000 and
# relevant chunks 0.3-0.99, which makes MIN_RELEVANCE a usable "the docs don't cover this" cutoff.
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
MIN_RELEVANCE = 0.02

NO_ANSWER = "I don't know based on the documents."

_ADD_BATCH = 500  # Chroma rejects very large single inserts


class _QueryPrefixedEmbeddings(Embeddings):
    """Adds QUERY_PREFIX to search queries only; documents are embedded as-is."""

    def __init__(self, base: Embeddings, prefix: str):
        self.base = base
        self.prefix = prefix

    def embed_documents(self, texts):
        return self.base.embed_documents(texts)

    def embed_query(self, text):
        return self.base.embed_query(self.prefix + text)


def _list_pdfs():
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    return sorted(f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf"))


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_and_split(filename, file_hash):
    pages = PyMuPDFLoader(os.path.join(DOCS_FOLDER, filename)).load()
    for doc in pages:
        # Keep metadata minimal; "page" is stored 1-based so it can be shown as-is.
        doc.metadata = {
            "source": filename,
            "page": doc.metadata.get("page", 0) + 1,
            "file_hash": file_hash,
        }
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return [c for c in splitter.split_documents(pages) if c.page_content.strip()]


def get_vectorstore():
    embeddings = _QueryPrefixedEmbeddings(OllamaEmbeddings(model=EMBEDDING_MODEL), QUERY_PREFIX)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


def sync_index(vectorstore):
    """Make the vector store match docs/: index new or changed PDFs, drop deleted ones.

    Each chunk records its file's SHA-256, so unchanged files are skipped without re-parsing.
    Returns {"added": [...], "updated": [...], "removed": [...], "failed": {name: error}}.
    """
    indexed = {}  # source -> set of file hashes currently stored for it
    for meta in vectorstore.get(include=["metadatas"])["metadatas"]:
        indexed.setdefault(meta.get("source"), set()).add(meta.get("file_hash"))

    on_disk = {name: _file_hash(os.path.join(DOCS_FOLDER, name)) for name in _list_pdfs()}
    report = {"added": [], "updated": [], "removed": [], "failed": {}}

    for source in indexed:
        if source not in on_disk:
            vectorstore.delete(where={"source": source})
            report["removed"].append(source)

    for name, file_hash in on_disk.items():
        if indexed.get(name) == {file_hash}:
            continue
        try:
            chunks = _load_and_split(name, file_hash)
        except Exception as e:  # keep any previously indexed version of this file
            report["failed"][name] = str(e)
            continue
        if name in indexed:
            vectorstore.delete(where={"source": name})
        ids = [f"{file_hash[:16]}-{i}" for i in range(len(chunks))]
        for start in range(0, len(chunks), _ADD_BATCH):
            vectorstore.add_documents(chunks[start:start + _ADD_BATCH], ids=ids[start:start + _ADD_BATCH])
        report["updated" if name in indexed else "added"].append(name)

    return report


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_retriever(vectorstore):
    """Hybrid retriever: BM25 keyword + vector search, merged and reranked with FlashRank.

    Returns None when nothing is indexed. Only chunks scoring at least MIN_RELEVANCE survive,
    so an empty result means the documents don't cover the question.
    """
    stored = vectorstore.get(include=["documents", "metadatas"])
    if not stored["ids"]:
        return None
    docs = [Document(page_content=t, metadata=m) for t, m in zip(stored["documents"], stored["metadatas"])]

    keyword = BM25Retriever.from_documents(docs, preprocess_func=_tokenize, k=FETCH_K)
    vector = vectorstore.as_retriever(search_kwargs={"k": FETCH_K})
    candidates = EnsembleRetriever(retrievers=[keyword, vector], weights=[0.5, 0.5])
    reranker = FlashrankRerank(model=RERANK_MODEL, top_n=TOP_N, score_threshold=MIN_RELEVANCE)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=candidates)


def count_word_across_pdfs(word):
    pattern = r"\b" + re.escape(word.lower()) + r"\b"
    total_count = 0
    for filename in _list_pdfs():
        pages = PyMuPDFLoader(os.path.join(DOCS_FOLDER, filename)).load()
        full_text = " ".join(doc.page_content for doc in pages)
        total_count += len(re.findall(pattern, full_text.lower()))
    return total_count


def get_llm():
    # Temperature 0.1 allows for slightly more natural phrasing while staying factual
    return ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=LLM_CONTEXT_TOKENS)


SYSTEM_PROMPT = (
    "You are V.E.R.A., a helpful and detailed AI assistant. Your task is to answer questions based on the provided context.\n"
    "Provide a comprehensive and detailed answer, synthesizing information from all relevant parts of the context. Use bullet points or lists if it helps clarify the information.\n"
    "Answer ONLY using the provided context. Do not make up information.\n"
    f"If the answer is not found in the context, say: '{NO_ANSWER}'\n\n"
    "Context: {context}"
)
