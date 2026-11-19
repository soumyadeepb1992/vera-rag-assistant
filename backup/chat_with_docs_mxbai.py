import os
import re
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ===============================
# ANALYTICAL FUNCTION (Exact Count)
# ===============================
def count_word_in_pdf(pdf_path, word):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    full_text = " ".join([doc.page_content for doc in docs])

    pattern = r"\b" + re.escape(word.lower()) + r"\b"
    matches = re.findall(pattern, full_text.lower())

    return len(matches)


# ===============================
# SAFE DOCUMENT LOADER + SPLITTER
# ===============================
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,          # 🔥 Reduced from 1200 → 600
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_documents(docs)


# ===============================
# MAIN BOT FUNCTION
# ===============================
def run_local_bot():
    load_dotenv()

    embedding_model = "mxbai-embed-large"
    pdf_path = "docs/C4CUST-1100_Recovery Repro Sequence ModificationV4.pdf"
    db_path = f"./my_local_db_{embedding_model}"

    if not os.path.exists(pdf_path):
        print(f"❌ Could not find {pdf_path}")
        return

    print(f"--- 🧠 Using Embedding Model: {embedding_model} ---")

    embeddings = OllamaEmbeddings(model=embedding_model)

    # ===============================
    # CREATE OR LOAD VECTOR DATABASE
    # ===============================
    if not os.path.exists(db_path):
        print("--- 📂 Reading Document ---")

        chunks = load_and_split_pdf(pdf_path)

        print(f"📄 Created {len(chunks)} chunks")
        print("🆕 Creating new vector database...")

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )
        except Exception as e:
            print("❌ Embedding failed:", str(e))
            print("👉 Try reducing chunk_size further (e.g., 400)")
            return

    else:
        print("📦 Loading existing vector database...")

        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    # ===============================
    # SETUP LLM
    # ===============================
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        streaming=True
    )

    system_prompt = (
        "You are V.E.R.A., a precise AI assistant.\n"
        "Use ONLY the provided context.\n"
        "If the answer is not in the context, say:\n"
        "'I don't know based on the document.'\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("\n✅ V.E.R.A. is online! (Type 'exit' to quit)")

    # ===============================
    # CHAT LOOP
    # ===============================
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        lower_input = user_input.lower()

        # ===============================
        # Detect Counting Questions
        # ===============================
        count_match = re.search(
            r"how many times.*?word\s+['\"]?(\w+)['\"]?",
            lower_input
        )

        if count_match:
            word_to_count = count_match.group(1)

            print(f"\nVERA: Counting occurrences of '{word_to_count}'...")

            count = count_word_in_pdf(pdf_path, word_to_count)

            print(f"VERA: The word '{word_to_count}' appears {count} times in the document.\n")
            continue

        # ===============================
        # Otherwise Use RAG
        # ===============================
        print("\nVERA: ", end="", flush=True)

        for chunk in rag_chain.stream({"input": user_input}):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)

        print("\n")


if __name__ == "__main__":
    run_local_bot()