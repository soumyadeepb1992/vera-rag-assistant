import os
import re
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ===============================
# LOAD & SPLIT MULTIPLE PDFs
# ===============================
def load_and_split_all_pdfs(docs_folder):
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_folder, filename)
            print(f"📄 Loading: {filename}")

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    return all_chunks


# ===============================
# COUNT WORD ACROSS ALL PDFs
# ===============================
def count_word_across_pdfs(docs_folder, word):
    total_count = 0

    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_folder, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            full_text = " ".join([doc.page_content for doc in docs])
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            matches = re.findall(pattern, full_text.lower())

            total_count += len(matches)

    return total_count


# ===============================
# MAIN BOT FUNCTION
# ===============================
def run_local_bot():
    load_dotenv()

    embedding_model = "mxbai-embed-large"
    docs_folder = "docs"
    db_path = f"./multi_pdf_db_{embedding_model}"

    if not os.path.exists(docs_folder):
        print("❌ 'docs' folder not found.")
        return

    print(f"--- 🧠 Using Embedding Model: {embedding_model} ---")

    embeddings = OllamaEmbeddings(model=embedding_model)

    # ===============================
    # CREATE OR LOAD VECTOR DATABASE
    # ===============================
    if not os.path.exists(db_path):
        print("--- 📂 Reading All PDFs ---")

        chunks = load_and_split_all_pdfs(docs_folder)

        print(f"📄 Total chunks created: {len(chunks)}")
        print("🆕 Creating vector database...")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
    else:
        print("📦 Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ===============================
    # SETUP LLM (Streaming Enabled)
    # ===============================
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        streaming=True
    )

    system_prompt = (
        "You are V.E.R.A., a precise AI assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not found, say:\n"
        "'I don't know based on the documents.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

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
        # COUNTING QUESTIONS
        # ===============================
        count_match = re.search(
            r"how many times.*?word\s+['\"]?(\w+)['\"]?",
            lower_input
        )

        if count_match:
            word_to_count = count_match.group(1)

            print(f"\nVERA: Counting occurrences of '{word_to_count}' across all documents...")

            count = count_word_across_pdfs(docs_folder, word_to_count)

            print(f"VERA: The word '{word_to_count}' appears {count} times across all PDFs.\n")
            continue

        # ===============================
        # SEMANTIC QUESTIONS (RAG)
        # ===============================
        retrieved_docs = retriever.invoke(user_input)

        if not retrieved_docs:
            print("\nVERA: I don't know based on the documents.\n")
            continue

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = system_prompt.format(
            context=context_text,
            question=user_input
        )

        print("\n🧠 Vera is Thinking...\n")
        print("VERA: ", end="", flush=True)

        # Streaming response (ChatGPT typing effect)
        for chunk in llm.stream(final_prompt):
            print(chunk.content, end="", flush=True)

        print("\n")

        # ===============================
        # DISPLAY SOURCES
        # ===============================
        print("📚 Sources:")
        shown_sources = set()

        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown File")
            page = doc.metadata.get("page", "Unknown Page")

            source_entry = f"- {source} (Page {page})"

            if source_entry not in shown_sources:
                print(source_entry)
                shown_sources.add(source_entry)

        print()


if __name__ == "__main__":
    run_local_bot()