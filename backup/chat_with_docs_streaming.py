import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def run_local_bot():
    # 1️⃣ Setup paths
    pdf_path = "docs/python.pdf"
    db_path = "./my_local_db"

    if not os.path.exists(pdf_path):
        print(f"❌ Could not find {pdf_path}")
        return

    print("--- 🧠 Loading / Creating Local AI Memory ---")

    # 2️⃣ Setup Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Optional better model:
    # embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 3️⃣ Create or Load Vector DB
    if not os.path.exists(db_path):
        print("--- 📂 Reading Document ---")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        print("🆕 Creating new vector database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        vectorstore.persist()
    else:
        print("📦 Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    # 4️⃣ Setup Streaming LLM
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        streaming=True   # 🔥 ENABLE STREAMING
    )

    system_prompt = (
        "You are a local AI assistant. Use the context to answer the question. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 5️⃣ Create Retrieval Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("\n✅ V.E.R.A. is online! (Type 'exit' to quit)")

    # 6️⃣ Chat Loop with Streaming
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
            
        print("VERA is thinking...\nVERA: ", end="", flush=True)

        # 🔥 STREAMING RESPONSE
        for chunk in rag_chain.stream({"input": user_input}):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)

        print("\n")  # Clean newline after response


if __name__ == "__main__":
    run_local_bot()