# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

V.E.R.A. — a local RAG ("chat with your PDFs") assistant with a voice-enabled Streamlit UI. Not a git repo; no tests, linter, or requirements file. Windows-only (`winsound`, and TTS playback via the Windows MCI API).

## Running

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Prerequisites: a running Ollama server with `llama3.2` (chat) and `mxbai-embed-large` (embeddings) pulled. Speech input uses Google speech recognition and TTS uses `edge_tts`, both online. `edge_tts` breaks when Microsoft changes its handshake (symptom: `WSServerHandshakeError: 403`); fix with `pip install -U edge-tts`.

The app can be exercised headlessly with `streamlit.testing.v1.AppTest.from_file("app.py")`; monkeypatch `chat_with_docs.DB_PATH` to a copy of the DB and `voice.SpeechPlayer.say` to avoid touching the real index or speakers.

## Architecture

- **`chat_with_docs.py`** — RAG backend (no `__main__`); all tuning constants live at the top.
  - `get_vectorstore()` opens Chroma at `./multi_pdf_db_<EMBEDDING_MODEL>`. Queries (not documents) get the mxbai `QUERY_PREFIX` via `_QueryPrefixedEmbeddings`.
  - `sync_index()` makes the store match `docs/` incrementally: each chunk stores its file's SHA-256 in `file_hash`, so unchanged PDFs are skipped, changed ones are deleted by `source` and re-added, removed ones are dropped. Chunk IDs are `<hash16>-<n>`. `page` metadata is stored 1-based. PDFs are parsed with `PyMuPDFLoader`.
  - `build_retriever()` = BM25 (over all chunks read back from Chroma) + vector search → `EnsembleRetriever` → `FlashrankRerank` (`ms-marco-MiniLM-L-12-v2`, `TOP_N`, `MIN_RELEVANCE`). An empty result means "not in the docs" — the app then answers `NO_ANSWER`. Don't switch back to FlashRank's default MultiBERT model: it scores ~0.99 for everything, so the threshold stops working. FlashRank caches models in `D:\tmp`.
- **`voice.py`** — `SentenceSplitter` turns streamed tokens into sentences (ignores decimals, URLs, abbreviations, list numbers); `clean_for_speech` strips markdown; `SpeechPlayer` runs a synth thread + play thread with a generation counter so `stop()` cancels in-flight audio. These threads must never call Streamlit APIs.
- **`app.py`** — Streamlit UI. Retriever, LLM and `SpeechPlayer` are `@st.cache_resource` singletons; the sidebar "Re-index documents" button clears the knowledge-base cache to re-run `sync_index`. The assistant message is appended to history before streaming and filled in place, so a Stop/rerun keeps the partial answer. Buttons are styled via their `st-key-<key>` CSS classes.
- **`backup/`** — earlier standalone CLI iterations and pre-change copies of the scripts. Reference only; not imported.

## Gotchas

- `env` (no leading dot) holds an `OPENAI_API_KEY`, but `load_dotenv()` looks for `.env`, and nothing currently uses OpenAI.
- LangChain 1.x is installed: legacy chains/retrievers come from `langchain_classic` (e.g. `ContextualCompressionRetriever`, `EnsembleRetriever`), not `langchain`.
