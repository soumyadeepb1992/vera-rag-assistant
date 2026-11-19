import streamlit as st
import re
import speech_recognition as sr
import threading
import base64
import asyncio
import edge_tts
import winsound
import queue
import time
from chat_with_docs import get_vectorstore, get_llm, count_word_across_pdfs, SYSTEM_PROMPT
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# --- GLOBAL INITIALIZATION ---
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = False
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

# --- AUDIO WORKER ---
def audio_worker():
    """Consumes sentences from the queue and plays them sequentially."""
    while True:
        try:
            # Get next sentence from queue
            text = st.session_state.audio_queue.get()
            if text is None or st.session_state.stop_signal: 
                st.session_state.audio_queue.task_done()
                continue

            # Generate and Play Audio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            communicate = edge_tts.Communicate(text, "en-US-AvaNeural")
            audio_data = b""
            
            # Use a temporary loop to run the async generator
            async def get_data():
                data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        data += chunk["data"]
                return data
            
            audio_bytes = loop.run_until_complete(get_data())
            
            if audio_bytes:
                b64 = base64.b64encode(audio_bytes).decode()
                # Injecting the audio directly into the Streamlit frontend
                st.write(f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
                
                # Wait for the audio to likely finish (approx 15 chars per second)
                time.sleep(len(text) / 15 + 0.5)

            st.session_state.audio_queue.task_done()
        except Exception:
            st.session_state.audio_queue.task_done()

# Start the worker thread once if it's not running
if "worker_started" not in st.session_state:
    threading.Thread(target=audio_worker, daemon=True).start()
    st.session_state.worker_started = True

# --- TOOLS ---
def play_beep():
    winsound.Beep(1000, 200)

def listen_to_mic():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        indicator = st.empty()
        indicator.error("🔴 VERA is listening...")
        play_beep()
        try:
            r.adjust_for_ambient_noise(source, duration=0.8)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            indicator.empty()
            text = r.recognize_google(audio)
            if "stop" in text.lower() and "vera" in text.lower():
                st.session_state.stop_signal = True
                return "STOP"
            return text
        except:
            indicator.empty()
            return None

# --- UI & STYLING ---
st.set_page_config(page_title="V.E.R.A. AI", layout="wide")

st.markdown("""
    <style>
    .main .block-container { padding-bottom: 150px !important; }
    div.stButton > button:first-child {
        position: fixed; bottom: 30px; right: 80px; z-index: 9999;
        border-radius: 50%; width: 60px; height: 60px;
        border: 2px solid #ff4b4b; background-color: white; font-size: 24px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    div.stButton > button:first-child:hover { background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 V.E.R.A. Neural Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    with st.spinner("🧠 Initializing Memory..."):
        st.session_state.vectorstore = get_vectorstore()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- MIC BUTTON ---
if st.button("🎤"):
    voice_data = listen_to_mic()
    if voice_data == "STOP":
        st.session_state.stop_signal = True
        st.warning("Stopping...")
    elif voice_data:
        st.session_state.voice_input = voice_data
        st.rerun()

# --- CHAT INPUT ---
user_query = st.chat_input("Ask V.E.R.A. something...")
if "voice_input" in st.session_state:
    user_query = st.session_state.pop("voice_input")

if user_query:
    st.session_state.stop_signal = False 
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sentence_buffer = ""
        
        base_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})
        compressor = FlashrankRerank()
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        
        with st.spinner("Processing Documents..."):
            docs = retriever.invoke(user_query)
        
        if not docs:
            msg = "I don't know based on the documents."
            response_placeholder.markdown(msg)
            st.session_state.audio_queue.put(msg)
        else:
            context = "\n\n".join([d.page_content for d in docs])
            formatted_prompt = f"{SYSTEM_PROMPT.format(context=context)}\n\nQuestion: {user_query}"
            
            llm = get_llm()
            
            # --- REAL-TIME STREAMING & QUEUING ---
            for chunk in llm.stream(formatted_prompt):
                if st.session_state.stop_signal:
                    break
                
                content = chunk.content
                full_response += content
                sentence_buffer += content
                response_placeholder.markdown(full_response + "▌")
                
                # If a sentence is complete, put it in the audio queue
                if any(mark in content for mark in [".", "!", "?", "\n"]):
                    clean_txt = sentence_buffer.strip()
                    if len(clean_txt) > 3:
                        st.session_state.audio_queue.put(clean_txt)
                        sentence_buffer = ""

            response_placeholder.markdown(full_response)
            
            # Display Sources
            with st.expander("📚 Sources"):
                for d in {f"- {d.metadata['source']} (Page {d.metadata.get('page','?')})" for d in docs}:
                    st.write(d)

    st.session_state.messages.append({"role": "assistant", "content": full_response})