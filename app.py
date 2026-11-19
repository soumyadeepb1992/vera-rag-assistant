import speech_recognition as sr
import streamlit as st
import winsound

from chat_with_docs import NO_ANSWER, SYSTEM_PROMPT, build_retriever, get_llm, get_vectorstore, sync_index
from voice import SentenceSplitter, SpeechPlayer

st.set_page_config(page_title="V.E.R.A. AI", layout="wide")


# --- SHARED RESOURCES (built once per server process) ---
@st.cache_resource
def get_speech_player():
    return SpeechPlayer()


@st.cache_resource(show_spinner="🧠 Indexing documents...")
def load_knowledge_base():
    vectorstore = get_vectorstore()
    report = sync_index(vectorstore)
    return build_retriever(vectorstore), report


@st.cache_resource
def load_llm():
    return get_llm()


player = get_speech_player()


# --- VOICE INPUT ---
def listen_to_mic():
    """Returns recognized text, or None (with a message shown) if nothing usable was heard."""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    status = st.empty()
    try:
        with sr.Microphone() as source:
            status.error("🔴 VERA is listening...")
            winsound.Beep(1000, 200)
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    except sr.WaitTimeoutError:
        status.warning("I didn't hear anything. Click 🎤 and try again.")
        return None
    except OSError as e:
        status.error(f"Microphone unavailable: {e}")
        return None

    status.info("Recognizing...")
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        status.warning("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError as e:
        status.error(f"Speech recognition service unavailable: {e}")
        return None
    status.empty()
    return text


# --- UI & STYLING ---
st.markdown("""
    <style>
    .main .block-container { padding-bottom: 150px !important; }
    .st-key-mic_btn button, .st-key-stop_btn button {
        position: fixed; bottom: 30px; z-index: 9999;
        border-radius: 50%; width: 60px; height: 60px;
        border: 2px solid #ff4b4b; background-color: white; font-size: 24px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .st-key-mic_btn button { right: 80px; }
    .st-key-stop_btn button { right: 150px; }
    .st-key-mic_btn button:hover, .st-key-stop_btn button:hover { background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 V.E.R.A. Neural Assistant")

retriever, index_report = load_knowledge_base()

with st.sidebar:
    st.subheader("📂 Documents")
    for label, key in (("Indexed", "added"), ("Re-indexed", "updated"), ("Removed", "removed")):
        if index_report[key]:
            st.success(f"{label}: " + ", ".join(index_report[key]))
    for name, error in index_report["failed"].items():
        st.error(f"Could not read {name}: {error}")
    if st.button("🔄 Re-index documents", key="reindex_btn"):
        load_knowledge_base.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"] or "_(stopped)_")

if retriever is None:
    st.info("No documents indexed yet. Add PDF files to the `docs` folder, then click **Re-index documents**.")
    st.stop()

# --- VOICE CONTROLS ---
# Clicking either button reruns the script, which also interrupts an answer that is still streaming.
if st.button("⏹", key="stop_btn", help="Stop speaking"):
    player.stop()

if st.button("🎤", key="mic_btn", help="Ask by voice"):
    voice_data = listen_to_mic()
    if voice_data and "stop" in voice_data.lower() and "vera" in voice_data.lower():
        player.stop()
        st.toast("Stopped.")
    elif voice_data:
        st.session_state.voice_input = voice_data
        st.rerun()

# --- CHAT INPUT ---
user_query = st.chat_input("Ask V.E.R.A. something...")
if "voice_input" in st.session_state:
    user_query = st.session_state.pop("voice_input")

if user_query:
    player.stop()  # cut off any earlier answer that is still being spoken
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        # Stored before streaming and filled in place, so a partial answer survives a Stop/rerun.
        answer = {"role": "assistant", "content": ""}
        st.session_state.messages.append(answer)

        with st.spinner("Searching documents..."):
            docs = retriever.invoke(user_query)

        if not docs:
            answer["content"] = NO_ANSWER
            response_placeholder.markdown(NO_ANSWER)
            player.say(NO_ANSWER)
        else:
            context = "\n\n".join(d.page_content for d in docs)
            formatted_prompt = f"{SYSTEM_PROMPT.format(context=context)}\n\nQuestion: {user_query}"
            splitter = SentenceSplitter()

            # --- REAL-TIME STREAMING & SPEAKING ---
            for chunk in load_llm().stream(formatted_prompt):
                answer["content"] += chunk.content
                response_placeholder.markdown(answer["content"] + "▌")
                for sentence in splitter.feed(chunk.content):
                    player.say(sentence)
            player.say(splitter.flush())
            response_placeholder.markdown(answer["content"])

            with st.expander("📚 Sources"):
                sources = sorted({(d.metadata["source"], d.metadata.get("page", "?")) for d in docs}, key=str)
                for source, page in sources:
                    st.write(f"- {source} (Page {page})")
