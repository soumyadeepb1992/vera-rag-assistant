"""Text-to-speech output for V.E.R.A.

SpeechPlayer speaks sentences in order on background threads. Those threads never call
Streamlit (Streamlit calls only work from the script thread); audio is synthesized with
edge-tts and played locally through the Windows MCI API, so no extra packages are needed.
"""
import asyncio
import ctypes
import itertools
import os
import queue
import re
import tempfile
import threading
import time

import edge_tts

VOICE = "en-US-AvaNeural"


# ===============================
# Text preparation
# ===============================
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")
_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "fig", "no", "approx", "inc", "ltd", "st"}


def clean_for_speech(text):
    """Strip markdown and URLs so only readable words get spoken. Returns '' if nothing is left."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)                        # [label](url) -> label
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^\s*(#{1,6}|[-*+•]|\d+[.)])\s+", "", text, flags=re.M)     # headings, bullets, numbering
    text = re.sub(r"[*_`~|>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if re.search(r"[A-Za-z0-9]", text) else ""


class SentenceSplitter:
    """Accumulates streamed LLM tokens and emits complete sentences.

    A boundary needs whitespace after the punctuation, so "3.5", "v1.2" and URLs never split,
    and periods after abbreviations, initials and list numbers ("e.g.", "J.", "1.") are ignored.
    """

    def __init__(self):
        self._buffer = ""

    def feed(self, text):
        self._buffer += text
        sentences, start = [], 0
        for m in _BOUNDARY.finditer(self._buffer):
            if "\n" not in m.group() and self._is_false_period(self._buffer[start:m.end()]):
                continue
            sentence = self._buffer[start:m.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = m.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self):
        rest, self._buffer = self._buffer.strip(), ""
        return rest

    @staticmethod
    def _is_false_period(segment):
        word = segment.split()[-1] if segment.split() else ""
        if "!" in word or "?" in word:
            return False
        core = word.rstrip(".\"')]").lower()
        return core in _ABBREVIATIONS or core.isdigit() or (len(core) == 1 and core.isalpha())


# ===============================
# Playback
# ===============================
_winmm = ctypes.windll.winmm
_aliases = itertools.count()


def _mci(command):
    buf = ctypes.create_unicode_buffer(256)
    err = _winmm.mciSendStringW(command, buf, len(buf) - 1, None)
    if err:
        raise RuntimeError(f"MCI error {err} on {command!r}")
    return buf.value


class SpeechPlayer:
    """Speaks queued text in order. stop() silences the current sentence and drops the rest.

    Two threads form a pipeline so the next sentence is synthesized while the current one plays.
    Every item carries the "generation" it was queued in; stop() bumps the generation, which
    makes both threads discard older items.
    """

    def __init__(self, voice=VOICE):
        self.voice = voice
        self._generation = 0
        self._texts = queue.Queue()
        self._audio = queue.Queue()
        threading.Thread(target=self._synthesize_loop, name="tts-synth", daemon=True).start()
        threading.Thread(target=self._play_loop, name="tts-play", daemon=True).start()

    def say(self, text):
        text = clean_for_speech(text)
        if text:
            self._texts.put((self._generation, text))

    def stop(self):
        self._generation += 1
        for q in (self._texts, self._audio):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _synthesize_loop(self):
        loop = asyncio.new_event_loop()
        while True:
            generation, text = self._texts.get()
            if generation != self._generation:
                continue
            try:
                audio = loop.run_until_complete(self._synthesize(text))
            except Exception as e:
                print(f"[tts] synthesis failed: {e!r}")
                continue
            if audio and generation == self._generation:
                self._audio.put((generation, audio))

    async def _synthesize(self, text):
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        return bytes(audio)

    def _play_loop(self):
        while True:
            generation, audio = self._audio.get()
            if generation != self._generation:
                continue
            try:
                self._play(audio, generation)
            except Exception as e:
                print(f"[tts] playback failed: {e!r}")

    def _play(self, audio, generation):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, "wb") as f:
            f.write(audio)
        alias = f"vera{next(_aliases)}"
        try:
            _mci(f'open "{path}" type mpegvideo alias {alias}')
            try:
                _mci(f"play {alias}")
                while _mci(f"status {alias} mode") == "playing":
                    if generation != self._generation:
                        _mci(f"stop {alias}")
                        break
                    time.sleep(0.05)
            finally:
                _mci(f"close {alias}")
        finally:
            os.remove(path)
