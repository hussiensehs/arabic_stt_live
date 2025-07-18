import streamlit as st
from faster_whisper import WhisperModel
import asyncio
from livekit import rtc
import soundfile as sf
import numpy as np
import tempfile
import os
from pydub import AudioSegment

# ⬇️ إعدادات الاتصال بـ LiveKit
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://stt-arabic-jbyb69nd.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "APIR2NRVYgdadun")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "vI1P58EouVNhTe3U0KpTSy1ffBnH9k2D91fJRcJhl6jA")

# ⬇️ إعداد واجهة Streamlit
st.set_page_config(page_title="Arabic STT Live", layout="wide")
st.title("🎤 Live Arabic Speech-to-Text")

# زر بدء البث
start = st.button("🔴 Start Live Recording")

# خانة لعرض النص المباشر
live_text = st.empty()

# تحميل نموذج faster-whisper
@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="int8")

model = load_model()

# تحويل الصوت إلى نص
def transcribe(audio_path):
    segments, _ = model.transcribe(audio_path, language="ar")
    full_text = ""
    for seg in segments:
        full_text += seg.text + " "
    return full_text.strip()

# استقبال صوت من LiveKit
async def receive_audio_and_transcribe():
    room = rtc.Room()
    await room.connect(LIVEKIT_URL, rtc.AccessToken(
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
        identity="streamlit-client"
    ).to_jwt())

    # استخدم هذه الخانة لحفظ الصوت مؤقتًا
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_path = temp_audio.name

    audio_chunks = []

    @room.on("track_subscribed")
    async def on_track(track, publication, participant):
        if track.kind == "audio":
            async for frame in track:
                # افترض أن `frame` يحتوي على PCM raw audio
                audio_chunks.append(frame.data)

                # بعد تجميع 2 ثانية صوت، حللها
                if len(audio_chunks) >= 20:
                    data = b"".join(audio_chunks)
                    audio = AudioSegment(data, sample_width=2, frame_rate=16000, channels=1)
                    audio.export(audio_path, format="wav")

                    # تحليل الصوت وعرض النص
                    text = transcribe(audio_path)
                    live_text.markdown(f"📝 **Transcription:** {text}")
                    audio_chunks.clear()

    await room.join()

if start:
    st.warning("🎙️ Listening... Speak Arabic clearly into your mic.")
    asyncio.run(receive_audio_and_transcribe())
