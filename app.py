import streamlit as st
import asyncio
import soundfile as sf
import numpy as np
from livekit import Room, RoomOptions, TrackSource
from livekit.rtc import create_token
from faster_whisper import WhisperModel
import librosa
import av
import os
import torch

# إعدادات LiveKit من secrets.toml
LIVEKIT_URL = st.secrets["LIVEKIT_URL"]
LIVEKIT_API_KEY = st.secrets["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = st.secrets["LIVEKIT_API_SECRET"]

# إعداد واجهة Streamlit
st.set_page_config(page_title="Arabic STT", layout="wide")
st.title("🎙️ تحويل الكلام العربي إلى نص مباشر")
st.markdown("""
<style>
.arabic-text { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# تحميل نموذج Whisper
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel("small", device=device)

model = load_model()

# توليد توكن LiveKit
def generate_token(identity="streamlit-user"):
    return create_token(
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room="arabic-stt",
        identity=identity,
        ttl=3600
    )

# معالجة الصوت وتحويله إلى نص
def process_audio(audio_data, sample_rate):
    try:
        segments, _ = model.transcribe(
            audio_data,
            language="ar",
            vad_filter=True,
            beam_size=5
        )
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        return f"⚠️ خطأ في التحويل: {str(e)}"

# الجلسة التفاعلية مع LiveKit
async def start_session():
    status_box = st.empty()
    transcript_box = st.empty()
    
    try:
        token = generate_token()
        room = await Room.connect(
            LIVEKIT_URL,
            token,
            RoomOptions(auto_subscribe=True)
        )
        status_box.success("✅ تم الاتصال بغرفة LiveKit، ابدأ بالتحدث...")

        @room.on("track_subscribed")
        async def on_audio_track(track, publication, participant):
            if track.kind != "audio":
                return

            container = av.open(track, format="s16le", mode="r")
            stream = container.streams.audio[0]
            
            for frame in container.decode(stream):
                audio_data = frame.to_ndarray()
                if stream.sample_rate != 16000:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=stream.sample_rate, 
                        target_sr=16000
                    )
                
                text = process_audio(audio_data, 16000)
                transcript_box.markdown(f"""
                <div class='arabic-text'>
                <h3>النص المتحصل عليه:</h3>
                <p>{text}</p>
                </div>
                """, unsafe_allow_html=True)

        await room.join()
    except Exception as e:
        status_box.error(f"❌ فشل الاتصال: {str(e)}")

# واجهة المستخدم
col1, col2 = st.columns(2)
with col1:
    st.header("الإعدادات")
    model_size = st.selectbox(
        "حجم النموذج",
        ["tiny", "base", "small", "medium"],
        index=2
    )
    
with col2:
    st.header("التحكم")
    if st.button("بدء التحويل المباشر", type="primary"):
        asyncio.run(start_session())

    uploaded_file = st.file_uploader("أو اختر ملف صوتي", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("تحويل الملف"):
            with st.spinner("جاري التحويل..."):
                audio_data, sr = librosa.load(uploaded_file, sr=16000)
                text = process_audio(audio_data, sr)
                st.markdown(f"""
                <div class='arabic-text'>
                <h3>نتيجة التحويل:</h3>
                <p>{text}</p>
                </div>
                """, unsafe_allow_html=True)
