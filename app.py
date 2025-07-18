import streamlit as st
import asyncio
import numpy as np
import librosa
import torch
from livekit import Room, RoomOptions, AudioFrame
from livekit.rtc import create_token
from faster_whisper import WhisperModel
from typing import Optional
import nest_asyncio

# حل مشكلة asyncio داخل Streamlit
nest_asyncio.apply()

# إعداد واجهة Streamlit
st.set_page_config(
    page_title="Arabic Speech-to-Text",
    page_icon="🎙️",
    layout="wide"
)

# دعم النص العربي (RTL)
st.markdown("""
<style>
.rtl {
    direction: rtl;
    text-align: right;
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ تحويل الكلام العربي إلى نص مباشر")
st.markdown('<div class="rtl">تطبيق يحول الكلام العربي إلى نص مكتوب في الوقت الفعلي</div>', unsafe_allow_html=True)

# تحميل مفاتيح LiveKit من secrets
try:
    LIVEKIT_URL = st.secrets["LIVEKIT_URL"]
    LIVEKIT_API_KEY = st.secrets["LIVEKIT_API_KEY"]
    LIVEKIT_API_SECRET = st.secrets["LIVEKIT_API_SECRET"]
except Exception as e:
    st.error(f"خطأ في تحميل المفاتيح: {str(e)}")
    st.stop()

# تحميل نموذج Whisper
@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        model_size_or_path="base",
        device="cpu",
        compute_type="int8",
        download_root="./whisper_models"
    )

model = load_whisper_model()

# إنشاء توكن LiveKit
def generate_livekit_token(identity: str = "streamlit-user"):
    return create_token(
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room_name="arabic-stt-room",
        identity=identity,
        ttl=3600
    )

# تحويل الصوت إلى نص
def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    try:
        segments, _ = model.transcribe(
            audio_data,
            language="ar",
            vad_filter=True,
            beam_size=3
        )
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        return f"⚠️ خطأ: {str(e)}"

# إدارة الجلسة
class LiveKitSession:
    def __init__(self):
        self.room: Optional[Room] = None
        self.transcription_box = st.empty()
        self.is_running = False

    async def start(self):
        try:
            self.is_running = True
            token = generate_livekit_token()
            self.room = await Room.connect(
                LIVEKIT_URL,
                token,
                RoomOptions(auto_subscribe=True)
            )

            @self.room.on("track_subscribed")
            async def on_track_subscribed(track, publication, participant):
                if track.kind != "audio":
                    return

                @track.on("frame")
                async def on_audio_frame(frame: AudioFrame):
                    if not self.is_running:
                        return
                    try:
                        if not frame.data:
                            return
                        audio_np = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        if frame.num_channels > 1:
                            audio_np = audio_np.reshape(-1, frame.num_channels).mean(axis=1)

                        if frame.sample_rate != 16000:
                            audio_np = librosa.resample(audio_np, orig_sr=frame.sample_rate, target_sr=16000)

                        text = transcribe_audio(audio_np, 16000)
                        self.update_transcription(text)
                    except Exception as e:
                        st.warning(f"خطأ في معالجة الصوت: {str(e)}")

            st.success("تم الاتصال بنجاح، ابدأ بالتحدث...")
        except Exception as e:
            st.error(f"فشل الاتصال: {str(e)}")
            self.is_running = False

    def update_transcription(self, text: str):
        self.transcription_box.markdown(f"""
        <div class="rtl" style="
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #FF4B4B;
        ">
            <p style='font-size: 18px;'>{text}</p>
        </div>
        """, unsafe_allow_html=True)

    async def stop(self):
        self.is_running = False
        if self.room:
            await self.room.disconnect()
        st.success("تم إيقاف البث بنجاح")

# الواجهة الرئيسية
def main():
    session = LiveKitSession()

    col1, col2 = st.columns(2)

    with col1:
        st.header("الإعدادات")
        st.info("""
        <div class="rtl">
        - تأكد من السماح باستخدام الميكروفون في المتصفح<br>
        - جودة التحويل تعتمد على وضوح الصوت والإنترنت
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.header("التحكم")
        if st.button("▶️ بدء البث المباشر", type="primary"):
            with st.spinner("جاري الاتصال بغرفة LiveKit..."):
                asyncio.get_event_loop().run_until_complete(session.start())

        if st.button("⏹️ إيقاف البث"):
            asyncio.get_event_loop().run_until_complete(session.stop())

    # رفع ملف صوتي
    st.divider()
    uploaded_file = st.file_uploader("تحميل ملف صوتي", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file)

        if st.button("تحويل الملف"):
            with st.spinner("جاري معالجة الملف..."):
                try:
                    audio_data, sample_rate = librosa.load(uploaded_file, sr=16000)
                    text = transcribe_audio(audio_data, sample_rate)

                    st.markdown(f"""
                    <div class="rtl" style="margin-top: 20px;">
                        <h3>نتيجة التحويل:</h3>
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                            {text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"خطأ في معالجة الملف: {str(e)}")

if __name__ == "__main__":
    main()
