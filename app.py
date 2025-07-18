import streamlit as st
import asyncio
import numpy as np
import av
import librosa
import torch
from livekit import Room, RoomOptions, TrackSource
from livekit.rtc import create_token
from faster_whisper import WhisperModel
from typing import Optional

# إعداد واجهة Streamlit
st.set_page_config(
    page_title="Arabic Speech-to-Text",
    page_icon="🎙️",
    layout="wide"
)

# عنوان التطبيق مع دعم RTL
st.markdown("""
<style>
.rtl {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ تحويل الكلام العربي إلى نص مباشر")
st.markdown('<div class="rtl">تطبيق يحول الكلام العربي إلى نص مكتوب في الوقت الفعلي باستخدام LiveKit وWhisper</div>', 
            unsafe_allow_html=True)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        model_size_or_path="small",
        device=device,
        compute_type="int8" if device == "cpu" else "float16"
    )
    return model

model = load_whisper_model()

# إنشاء توكن LiveKit
def generate_livekit_token(identity: str = "streamlit-user"):
    return create_token(
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room_name="arabic-stt-room",
        identity=identity,
        ttl=3600  # صلاحية ساعة واحدة
    )

# معالجة وتحويل الصوت إلى نص
def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    try:
        segments, _ = model.transcribe(
            audio_data,
            language="ar",
            vad_filter=True,
            beam_size=5
        )
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        return f"⚠️ خطأ في التحويل: {str(e)}"

# إدارة جلسة LiveKit
class LiveKitSession:
    def __init__(self):
        self.room: Optional[Room] = None
        self.transcription_box = st.empty()

    async def start(self):
        try:
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

                container = av.open(track, format="s16le", mode="r")
                stream = container.streams.audio[0]
                
                for frame in container.decode(stream):
                    audio_data = frame.to_ndarray()
                    
                    # تحويل معدل العينة إذا لزم الأمر
                    if stream.sample_rate != 16000:
                        audio_data = librosa.resample(
                            audio_data,
                            orig_sr=stream.sample_rate,
                            target_sr=16000
                        )
                    
                    # التحويل إلى نص
                    text = transcribe_audio(audio_data, 16000)
                    self.update_transcription(text)

        except Exception as e:
            st.error(f"فشل الاتصال: {str(e)}")

    def update_transcription(self, text: str):
        self.transcription_box.markdown(f"""
        <div class="rtl" style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
            <h3>النص المتحصل عليه:</h3>
            <p>{text}</p>
        </div>
        """, unsafe_allow_html=True)

    async def stop(self):
        if self.room:
            await self.room.disconnect()

# الواجهة الرئيسية
def main():
    session = LiveKitSession()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("الإعدادات")
        st.info("""
        <div class="rtl">
        - تأكد من السماح باستخدام الميكروفون في المتصفح
        - جودة التحويل تعتمد على جودة الصوت
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.header("التحكم")
        
        if st.button("بدء البث المباشر", type="primary"):
            with st.spinner("جاري الاتصال بغرفة LiveKit..."):
                asyncio.run(session.start())
                
        if st.button("إيقاف البث"):
            asyncio.run(session.stop())
            st.success("تم إيقاف البث بنجاح")

    # قسم لتحميل الملفات الصوتية
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
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
                            {text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"خطأ في معالجة الملف: {str(e)}")

if __name__ == "__main__":
    main()
