import streamlit as st
from datetime import datetime
import faster_whisper
import soundfile as sf
import numpy as np
import pydub
import os
import logging
import asyncio
import livekit
from livekit.rtc import AccessToken, VideoGrant
import json

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit setup ---
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")
st.title("Arabic Speech-to-Text App")
st.markdown("يتسجل الكلام مباشرة ويكتب النص فوراً باستخدام faster-whisper و LiveKit.")

# --- Model caching ---
@st.cache_resource
def load_model():
    try:
        logger.info("Loading faster-whisper model...")
        model = faster_whisper.WhisperModel("tiny", device="cpu", download_root="/tmp")
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"فشل تحميل النموذج: {e}")
        return None

model = load_model()

# --- Session defaults ---
st.session_state.setdefault("transcription", "")
st.session_state.setdefault("recording", False)
st.session_state.setdefault("mic_status", "غير متصل")
st.session_state.setdefault("livekit_token", "")

# --- LiveKit secrets ---
LIVEKIT_URL = st.secrets.get("LIVEKIT_URL")
LIVEKIT_API_KEY = st.secrets.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = st.secrets.get("LIVEKIT_API_SECRET")

# --- Generate JWT ---
def generate_token():
    try:
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(f"user-{datetime.now().timestamp()}")
        token.with_name("Streamlit User")
        token.with_grant(VideoGrant(room_join=True, room="arabic-stt-room", can_publish=True, can_subscribe=True))
        token.with_ttl(3600)
        return token.to_jwt()
    except Exception as e:
        logger.error(f"Token error: {e}")
        st.error(f"فشل إنشاء رمز LiveKit: {e}")
        return ""

# --- Inject LiveKit Client JS ---
if st.session_state.recording and not st.session_state.livekit_token:
    st.session_state.livekit_token = generate_token()
    escaped_token = json.dumps(st.session_state.livekit_token)  # Proper escaping
    st.markdown(f"""
    <script>
    document.getElementById('livekit-token').value = {escaped_token};
    startRecording();
    </script>
    """, unsafe_allow_html=True)

st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/livekit-client@1.12.2/dist/livekit-client.min.js"></script>
<script>
let room;
async function startRecording() {
    const token = document.getElementById('livekit-token').value;
    if (!token) {
        document.getElementById('mic-status').innerText = 'خطأ: لا يوجد رمز';
        return;
    }
    try {
        room = new LivekitClient.Room();
        await room.connect('""" + LIVEKIT_URL + """', token);
        const track = await LivekitClient.createLocalAudioTrack();
        await room.localParticipant.publishTrack(track);
        document.getElementById('mic-status').innerText = 'الميكروفون متصل';
    } catch (err) {
        alert('فشل الاتصال بـ LiveKit: ' + err.message);
    }
}
async function stopRecording() {
    if (room) {
        await room.disconnect();
        document.getElementById('mic-status').innerText = 'الميكروفون غير متصل';
    }
}
window.startRecording = startRecording;
window.stopRecording = stopRecording;
</script>
<input type="hidden" id="livekit-token" value="">
<div id="mic-status">غير متصل</div>
""", unsafe_allow_html=True)

# --- Async audio handler ---
async def process_audio():
    try:
        room = livekit.Room()
        await room.connect(LIVEKIT_URL, st.session_state.livekit_token)
        logger.info("Connected to room.")
        async for event in room.events():
            if isinstance(event, livekit.rtc.AudioFrameEvent):
                audio = pydub.AudioSegment(
                    data=event.frame.data,
                    sample_width=2,
                    frame_rate=event.frame.sample_rate,
                    channels=event.frame.num_channels
                )
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                result, _ = model.transcribe(audio_data, language="ar")
                text = "".join(seg.text for seg in result)
                st.session_state.transcription += text + " "
                st.experimental_rerun()
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        st.error(f"فشل معالجة الصوت: {e}")

# --- Manual Upload ---
uploaded = st.file_uploader("أو اختر ملف صوتي", type=["wav"])
if uploaded:
    audio = pydub.AudioSegment.from_file(uploaded).set_channels(1).set_frame_rate(16000)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    result, _ = model.transcribe(audio_data, language="ar")
    st.session_state.transcription += "".join(seg.text for seg in result) + " "
    st.experimental_rerun()

# --- Recording Controls ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 ابدأ التسجيل"):
        st.session_state.recording = True
        st.session_state.livekit_token = generate_token()
with col2:
    if st.button("🛑 إيقاف التسجيل"):
        st.session_state.recording = False
        st.session_state.livekit_token = ""
        st.markdown("<script>stopRecording();</script>", unsafe_allow_html=True)

# --- Async audio trigger ---
if st.session_state.recording and model and st.session_state.livekit_token:
    if "audio_task" not in st.session_state:
        st.session_state.audio_task = asyncio.create_task(process_audio())

# --- Transcription Output ---
st.write("### النص المحول")
st.write(st.session_state.transcription)

# --- Download ---
if st.session_state.transcription:
    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("⬇️ تحميل النص", data=st.session_state.transcription, file_name=filename, mime="text/plain")

# --- Footer ---
st.markdown("تم التطوير باستخدام Streamlit و faster-whisper و LiveKit.")
