import streamlit as st
from datetime import datetime
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import os
import logging
import asyncio
import json

from livekit_server_sdk import AccessToken, VideoGrant  # ✅ التعديل هنا

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit setup ---
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")
st.title("Arabic Speech-to-Text App")
st.markdown("🎙️ يتم تسجيل الصوت وتحويله إلى نص مباشرة باستخدام faster-whisper و LiveKit")

# --- Model caching ---
@st.cache_resource
def load_model():
    try:
        model = WhisperModel("tiny", device="cpu", download_root="/tmp")
        return model
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
        return None

model = load_model()

# --- Session defaults ---
st.session_state.setdefault("transcription", "")
st.session_state.setdefault("recording", False)
st.session_state.setdefault("mic_status", "غير متصل")
st.session_state.setdefault("livekit_token", "")

# --- Secrets ---
LIVEKIT_URL = st.secrets.get("LIVEKIT_URL")
LIVEKIT_API_KEY = st.secrets.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = st.secrets.get("LIVEKIT_API_SECRET")

# --- Generate JWT token ---
def generate_token():
    try:
        identity = f"user-{datetime.now().timestamp()}"
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, identity=identity)
        grant = VideoGrant(room="arabic-stt-room")
        token.add_grant(grant)
        return token.to_jwt()
    except Exception as e:
        st.error(f"فشل إنشاء رمز LiveKit: {e}")
        return ""

# --- Inject JS Client ---
if st.session_state.recording and not st.session_state.livekit_token:
    st.session_state.livekit_token = generate_token()
    escaped_token = json.dumps(st.session_state.livekit_token)
    st.markdown(f"""
    <script>
    document.getElementById('livekit-token').value = {escaped_token};
    startRecording();
    </script>
    """, unsafe_allow_html=True)

# --- Inject JavaScript for LiveKit client ---
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/livekit-client@1.12.2/dist/livekit-client.min.js"></script>
<script>
let room;
async function startRecording() {
    const token = document.getElementById('livekit-token').value;
    if (!token) {
        document.getElementById('mic-status').innerText = '⚠️ لا يوجد رمز';
        return;
    }
    try {
        room = new LivekitClient.Room();
        await room.connect('""" + LIVEKIT_URL + """', token);
        const track = await LivekitClient.createLocalAudioTrack();
        await room.localParticipant.publishTrack(track);
        document.getElementById('mic-status').innerText = '🎤 الميكروفون متصل';
    } catch (err) {
        alert('فشل الاتصال بـ LiveKit: ' + err.message);
    }
}
async function stopRecording() {
    if (room) {
        await room.disconnect();
        document.getElementById('mic-status').innerText = '🛑 الميكروفون غير متصل';
    }
}
window.startRecording = startRecording;
window.stopRecording = stopRecording;
</script>
<input type="hidden" id="livekit-token" value="">
<div id="mic-status">غير متصل</div>
""", unsafe_allow_html=True)

# --- Audio Upload (manual) ---
uploaded = st.file_uploader("📤 أو اختر ملف صوتي", type=["wav", "mp3", "m4a"])
if uploaded:
    audio = AudioSegment.from_file(uploaded).set_channels(1).set_frame_rate(16000)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    segments, _ = model.transcribe(audio_data, language="ar")
    st.session_state.transcription += "".join(seg.text for seg in segments) + " "
    st.experimental_rerun()

# --- Recording Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ ابدأ التسجيل"):
        st.session_state.recording = True
        st.session_state.livekit_token = generate_token()
with col2:
    if st.button("🛑 إيقاف التسجيل"):
        st.session_state.recording = False
        st.session_state.livekit_token = ""
        st.markdown("<script>stopRecording();</script>", unsafe_allow_html=True)

# --- Transcription Output ---
st.markdown("## 📝 النص المحول")
st.write(st.session_state.transcription)

# --- Download Transcript ---
if st.session_state.transcription:
    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("⬇️ تحميل النص", data=st.session_state.transcription, file_name=filename, mime="text/plain")

# --- Footer ---
st.markdown("---")
st.markdown("تم التطوير باستخدام Streamlit و Faster-Whisper و LiveKit 🎧")
