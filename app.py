import streamlit as st
from datetime import datetime
import faster_whisper
import soundfile as sf
import numpy as np
import pydub
import librosa
import os
from io import BytesIO
import livekit
import logging
import asyncio
from livekit.rtc import AccessToken, VideoGrant  # Correct import for livekit==0.12.1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")

# Title and description
st.title("Arabic Speech-to-Text App")
st.markdown("يتسجل الكلام مباشرة ويكتب النص فوراً باستخدام faster-whisper و LiveKit.")

# Initialize faster-whisper model
@st.cache_resource
def load_model():
    try:
        logger.info("Attempting to load faster-whisper model...")
        model = faster_whisper.WhisperModel(
            model_size_or_path="tiny",
            device="cpu",
            download_root="/tmp",
            local_files_only=False
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"فشل تحميل النموذج: {str(e)}. حاول مرة أخرى أو تحقق من الاتصال.")
        return None

model = load_model()

# Session state
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "recording" not in st.session_state:
    st.session_state.recording = False
if "mic_status" not in st.session_state:
    st.session_state.mic_status = "غير متصل"
if "livekit_token" not in st.session_state:
    st.session_state.livekit_token = ""

# LiveKit configuration
LIVEKIT_URL = st.secrets.get("LIVEKIT_URL", "wss://stt-arabic-jbyb69nd.livekit.cloud")
LIVEKIT_API_KEY = st.secrets.get("LIVEKIT_API_KEY", "APIR2NRVYgdadun")
LIVEKIT_API_SECRET = st.secrets.get("LIVEKIT_API_SECRET", "vI1P58EouVNhTe3U0KpTSy1ffBnH9k2D91fJRcJhl6jA")

# Generate LiveKit access token
def generate_token():
    try:
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity("user-" + str(datetime.now().timestamp()))
        token.with_name("Streamlit User")
        token.with_grant(VideoGrant(
            room_join=True,
            room="arabic-stt-room",
            can_publish=True,
            can_subscribe=True
        ))
        token.with_ttl(3600)  # 1 hour
        return token.to_jwt()
    except Exception as e:
        logger.error(f"Token generation failed: {str(e)}")
        st.error(f"فشل إنشاء رمز LiveKit: {str(e)}")
        return ""

# JavaScript for LiveKit client
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/livekit-client@1.12.2/dist/livekit-client.min.js"></script>
<script>
let room;
async function startRecording() {
    console.log("Connecting to LiveKit...");
    const token = document.getElementById('livekit-token').value;
    if (!token) {
        console.error("No LiveKit token provided");
        document.getElementById('mic-status').innerText = 'خطأ: لا يوجد رمز LiveKit';
        alert('خطأ: لا يوجد رمز LiveKit');
        return;
    }
    try {
        room = new LivekitClient.Room();
        await room.connect('""" + LIVEKIT_URL + """', token);
        console.log("Connected to LiveKit room");
        document.getElementById('mic-status').innerText = 'الميكروفون متصل';
        const audioTrack = await LivekitClient.createLocalAudioTrack();
        await room.localParticipant.publishTrack(audioTrack);
        console.log("Audio track published");
    } catch (error) {
        console.error('LiveKit connection error:', error);
        document.getElementById('mic-status').innerText = 'خطأ في الوصول للميكروفون: ' + error.message;
        alert('خطأ في الوصول للميكروفون: ' + error.message);
    }
}

async function stopRecording() {
    if (room) {
        console.log("Disconnecting from LiveKit");
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

# Set LiveKit token in session state and update DOM
if st.session_state.recording and not st.session_state.livekit_token:
    st.session_state.livekit_token = generate_token()
    escaped_token = st.session_state.livekit_token.replace("'", "\\'")
    st.markdown(f"""
    <script>
    document.getElementById('livekit-token').value = '{escaped_token}';
    </script>
    """, unsafe_allow_html=True)

# LiveKit audio processing
async def process_audio():
    if st.session_state.recording and model is not None:
        try:
            logger.info("Connecting to LiveKit server")
            room = livekit.Room()
            await room.connect(LIVEKIT_URL, st.session_state.livekit_token)
            logger.info("Connected to LiveKit room")
            async for event in room.events():
                if isinstance(event, livekit.rtc.AudioFrameEvent):  # Updated for livekit==0.12.1
                    logger.info("Received audio frame")
                    st.write("تلقي إطار صوتي")  # Debug: confirm frame received
                    audio = pydub.AudioSegment(
                        data=event.frame.data,
                        sample_width=2,
                        frame_rate=event.frame.sample_rate,
                        channels=event.frame.num_channels
                    )
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                    result, _ = model.transcribe(audio_data, language="ar")
                    transcription = "".join(segment.text for segment in result)
                    st.session_state.transcription += transcription + " "
                    logger.info(f"Transcription: {transcription}")
                    st.experimental_rerun()
        except Exception as e:
            logger.error(f"LiveKit audio processing failed: {str(e)}")
            st.error(f"فشل معالجة التسجيل: {str(e)}")

# Run LiveKit audio processing in background
if st.session_state.recording and model is not None and st.session_state.livekit_token:
    try:
        asyncio.run(process_audio())
    except Exception as e:
        logger.error(f"Async audio processing failed: {str(e)}")
        st.error(f"فشل معالجة الصوت: {str(e)}")

# Manual file uploader for testing
uploaded_file = st.file_uploader("Upload audio (optional)", type=["wav"], key="manual_upload", label_visibility="collapsed")
if uploaded_file is not None and model is not None:
    try:
        logger.info("Processing manual audio upload")
        audio = pydub.AudioSegment.from_file(uploaded_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        result, _ = model.transcribe(audio_data, language="ar")
        transcription = "".join(segment.text for segment in result)
        st.session_state.transcription += transcription + " "
        logger.info(f"Manual transcription: {transcription}")
        st.experimental_rerun()
    except Exception as e:
        logger.error(f"Manual audio processing failed: {str(e)}")
        st.error(f"فشل معالجة التسجيل: {str(e)}")

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 ابدأ التسجيل") and model is not None:
        st.session_state.recording = True
        st.session_state.livekit_token = generate_token()
        st.write("<script>startRecording();</script>", unsafe_allow_html=True)
        logger.info("Recording started")
with col2:
    if st.button("🛑 إيقاف التسجيل"):
        st.session_state.recording = False
        st.session_state.livekit_token = ""
        st.write("<script>stopRecording();</script>", unsafe_allow_html=True)
        logger.info("Recording stopped")

# Display transcription
st.write("النص المحول:", st.session_state.transcription)

# Display microphone status
st.write("حالة الميكروفون:", st.session_state.mic_status)

# Download transcription
if st.session_state.transcription:
    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(st.session_state.transcription)
    with open(filename, "rb") as f:
        st.download_button(
            label="⬇️ تحميل النص كملف",
            data=f,
            file_name=filename,
            mime="text/plain"
        )
    os.remove(filename)

# Footer
st.markdown("Built with Streamlit, faster-whisper, and LiveKit for Arabic speech recognition.")
