import streamlit as st
from datetime import datetime
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import os
from io import BytesIO

# Page configuration
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")

# Title and description
st.title("Arabic Speech-to-Text App")
st.markdown("Record or upload audio to transcribe Arabic speech using faster-whisper.")

# Initialize whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base", device="cpu")  # Use 'base' model for faster-whisper

model = load_model()

# Session state for audio and transcription
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

# JavaScript for microphone recording (via recorder.js)
st.markdown("""
<script src="recorder.js"></script>
<script>
let recorder;
function startRecording() {
    navigator.mediaDevices.getUserMedia({audio: true}).then(stream => {
        recorder = new Recorder(stream);
        recorder.record();
    });
}
function stopRecording() {
    recorder.stop();
    recorder.exportWAV(blob => {
        let url = URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.href = url;
        a.download = 'recording.wav';
        a.click();
    });
}
</script>
""", unsafe_allow_html=True)

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 ابدأ التسجيل"):
        st.write("<button onclick='startRecording()'>Recording...</button>", unsafe_allow_html=True)
with col2:
    if st.button("🛑 إيقاف التسجيل"):
        st.write("<button onclick='stopRecording()'>Processing...</button>", unsafe_allow_html=True)
        # Assume recording.wav is saved locally by recorder.js
        if os.path.exists("recording.wav"):
            st.session_state.audio_data = "recording.wav"

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
if uploaded_file:
    st.session_state.audio_data = uploaded_file

# Transcription logic
if st.session_state.audio_data:
    st.write("Transcribing...")
    try:
        if isinstance(st.session_state.audio_data, str):
            # Load audio from file path (recorded via recorder.js)
            audio = st.session_state.audio_data
        else:
            # Load audio from uploaded file
            audio = BytesIO(st.session_state.audio_data.read())
        result = model.transcribe(audio, language="ar")
        st.session_state.transcription = result["text"]
        st.write("Transcription:", st.session_state.transcription)
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")

# Download transcription as text file
if st.session_state.transcription:
    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"  # Fixed f-string
    with open(filename, "w", encoding="utf-8") as f:
        f.write(st.session_state.transcription)
    with open(filename, "rb") as f:
        st.download_button(
            label="⬇️ تحميل النص كملف",
            data=f,
            file_name=filename,
            mime="text/plain"
        )
    os.remove(filename)  # Clean up temporary file

# Footer
st.markdown("Built with Streamlit and faster-whisper for Arabic speech recognition.")
