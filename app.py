import streamlit as st
from datetime import datetime
import faster_whisper
import soundfile as sf
import numpy as np
import pydub
import librosa
import os
from io import BytesIO

# Page configuration
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")

# Title and description
st.title("Arabic Speech-to-Text App")
st.markdown("Record or upload audio to transcribe Arabic speech using faster-whisper.")

# Initialize faster-whisper model
@st.cache_resource
def load_model():
    return faster_whisper.WhisperModel("base", device="cpu")  # Use 'base' model for faster-whisper

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
    # Convert uploaded file to wav for faster-whisper
    audio = pydub.AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Standardize for whisper
    st.session_state.audio_data = BytesIO()
    audio.export(st.session_state.audio_data, format="wav")

# Transcription logic
if st.session_state.audio_data:
    st.write("Transcribing...")
    try:
        if isinstance(st.session_state.audio_data, str):
            # Load audio from file path (recorded via recorder.js)
            audio_data, sample_rate = sf.read(st.session_state.audio_data)
        else:
            # Load audio from uploaded file (BytesIO)
            audio_data, sample_rate = sf.read(st.session_state.audio_data)
        # Ensure mono audio and 16kHz sample rate
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sample_rate != 16000:
            st.warning("Resampling audio to 16kHz for transcription.")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        result, _ = model.transcribe(audio_data, language="ar")
        transcription = "".join(segment.text for segment in result)
        st.session_state.transcription = transcription
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
