import streamlit as st
from datetime import datetime
import faster_whisper
import soundfile as sf
import numpy as np
import pydub
import librosa
import os
from io import BytesIO
import time

# Page configuration
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")

# Title and description
st.title("Arabic Speech-to-Text App")
st.markdown("يتسجل الكلام مباشرة ويكتب النص فوراً باستخدام faster-whisper.")

# Initialize faster-whisper model
@st.cache_resource
def load_model():
    return faster_whisper.WhisperModel("base", device="cpu")

model = load_model()

# Session state
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

# JavaScript for continuous recording
st.markdown("""
<script src="recorder.js"></script>
<script>
let recorder;
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        recorder = new Recorder(stream);
        recorder.record();
        streamToStreamlit();
    });
}
function streamToStreamlit() {
    if (recorder) {
        recorder.exportWAV(blob => {
            let formData = new FormData();
            formData.append('audio', blob, 'chunk.wav');
            fetch(window.location.href, {
                method: 'POST',
                body: formData
            }).then(response => response.text()).then(data => {
                document.getElementById('transcription').innerText = data;
            });
            setTimeout(streamToStreamlit, 1000); // Stream every 1 second
        }, 1000);
    }
}
function stopRecording() {
    if (recorder) {
        recorder.stop();
        recorder.stream.getTracks().forEach(track => track.stop());
    }
}
window.startRecording = startRecording;
window.stopRecording = stopRecording;
</script>
<div id="transcription"></div>
""", unsafe_allow_html=True)

# Handle audio stream
if st._is_running_with_streamlit:
    import streamlit.server.server as server
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    @st.experimental_singleton
    def get_server():
        ctx = get_script_run_ctx()
        return ctx.server if ctx else server.Server.get_current()

    server = get_server()

    @server.route('/stream', methods=['POST'])
    def handle_audio_stream():
        import flask
        file = flask.request.files['audio']
        audio = pydub.AudioSegment.from_file(file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_data = np.array(audio.get_array_of_samples())
        result, _ = model.transcribe(audio_data, language="ar")
        transcription = "".join(segment.text for segment in result)
        st.session_state.transcription += transcription
        return transcription

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 ابدأ التسجيل"):
        st.write("<script>startRecording();</script>", unsafe_allow_html=True)
with col2:
    if st.button("🛑 إيقاف التسجيل"):
        st.write("<script>stopRecording();</script>", unsafe_allow_html=True)

# Display transcription
st.write("النص المحول:", st.session_state.transcription)

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
