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
import base64

# Page configuration
st.set_page_config(page_title="Arabic Speech-to-Text", page_icon="🎤", layout="centered")

# Title and description
st.title("Arabic Speech-to-Text App")
st.markdown("يتسجل الكلام مباشرة ويكتب النص فوراً باستخدام faster-whisper.")

# Initialize faster-whisper model with error handling
@st.cache_resource
def load_model():
    try:
        return faster_whisper.WhisperModel(
            model_size_or_path="base",
            device="cpu",
            download_root="/tmp",
            local_files_only=False
        )
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {str(e)}. حاول مرة أخرى أو تحقق من الاتصال.")
        return None

model = load_model()

# Session state
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []

# JavaScript for continuous recording
st.markdown("""
<script src="recorder.js"></script>
<script>
let recorder;
let streamInterval;

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        recorder = new Recorder(stream);
        recorder.record();
        streamInterval = setInterval(() => {
            recorder.exportWAV(blob => {
                let reader = new FileReader();
                reader.onload = function(e) {
                    let data = e.target.result.split(',')[1]; // Get base64 data
                    document.getElementById('audio-chunk').value = data;
                    document.getElementById('audio-form').dispatchEvent(new Event('submit'));
                };
                reader.readAsDataURL(blob);
            }, 1000);
        }, 1000);
    }).catch(error => {
        console.error('Microphone access error:', error);
        alert('خطأ في الوصول للميكروفون: ' + error.message);
    });
}

function stopRecording() {
    if (recorder) {
        recorder.stop();
        recorder.stream.getTracks().forEach(track => track.stop());
        clearInterval(streamInterval);
    }
}

window.startRecording = startRecording;
window.stopRecording = stopRecording;
</script>
<form id="audio-form" style="display:none;">
    <input type="hidden" id="audio-chunk" name="audio_chunk">
</form>
<div id="transcription"></div>
""", unsafe_allow_html=True)

# Handle audio chunk submission
if "audio_chunk" in st.form_submit_data and model is not None:
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(st.form_submit_data["audio_chunk"])
        audio = pydub.AudioSegment.from_file(BytesIO(audio_data))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        result, _ = model.transcribe(audio_data, language="ar")
        transcription = "".join(segment.text for segment in result)
        st.session_state.transcription += transcription + " "
        st.session_state.audio_chunks.append(transcription)
        st.write("<script>document.getElementById('transcription').innerText = '" + st.session_state.transcription.replace("'", "\\'") + "';</script>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"فشل معالجة التسجيل: {str(e)}")

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 ابدأ التسجيل") and model is not None:
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

# Footer
st.markdown("Built with Streamlit and faster-whisper for Arabic speech recognition.")
