import streamlit as st
import asyncio
import soundfile as sf
import numpy as np
from livekit import Room, RoomOptions, TrackSource, create_token
from faster_whisper import WhisperModel
import tempfile
import os
import av

# Get LiveKit credentials from Streamlit secrets
LIVEKIT_URL = st.secrets["LIVEKIT_URL"]
LIVEKIT_API_KEY = st.secrets["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = st.secrets["LIVEKIT_API_SECRET"]

# Initialize Whisper model
@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="cpu")

model = load_model()

# Title and UI
st.title("🎙️ Real-Time Arabic Speech-to-Text with LiveKit & Whisper")

status_box = st.empty()
transcript_box = st.empty()

# Generate token for LiveKit connection
def generate_token(identity="streamlit-user"):
    return create_token(
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room="arabic-stt",
        identity=identity,
        ttl=3600
    )

# Start real-time session
async def start_session():
    token = generate_token()
    room = await Room.connect(LIVEKIT_URL, token, RoomOptions(auto_subscribe=True))
    status_box.info("🎧 Connected to LiveKit room. Waiting for audio...")

    @room.on("track_subscribed")
    async def on_audio_track(track, publication, participant):
        if track.kind != "audio":
            return

        container = av.open(track, format="s16le", mode="r")
        audio_frames = []

        for packet in container.demux():
            for frame in packet.decode():
                audio_frames.append(frame.to_ndarray())

                if len(audio_frames) >= 50:  # Process every ~50 frames
                    audio_data = np.concatenate(audio_frames)
                    audio_frames.clear()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        sf.write(tmpfile.name, audio_data, samplerate=16000)
                        segments, _ = model.transcribe(tmpfile.name, language="ar")
                        full_text = " ".join([seg.text for seg in segments])
                        transcript_box.markdown(f"📝 **Live Transcription:**\n\n{full_text}")

    await room.join()

# Streamlit button to trigger session
if st.button("🎤 Start Recording"):
    asyncio.run(start_session())
