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

   # LiveKit configuration
   LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://stt-arabic-jbyb69nd.livekit.cloud")
   LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "APIR2NRVYgdadun")
   LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")  # Set in Streamlit Cloud secrets

   # JavaScript for LiveKit client
   st.markdown("""
   <script src="https://unpkg.com/livekit-client@1.12.2/dist/livekit-client.min.js"></script>
   <script>
   async function startRecording() {
       console.log("Connecting to LiveKit...");
       const room = new LivekitClient.Room();
       try {
           await room.connect('""" + LIVEKIT_URL + """', 'token', {
               autoSubscribe: true
           });
           console.log("Connected to LiveKit room");
           document.getElementById('mic-status').innerText = 'الميكروفون متصل';
           const audioTrack = await LivekitClient.createLocalAudioTrack();
           room.localParticipant.publishTrack(audioTrack);
           console.log("Audio track published");
       } catch (error) {
           console.error('LiveKit connection error:', error);
           document.getElementById('mic-status').innerText = 'خطأ في الوصول للميكروفون: ' + error.message;
           alert('خطأ في الوصول للميكروفون: ' + error.message);
       }
   }

   function stopRecording() {
       console.log("Disconnecting from LiveKit");
       document.getElementById('mic-status').innerText = 'الميكروفون غير متصل';
   }

   window.startRecording = startRecording;
   window.stopRecording = stopRecording;
   </script>
   <div id="mic-status">غير متصل</div>
   """, unsafe_allow_html=True)

   # LiveKit audio processing (server-side)
   async def process_audio():
       if st.session_state.recording and model is not None:
           try:
               logger.info("Connecting to LiveKit server")
               room = livekit.Room()
               await room.connect(LIVEKIT_URL, livekit.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET))
               logger.info("Connected to LiveKit room")
               async for event in room.events():
                   if isinstance(event, livekit.AudioFrameEvent):
                       logger.info("Received audio frame")
                       audio_data = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
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
   if st.session_state.recording and model is not None:
       asyncio.run(process_audio())

   # Recording controls
   col1, col2 = st.columns(2)
   with col1:
       if st.button("🎤 ابدأ التسجيل") and model is not None:
           st.session_state.recording = True
           st.write("<script>startRecording();</script>", unsafe_allow_html=True)
           logger.info("Recording started")
   with col2:
       if st.button("🛑 إيقاف التسجيل"):
           st.session_state.recording = False
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
