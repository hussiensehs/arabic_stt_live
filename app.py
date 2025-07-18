import streamlit as st
import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
import threading
import soundfile as sf
import queue
from threading import Lock

# إعداد واجهة Streamlit
st.set_page_config(
    page_title="Arabic Live Speech-to-Text",
    page_icon="🎙️",
    layout="wide"
)

# دعم النص العربي (RTL)
st.markdown("""
<style>
.rtl {
    direction: rtl;
    text-align: right;
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ تحويل الكلام العربي إلى نص مباشر")
st.markdown('<div class="rtl">تطبيق يحول الكلام العربي إلى نص مكتوب في الوقت الفعلي</div>', unsafe_allow_html=True)

# تحميل نموذج Whisper
@st.cache_resource
def load_model():
    return WhisperModel(
        "base",  # يمكن تغييرها إلى "tiny" للأجهزة الضعيفة
        device="cpu",
        compute_type="int8",
        download_root="./whisper_models"
    )

model = load_model()

# طابور لمعالجة البيانات الصوتية
audio_queue = queue.Queue()
text_lock = Lock()
latest_text = ""

# HTML و JavaScript لتسجيل الصوت
def audio_recorder_component():
    return f"""
    <!DOCTYPE html>
    <html>
    <body>
        <h3 style='text-align:center'>🎤 التحكم في التسجيل</h3>
        <div style='display:flex; justify-content:center; gap:20px; margin-bottom:20px'>
            <button onclick='startRecording()' style='
                padding: 10px 20px;
                background-color: #FF4B4B;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            '>▶️ بدء التسجيل</button>
            
            <button onclick='stopRecording()' style='
                padding: 10px 20px;
                background-color: #f0f2f6;
                color: #262730;
                border: 1px solid #FF4B4B;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            '>⏹️ إيقاف</button>
        </div>

        <script>
        let socket;
        let recorder;
        let audioContext;
        let processor;
        let stream;

        async function startRecording() {{
            try {{
                // إعداد اتصال WebSocket
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = wsProtocol + '//' + window.location.hostname + ':8502/audio';
                socket = new WebSocket(wsUrl);
                
                socket.onopen = () => {{
                    console.log("✅ تم الاتصال بالسيرفر");
                }};
                
                socket.onclose = () => {{
                    console.log("❌ تم إغلاق الاتصال");
                }};

                // بدء تسجيل الصوت
                stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                audioContext = new AudioContext();
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {{
                    if (socket.readyState === WebSocket.OPEN) {{
                        const audioData = e.inputBuffer.getChannelData(0);
                        const int16Data = new Int16Array(audioData.length);
                        for (let i = 0; i < audioData.length; i++) {{
                            int16Data[i] = Math.min(32767, Math.max(-32768, audioData[i] * 32768));
                        }}
                        socket.send(int16Data.buffer);
                    }}
                }};

                source.connect(processor);
                processor.connect(audioContext.destination);
                console.log("🎤 بدء التسجيل الصوتي");
            }} catch (error) {{
                console.error("Error:", error);
            }}
        }}

        function stopRecording() {{
            if (stream) {{
                stream.getTracks().forEach(track => track.stop());
            }}
            if (socket) {{
                socket.close();
            }}
            if (audioContext) {{
                audioContext.close();
            }}
            console.log("⏹️ توقف التسجيل");
        }}
        </script>
    </body>
    </html>
    """

# عرض مكون تسجيل الصوت
st.components.v1.html(audio_recorder_component(), height=150)

# مربع عرض النتائج
transcription_box = st.empty()

# معالجة الصوت وتحويله إلى نص
def process_audio():
    global latest_text
    while True:
        try:
            audio_data = audio_queue.get(timeout=1)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # تجاهل القطع الصوتية القصيرة جدًا
            if len(audio_np) < 16000:  # أقل من ثانية واحدة
                continue
                
            segments, _ = model.transcribe(
                audio_np,
                language="ar",
                vad_filter=True,
                beam_size=3
            )
            text = " ".join([seg.text for seg in segments])
            
            if text.strip():
                with text_lock:
                    latest_text = text
                    transcription_box.markdown(f"""
                    <div class="rtl" style="
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 10px;
                        margin-top: 20px;
                        border-left: 5px solid #FF4B4B;
                    ">
                        <p style='font-size: 18px;'>{latest_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"خطأ في المعالجة: {str(e)}")

# تشغيل خيط معالجة الصوت
processing_thread = threading.Thread(target=process_audio, daemon=True)
processing_thread.start()

# خادم WebSocket
async def audio_server(websocket, path):
    try:
        async for message in websocket:
            audio_queue.put(message)
    except websockets.exceptions.ConnectionClosed:
        pass

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(audio_server, "0.0.0.0", 8502)  # تغيير البورت إلى 8502
    loop.run_until_complete(server)
    loop.run_forever()

# تشغيل خادم WebSocket في خيط منفصل
websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
websocket_thread.start()

# قسم تحميل الملفات الصوتية
st.divider()
st.header("أو رفع ملف صوتي")
uploaded_file = st.file_uploader("اختر ملف صوتي", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    if st.button("تحويل الملف"):
        with st.spinner("جاري تحويل الصوت إلى نص..."):
            try:
                audio_data, sample_rate = sf.read(uploaded_file)
                if sample_rate != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                
                segments, _ = model.transcribe(audio_data, language="ar")
                text = " ".join([seg.text for seg in segments])
                
                st.markdown(f"""
                <div class="rtl" style="
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 10px;
                    margin-top: 20px;
                ">
                    <h3>نتيجة التحويل:</h3>
                    <p>{text}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"خطأ في معالجة الملف: {str(e)}")
