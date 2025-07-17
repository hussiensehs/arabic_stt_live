import streamlit as st
import os
import tempfile
from faster_whisper import WhisperModel
from datetime import datetime

st.set_page_config(page_title='🎙️ Arabic Live Speech-to-Text', layout='centered')

st.title('🎧 استمع وحوّل الكلام العربي إلى نص')

# لتحزين الجمل المتفرغة
if 'full_text' not in st.session_state:
    st.session_state.full_text = ''

# تحميل النموذج
@st.cache_resource
def load_model():
    return WhisperModel('base', compute_type='int8')

model = load_model()

# عرض الزر لتسجيل الصوت
st.markdown('''
    <script src="recorder.js"></script>
    <button onclick="startRecording()">🎤 ابدأ التسجيل</button>
    <button onclick="stopRecording()">🛑 إيقاف التسجيل</button>
''', unsafe_allow_html=True)

# استقبال الصوت بعد التسجيل
audio_file = st.file_uploader('⬆️ أو ارفع ملف الصوت هنا', type=['wav', 'mp3', 'm4a'])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    st.info('⏳ جاري تحويل الصوت للنص...')
    segments, _ = model.transcribe(tmp_path)

    for segment in segments:
        st.session_state.full_text += segment.text + ' '

    st.success('✅ النص ظهر بنجاح:')
    st.write(st.session_state.full_text)

# تحميل النص النهائي
if st.session_state.full_text:
    if st.button('⬇️ تحميل النص كملف'):
        filename = f'transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(st.session_state.full_text)
        with open(filename, 'rb') as f:
            st.download_button('📄 اضغط لتحميل النص', f, file_name=filename)
