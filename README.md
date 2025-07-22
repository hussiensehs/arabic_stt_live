# Arabic Live Speech-to-Text

A web-based application built with Streamlit to transcribe Arabic speech in real-time using the browser's microphone. The app converts spoken Arabic to text and allows users to download the transcription as a `.txt` file. It runs entirely online, requiring no local installation.

## Features
- Record audio directly in the browser using the microphone.
- Real-time transcription of Arabic speech using the `faster-whisper` model.
- Download transcribed text as a `.txt` file.
- Option to upload audio files (`.wav`, `.mp3`, `.m4a`) for transcription.
- Fully hosted on Streamlit Cloud for online access.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hussiensehs/arabic_stt_live.git
   cd arabic_stt_live
   ```
2. **Install Dependencies** (for local testing, optional):
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `ffmpeg` is installed on your system for audio processing.
3. **Run Locally** (optional):
   ```bash
   streamlit run app.py
   ```
4. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://share.streamlit.io).
   - Connect your GitHub account and select the `hussiensehs/arabic_stt_live` repository.
   - Set `app.py` as the main file and click "Deploy".
   - Access the app at the provided URL (e.g., `https://hussiensehs-arabic-stt-live.streamlit.app`).

## Usage
1. Open the deployed app URL in a browser (e.g., Chrome, Firefox).
2. Allow microphone access when prompted.
3. Click "🎤 ابدأ التسجيل" to start recording Arabic speech.
4. Click "🛑 إيقاف التسجيل" to stop and view the transcribed text.
5. Click "⬇️ تحميل النص كملف" to download the transcription as a `.txt` file.
6. Optionally, upload an audio file (`.wav`, `.mp3`, `.m4a`) to transcribe.

## Dependencies
Listed in `requirements.txt`:
- streamlit
- faster-whisper
- pydub
- ffmpeg-python
- soundfile
- numpy
- torch

## Notes
- The app uses the `faster-whisper` `base` model for Arabic transcription, optimized for performance.
- Ensure your browser supports the MediaRecorder API (Chrome, Firefox, Edge).
- For better accuracy, consider upgrading to the `medium` model (future enhancement).

## Future Enhancements
- Automatic language detection.
- Support for multiple `faster-whisper` models (e.g., `tiny`, `medium`).
- Translation of transcribed text.
- Recording timer display.
- Email export for transcriptions.

## License
MIT License
