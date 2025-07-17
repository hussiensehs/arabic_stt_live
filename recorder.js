let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const file = new File([audioBlob], 'recorded_audio.wav', { type: 'audio/wav' });

                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);

                const uploader = document.querySelector('input[type="file"]');
                uploader.files = dataTransfer.files;
                uploader.dispatchEvent(new Event('change', { bubbles: true }));
            };
            audioChunks = [];
            mediaRecorder.start();
        });
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
    }
}
