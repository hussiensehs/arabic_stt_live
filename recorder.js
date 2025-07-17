let recorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        recorder = new Recorder(stream);
        audioChunks = [];
        recorder.record();
        streamToStreamlit();
    });
}

function streamToStreamlit() {
    if (recorder) {
        recorder.exportWAV(blob => {
            audioChunks.push(blob);
            // Send to Streamlit via a custom event or fetch (simplified)
            let reader = new FileReader();
            reader.onload = function(e) {
                let data = new Uint8Array(e.target.result);
                // Simulate sending to Streamlit (requires WebSocket or custom endpoint)
                console.log("Streaming audio chunk:", data);
            };
            reader.readAsArrayBuffer(blob);
            setTimeout(streamToStreamlit, 1000); // Stream every 1 second
        }, 1000); // Export every 1 second
    }
}

function stopRecording() {
    if (recorder) {
        recorder.stop();
        recorder.stream.getTracks().forEach(track => track.stop());
    }
}

// Expose functions to Streamlit
window.startRecording = startRecording;
window.stopRecording = stopRecording;
