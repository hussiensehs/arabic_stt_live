let recorder;

function Recorder(stream) {
    this.stream = stream;
    this.mediaRecorder = new MediaRecorder(stream);
    this.audioChunks = [];
    
    this.mediaRecorder.ondataavailable = (e) => {
        this.audioChunks.push(e.data);
    };
    
    this.record = function() {
        this.audioChunks = [];
        this.mediaRecorder.start();
    };
    
    this.stop = function() {
        this.mediaRecorder.stop();
    };
    
    this.exportWAV = function(cb, interval) {
        if (this.audioChunks.length) {
            let blob = new Blob(this.audioChunks, { type: 'audio/wav' });
            this.audioChunks = [];
            cb(blob);
        }
    };
}
