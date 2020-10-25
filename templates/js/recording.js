var video = document.querySelector('video');
var h1 = document.querySelector('h1');
var default_h1 = h1.innerHTML;

function captureCamera(callback) {
    navigator.mediaDevices.getUserMedia({ audio: true, video: true }).then(function(camera) {
        callback(camera);
    }).catch(function(error) {
        alert('Unable to capture your camera. Please check console logs.');
        console.error(error);
    });
}

function stopRecordingCallback() {
    video.srcObject = null;
    var blob = recorder.getBlob();
    video.src = URL.createObjectURL(blob);

    recorder.camera.stop();
    video.muted = false;
}

var recorder; // globally accessible

document.getElementById('btn-start-recording').onclick = function() {
    this.disabled = true;
    captureCamera(function(camera) {
        video.muted = true;
        video.srcObject = camera;

        recorder = RecordRTC(camera, {
            type: 'video'
        });

        recorder.startRecording();

        var max_seconds = 3;
        var stopped_speaking_timeout;
        var speechEvents = hark(camera, {});

        speechEvents.on('speaking', function() {
            if(recorder.getBlob()) return;

            clearTimeout(stopped_speaking_timeout);

            if(recorder.getState() === 'paused') {
                // recorder.resumeRecording();
            }
            
            h1.innerHTML = default_h1;
        });

        speechEvents.on('stopped_speaking', function() {
            if(recorder.getBlob()) return;

            // recorder.pauseRecording();
            stopped_speaking_timeout = setTimeout(function() {
                document.getElementById('btn-stop-recording').click();
                h1.innerHTML = 'Recording is now stopped.';
            }, max_seconds * 1000);

            
            // just for logging purpose (you ca remove below code)
            var seconds = max_seconds;
            (function looper() {
                h1.innerHTML = 'Recording is going to be stopped in ' + seconds + ' seconds.';
                seconds--;

                if(seconds <= 0) {
                    h1.innerHTML = default_h1;
                    return;
                }

                setTimeout(looper, 1000);
            })();
        });

        // release camera on stopRecording
        recorder.camera = camera;

        document.getElementById('btn-stop-recording').disabled = false;
    });
};

document.getElementById('btn-stop-recording').onclick = function() {
    this.disabled = true;
    recorder.stopRecording(stopRecordingCallback);
};