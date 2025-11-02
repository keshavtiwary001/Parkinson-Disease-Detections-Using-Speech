// webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; // stream from getUserMedia()
var rec;       // Recorder.js object
var input;     // MediaStreamAudioSourceNode
var audioContext;

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");

// Add event listeners
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
  console.log("🎙️ Record button clicked");
  var constraints = { audio: true, video: false };
  recordButton.disabled = true;
  stopButton.disabled = false;
  pauseButton.disabled = false;

  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    console.log("✅ getUserMedia success, initializing Recorder.js...");
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    gumStream = stream;
    input = audioContext.createMediaStreamSource(stream);
    rec = new Recorder(input, { numChannels: 1 });
    rec.record();
    console.log("🎧 Recording started");
  }).catch(function (err) {
    alert("⚠️ Microphone access denied or unavailable.");
    recordButton.disabled = false;
    stopButton.disabled = true;
    pauseButton.disabled = true;
  });
}

function pauseRecording() {
  console.log("⏸️ Pause button clicked", rec.recording);
  if (rec.recording) {
    rec.stop();
    pauseButton.innerHTML = "Resume";
  } else {
    rec.record();
    pauseButton.innerHTML = "Pause";
  }
}

function stopRecording() {
  console.log("⏹️ Stop button clicked");
  stopButton.disabled = true;
  recordButton.disabled = false;
  pauseButton.disabled = true;
  pauseButton.innerHTML = "Pause";
  rec.stop();
  gumStream.getAudioTracks()[0].stop();
  rec.exportWAV(uploadRecording);
}

function uploadRecording(blob) {
  console.log("📤 Uploading recording to Flask...");
  const formData = new FormData();
  const filename = new Date().toISOString() + ".wav";
  formData.append("file", blob, filename);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then(response => {
      if (!response.ok) throw new Error("Upload failed");
      return response.text();
    })
    .then(data => {
      console.log("✅ Upload success:", data);
      const fileParam = data.trim();
      console.log("➡️ Redirecting to loading with file:", fileParam);
      window.location.href = `/loading?file=${encodeURIComponent(fileParam)}`;
    })
    .catch(error => {
      console.error("❌ Upload error:", error);
      alert("Error uploading recording. Please try again.");
    });
}
