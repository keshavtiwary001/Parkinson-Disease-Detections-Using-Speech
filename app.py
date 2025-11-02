import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
import os
import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import spectrogram
from models.ensemble import classify  # ✅ Main function to call


app = Flask(__name__)

# --------------------------
# Folder setup
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_FOLDER = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = TMP_FOLDER


# --------------------------
# Helper: Convert MP3 to WAV and generate spectrogram
# --------------------------
def create_spectrogram(audio_path, save_path):
    """Converts audio to WAV if needed and creates a spectrogram. Returns final WAV path."""
    try:
        # Convert to WAV if not already
        if not audio_path.lower().endswith(".wav"):
            wav_path = os.path.splitext(audio_path)[0] + ".wav"
            audio = AudioSegment.from_file(audio_path, format=audio_path.split('.')[-1])
            audio.export(wav_path, format="wav")
            audio_path = wav_path
            print(f"🎧 Converted to WAV: {wav_path}")

        # Generate spectrogram
        rate, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data[:, 0]
        f, t, Sxx = spectrogram(data, rate)
        Sxx = np.where(Sxx == 0, 1e-10, Sxx)
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"✅ Spectrogram saved at {save_path}")

        return audio_path  # ✅ Return final WAV path

    except Exception as e:
        print(f"❌ Error generating spectrogram: {e}")
        return None



# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("failure.html")

    file = request.files["file"]
    if file.filename == "":
        return render_template("failure.html")

    # Save file to tmp folder
    filename = secure_filename(file.filename)
    save_path = os.path.join(TMP_FOLDER, filename)
    file.save(save_path)
    print(f"✅ File saved at {save_path}")

    # Redirect to loading page, passing file name as query parameter
    return redirect(url_for("loading", file=filename))


@app.route("/loading")
def loading():
    file_name = request.args.get("file")
    return render_template("loading.html", file_name=file_name)


@app.route("/execute_pipeline")
def execute_pipeline():
    file_name = request.args.get("file")
    if not file_name:
        return render_template("failure.html")

    try:
        # File path
        audio_path = os.path.join(TMP_FOLDER, file_name)

        # Create spectrogram and get the new WAV path
        spectrogram_path = audio_path.replace(".mp3", ".png").replace(".wav", ".png").replace(".m4a", ".png")
        processed_audio = create_spectrogram(audio_path, spectrogram_path)
        if not processed_audio:
            return render_template("failure.html")

        # ✅ Use converted WAV file
        label, confidence = classify(processed_audio, use_ensemble=False)

        if label is None or confidence is None:
            return render_template("failure.html")

        result_label = "Parkinson" if int(label) == 1 else "Healthy"
        confidence_pct = f"{confidence * 100:.2f}%"

        print(f"🎯 Final Result: {result_label} ({confidence_pct})")

        return render_template(
            "results.html",
            prediction=result_label,
            probability=confidence_pct,
            file_name=file_name
        )

    except Exception as e:
        print(f"❌ Error in execute_pipeline: {e}")
        return render_template("failure.html")



if __name__ == "__main__":
    app.run(debug=True)
