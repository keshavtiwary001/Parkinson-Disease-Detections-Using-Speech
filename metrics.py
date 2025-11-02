import os
from models import ensemble
from pydub import AudioSegment  # for format conversion

# -------------------------------
# Folder containing test audios
# -------------------------------
test_dir = "mp3"  # 👈 change this if needed (can contain wav/mp3/m4a files)

if not os.path.exists(test_dir):
    print(f"❌ Folder not found: {test_dir}")
    exit()

# -------------------------------
# Helper: Convert audio to WAV if needed
# -------------------------------
def convert_to_wav_if_needed(filepath):
    """
    Converts .mp3 or .m4a to .wav if required and returns the new path.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".mp3", ".m4a"]:
        wav_path = filepath.replace(ext, ".wav")
        if not os.path.exists(wav_path):
            try:
                audio = AudioSegment.from_file(filepath, format=ext.replace(".", ""))
                audio.export(wav_path, format="wav")
                print(f"🎵 Converted {os.path.basename(filepath)} → {os.path.basename(wav_path)}")
            except Exception as e:
                print(f"❌ Conversion failed for {os.path.basename(filepath)}: {e}")
                return None
        return wav_path
    return filepath


print(f"🎧 Evaluating audio files from: {test_dir}\n")

# -------------------------------
# Loop through all audio files
# -------------------------------
for filename in os.listdir(test_dir):
    if not (filename.lower().endswith((".wav", ".mp3", ".m4a"))):
        continue

    filepath = os.path.join(test_dir, filename)
    filepath = convert_to_wav_if_needed(filepath)

    if filepath is None or not os.path.exists(filepath):
        print(f"⚠️ Skipped {filename}: Conversion failed or file missing\n")
        continue

    print(f"🔹 Processing: {filename}")

    try:
        # Run prediction using ensemble model
        result, confidence = ensemble.classify(filepath)

        print(f"✅ Result: {result}")
        print(f"   Confidence: {confidence if confidence != 'N/A' else 'Unavailable'}\n")

    except Exception as e:
        print(f"⚠️ Skipped {filename}: {e}\n")

print("🎯 Evaluation complete.")
