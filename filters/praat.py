import parselmouth
import os
from parselmouth.praat import call
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class Praat:
    def getFeatures(self, audio_path: str, f0min=75, f0max=200):
        print(f"🎤 Extracting features from {audio_path}")
        sound = parselmouth.Sound(audio_path)
        features = {}

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        # Jitter & Shimmer Features
        features["Jitter(%)"] = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["Jitter(Abs)"] = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        features["Jitter:RAP"] = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features["Jitter:PPQ5"] = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        features["Shimmer"] = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features["Shimmer(dB)"] = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features["HNR"] = call(harmonicity, "Get mean", 0, 0)

        print("✅ Features extracted successfully.")
        return features

    def generateSpectrogram(self, audio_path: str, output_dir="data/spectrograms/tmp"):
        print(f"📊 Generating spectrogram for {audio_path}")
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use('agg')

        try:
            sound = parselmouth.Sound(audio_path)
            spectrogram = sound.to_spectrogram()
            X, Y = spectrogram.x_grid(), spectrogram.y_grid()
            sg_db = 10 * np.log10(spectrogram.values)

            plt.figure(figsize=(6, 4))
            plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - 70, cmap='afmhot')
            plt.axis('off')
            plt.grid(False)

            output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".wav", ".png"))
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"✅ Spectrogram saved at {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Spectrogram generation error: {e}")
            raise
