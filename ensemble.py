import os
import pandas as pd
import torch
from torchvision import transforms, datasets
from torch import nn
from joblib import load
from pydub import AudioSegment
from filters.praat import Praat


# --- Random Forest Classification ---
def classify_using_saved_model(audio_sample):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "randomforest.joblib")

    if not os.path.exists(model_path):
        print("❌ RandomForest model not found at:", model_path)
        return None, None

    # Convert MP3 to WAV if needed
    if audio_sample.lower().endswith(".mp3"):
        wav_path = audio_sample.replace(".mp3", ".wav")
        try:
            AudioSegment.from_mp3(audio_sample).export(wav_path, format="wav")
            audio_sample = wav_path
            print(f"🎵 Converted MP3 → WAV: {wav_path}")
        except Exception as e:
            print(f"❌ MP3 conversion failed: {e}")
            return None, None

    # Extract Praat features
    praat = Praat()
    try:
        features = praat.getFeatures(audio_sample, 75, 200)
        print(f"✅ Extracted {len(features)} features using Praat.")
    except Exception as e:
        print(f"❌ Error extracting Praat features: {e}")
        return None, None

    df = pd.DataFrame([features])
    model = load(model_path)

    # Align feature order with training
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        label = model.predict(df)[0]
        score = model.predict_proba(df)[0][label]
        print(f"🌲 RandomForest Prediction: Label={label}, Confidence={score:.4f}")
        return label, score
    except Exception as e:
        print(f"❌ RandomForest prediction failed: {e}")
        return None, None


# --- ViT Classification (using spectrograms) ---
def classify_using_pytorch(audio_sample):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TMP_DIR = os.path.join(BASE_DIR, "tmp", "spectrograms", "0")
    os.makedirs(TMP_DIR, exist_ok=True)

    praat = Praat()
    try:
        praat.generateSpectrogram(audio_sample, TMP_DIR)
        print(f"🎧 Spectrogram generated at {TMP_DIR}")
    except Exception as e:
        print(f"❌ Spectrogram generation failed: {e}")
        return None, None

    model_path = os.path.join(BASE_DIR, "models", "vit.pth")
    if not os.path.exists(model_path):
        print("❌ ViT model not found at:", model_path)
        return None, None

    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        print("✅ ViT model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading ViT model: {e}")
        return None, None

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "tmp"), transform=transform)
    if len(dataset) == 0:
        print("❌ No spectrogram images found for ViT inference.")
        return None, None

    image = dataset[0][0].unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, label = torch.max(probs, 1)
        confidence = confidence.item()
        label = label.item()

    print(f"🧠 ViT Prediction: Label={label}, Confidence={confidence:.4f}")
    return label, confidence


# --- Final Combined Classifier (used by Flask) ---
def classify(audio_sample, use_ensemble=False):
    print(f"🔍 classify() called with file: {audio_sample}")

    try:
        # --- RandomForest-only (default) ---
        if not use_ensemble:
            label, conf = classify_using_saved_model(audio_sample)
            if label is None or conf is None:
                print("⚠️ RandomForest returned None values.")
                return None, None
            print(f"✅ Final (RF only): Label={label}, Confidence={conf:.4f}")
            return label, conf

        # --- Ensemble: RF + ViT ---
        label_rf, conf_rf = classify_using_saved_model(audio_sample)
        label_vit, conf_vit = classify_using_pytorch(audio_sample)

        if label_rf is None and label_vit is None:
            print("⚠️ Both models failed to classify.")
            return None, None

        # Weighted average (simple mean)
        confidence = (conf_rf + conf_vit) / 2 if (conf_rf and conf_vit) else conf_rf or conf_vit
        label_final = label_rf if label_rf is not None else label_vit

        print(f"✅ Final (Ensemble): Label={label_final}, Confidence={confidence:.4f}")
        return label_final, confidence

    except Exception as e:
        print(f"❌ Error in classify(): {e}")
        return None, None


if __name__ == "__main__":
    test_path = "mp3/test_audio.mp3"
    result, conf = classify(test_path, use_ensemble=False)
    print("🧩 Output:", result, conf)
