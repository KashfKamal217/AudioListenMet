import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import noisereduce as nr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# ---- Define Custom Layer for Loading Model ----
class AbsDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

# ---- Load the Siamese Model ----
model_path = r"C:\Users\KASHF KAMAL\Documents\AudioListenMet\siamese_model_optimized2.keras"
model = load_model(model_path, custom_objects={'AbsDifference': AbsDifference})
print(f"✅ Model loaded from: {model_path}")

# ---- Feature Extraction ----
def preprocess(audio, sr=16000):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs.T[:80]

def pad_sequence(mfccs, max_length=80):
    return np.pad(mfccs, ((0, max(0, max_length - len(mfccs))), (0, 0)), mode='constant')

# ---- Record Audio ----
def record_audio(output_path, duration=2, sr=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording finished.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write(output_path, sr, audio)
    print(f"📂 Saved: {output_path}")

# ---- Remove Background Noise ----
def remove_noise(filepath):
    y, sr = librosa.load(filepath, sr=None)
    reduced = nr.reduce_noise(y=y, sr=sr)
    return reduced, sr

import tkinter as tk
from tkinter import filedialog

def upload_audio():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    root.attributes('-topmost', True)  # Bring dialog on top
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")]
    )
    root.destroy()  # Close the root window
    if file_path:
        print(f"✅ Uploaded file: {file_path}")
        return file_path
    else:
        print("❌ No file selected.")
        return None


# ---- Main Function: Record or Upload + Match ----
def match_audio(model, data_dir, option="record", duration=2):
    if option == "record":
        output_path = "recordings/new_audio.wav"
        record_audio(output_path, duration=duration)
    else:
        output_path = upload_audio()
        if not output_path:
            print("❌ No file selected.")
            return None, None

    # Step 2: Remove noise
    clean_audio, sr = remove_noise(output_path)
    
    # Step 3: Preprocess
    new_mfcc = preprocess(clean_audio, sr)
    new_mfcc = pad_sequence(new_mfcc)
    new_mfcc = np.expand_dims(new_mfcc, axis=0)
    
    best_score = -1
    best_label = None
    
    # Step 4: Compare with stored recordings
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            existing_audio, sr2 = librosa.load(file_path, sr=None)
            existing_mfcc = preprocess(existing_audio, sr2)
            existing_mfcc = pad_sequence(existing_mfcc)
            existing_mfcc = np.expand_dims(existing_mfcc, axis=0)
            
            score = model.predict([new_mfcc, existing_mfcc])[0][0]
            if score > best_score:
                best_score = score
                best_label = label
    
    return best_label, best_score

# ---- Run Matching ----
if __name__ == "__main__":
    data_directory = r"C:\Users\KASHF KAMAL\Documents\AudioListenMet\dataurdu1\alphabets\alphabets"
    
    # Choose option: record or upload
    print("\nChoose an option:\n1. Record New Audio\n2. Upload Audio File")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        predicted_label, similarity_score = match_audio(model, data_directory, option="record")
    else:
        predicted_label, similarity_score = match_audio(model, data_directory, option="upload")
    
    if predicted_label:
        print(f"🔍 Best Match: {predicted_label} (Score: {similarity_score:.4f})")
    else:
        print("⚠️ No match found.")


