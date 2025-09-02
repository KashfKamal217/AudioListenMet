# AudioListenMet - Urdu Alphabet Recognition

A deep learning-based audio recognition system that identifies Urdu alphabet characters from spoken audio using a **Siamese neural network architecture**.

---

## 📌 Overview
This project implements a Siamese neural network to recognize and match Urdu alphabet characters from audio input.  
The system can either **record new audio** or **process uploaded audio files**, then compare them against a pre-trained dataset of Urdu alphabet pronunciations.

---

## 🧠 Model Architecture
The system uses a **Siamese neural network** with the following architecture:

- **Conv1D Layer** (64 filters, kernel size 5, ReLU activation)  
- **MaxPooling1D** (pool size 2)  
- **Conv1D Layer** (128 filters, kernel size 5, ReLU activation)  
- **Global Average Pooling**  
- **Dense Layer** (128 units, ReLU activation)  

### 🔗 Similarity Calculation
- Compute absolute difference (L1 distance) between the two audio embeddings using a custom **AbsDifference** layer.  
- Pass the difference through a Dense layer with **sigmoid activation** to output a similarity score:  
  - `0 → Different audio`  
  - `1 → Same audio`

---

## 📊 Dataset
- Trained on a **custom dataset of Urdu alphabet pronunciations** (`urdualphabets11`).  
- Contains multiple recordings for each Urdu alphabet character.  
- Includes variations across different speakers and pronunciations.  

---

## ⚙️ Installation

Clone the repo and install dependencies:
```bash
git clone https://github.com/username/AudioListenMet.git
cd AudioListenMet
pip install -r requirements.txt

