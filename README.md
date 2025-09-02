AudioListenMet - Urdu Alphabet Recognition
A deep learning-based audio recognition system that identifies Urdu alphabet characters from spoken audio using a Siamese neural network architecture.

Overview
This project implements a Siamese neural network to recognize and match Urdu alphabet characters from audio input. The system can either record new audio or process uploaded audio files, then compare them against a pre-trained dataset of Urdu alphabet pronunciations.

Model Architecture
The system uses a Siamese neural network with the following architecture:

![Demo Screenshot](images\siamesearchitecture.jpg)

 
Encoder Details:

1. Conv1D Layer (64 filters, kernel size 5, ReLU activation)
2. MaxPooling1D (pool size 2)
3. Conv1D Layer (128 filters, kernel size 5, ReLU activation)
4. Global Average Pooling
5. Dense Layer (128 units, ReLU activation)

The model calculates the absolute difference between encoded features and passes them through a sigmoid function to produce a similarity score between 0 and 1.

Dataset
The model was trained on a custom dataset of Urdu alphabet pronunciations (urdualphabets11). The dataset contains multiple audio recordings for each Urdu alphabet character, with each recording featuring different speakers and variations in pronunciation.

Dataset Structure:

urdualphabets11/
├── alif/
│   ├── alif_1.m4a
│   ├── alif_2.m4a
│   └── ...
├── bay/
│   ├── bay_1.m4a
│   ├── bay_2.m4a
│   └── ...
└── ... (other Urdu characters)

Installation

Install required dependencies:
pip install -r requirements.txt


Usage
Run the main script:

python
python predictiion.py

Choose an option:
Record new audio (2-second recording)/Upload an existing audio file

The system will:
1. Record/upload audio
2. Remove background noise
3. Extract MFCC features
4. Compare with stored Urdu alphabet recordings
5. Return the best match with similarity score

Training Process

1. The model was trained using a Siamese network approach with:

2. Positive pairs: Different recordings of the same Urdu character

3. Negative pairs: Recordings of different Urdu characters

4. Data augmentation: Noise addition, pitch shifting, and time stretching

5. Optimizer: Adam with learning rate 1e-4

6. Loss function: Binary cross-entropy

The training achieved over 96% accuracy on the validation set after 15 epochs.

File Structure
AudioListenMet/
├── siamese_model_optimized1.keras    # trained model
├── prediction.py                     # Main application script
├── training.ipynb                    # Jupyter notebook with training code
├── recordings/                       # Directory for new recordings
└── dataurdu/                         # Dataset directory
    └── urdualphabets11/              # Urdu alphabet recordings

Results
The system can effectively:

1. Recognize Urdu alphabet characters from spoken audio
2. Handle variations in pronunciation and recording quality
3. Provide similarity scores to indicate confidence in matches

Example output:
🔍 Best Match: bay (Score: 0.9132)

Future Improvements
1. Expand dataset with more speakers and variations
2. Implement real-time continuous speech recognition
3. Add support for Urdu word and phrase recognition
4. Develop a user-friendly GUI interface

Acknowledgments
This project was developed as part of audio recognition research, specifically focused on Urdu language processing using deep learning techniques.