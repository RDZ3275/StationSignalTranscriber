import os
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from fastdtw import fastdtw

# Folder path containing recorded signal files
RECORDED_SIGNALS_FOLDER = "recorded_signals/"

# Recorded signal filenames
RECORDED_SIGNALS = [
    "A.wav",
    "B.wav",
    "C.wav",
    "D.wav",
    "E.wav",
    "F.wav",
    "G.wav",
    "H.wav",
    "I.wav",
    "J.wav",
    "K.wav",
    "L.wav",
    "M.wav",
    "N.wav",
    "O.wav",
    "P.wav",
    "Q.wav",
    "R.wav",
    "S.wav",
    "T.wav",
    "U.wav",
    "V.wav",
    "W.wav",
    "X.wav",
    "Y.wav",
    "Z.wav",
    "1.wav",
    "2.wav",
    "3.wav",
    "4.wav",
    "5.wav",
    "6.wav",
    "7.wav",
    "8.wav",
    "9.wav",
    "0.wav"
]

# Sampling rate and duration for audio recording
SAMPLING_RATE = 16000
RECORDING_DURATION = 2.0

def preprocess_audio(audio_data):
    # Convert audio data to mono
    if audio_data.shape[1] == 2:
        audio_data = np.mean(audio_data, axis=1)
    return audio_data

def calculate_features(audio_data):
    # Extract audio features using pyAudioAnalysis
    features, _ = audioFeatureExtraction.stFeatureExtraction(
        audio_data, SAMPLING_RATE, SAMPLING_RATE, 0.050, 0.025
    )
    return features

def match_audio_features(features, recorded_features):
    min_distance = float("inf")
    best_match = None

    # Perform DTW matching for each recorded signal feature
    for i, recorded_feature in enumerate(recorded_features):
        distance, _ = fastdtw(features.T, recorded_feature.T)
        if distance < min_distance:
            min_distance = distance
            best_match = os.path.splitext(RECORDED_SIGNALS[i])[0]

    return best_match

def main():
    # Load recorded signal features
    recorded_features = []
    for signal_file in RECORDED_SIGNALS:
        signal_path = os.path.join(RECORDED_SIGNALS_FOLDER, signal_file)
        signal = AudioSegment.from_wav(signal_path)
        signal_data = np.array(signal.get_array_of_samples())
        signal_data = preprocess_audio(signal_data)
        features = calculate_features(signal_data)
        recorded_features.append(features)

    # Set up audio recording
    recording_samples = int(SAMPLING_RATE * RECORDING_DURATION)

    print("Listening for signals...")

    while True:
        # Record audio from the microphone
        audio_data = sd.rec(recording_samples, samplerate=SAMPLING_RATE, channels=1)
        sd.wait()

        # Preprocess live audio
        audio_data = preprocess_audio(audio_data.flatten())

        # Calculate features for live audio
        live_features = calculate_features(audio_data)

        # Match live audio features with recorded signals
        match = match_audio_features(live_features, recorded_features)

        # Output the name of the matching signal recording file
        print("Match:", match)

if __name__ == "__main__":
    main()
