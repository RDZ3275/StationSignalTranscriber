import os
import numpy as np
from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from fastdtw import fastdtw

# Folder path containing recorded audio files
RECORDED_AUDIO_FOLDER = "E:\_ELITE\_NEW ANTENA SOUNDs\App recording\template_alfabet/"

# Recorded audio filenames (in WAV format)
RECORDED_AUDIO_FILES = [
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
    "0.wav",
]

# Duration of the signal segment for matching (in seconds)
SIGNAL_DURATION = 2.0

def preprocess_audio(audio_file):
    # Load audio file using pydub
    audio = AudioSegment.from_file(audio_file)

    # Convert to mono and set frame rate to 16 kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Convert audio to numpy array
    audio_data = np.array(audio.get_array_of_samples())

    return audio_data

def calculate_features(audio_data):
    # Extract audio features using pyAudioAnalysis
    features, _ = audioFeatureExtraction.stFeatureExtraction(
        audio_data, 16000, 16000, 0.050, 0.025
    )

    return features

def match_audio_features(features, recorded_features):
    min_distance = float("inf")
    best_match = None

    # Perform DTW matching for each recorded audio feature
    for i, recorded_feature in enumerate(recorded_features):
        distance, _ = fastdtw(features.T, recorded_feature.T)
        if distance < min_distance:
            min_distance = distance
            best_match = os.path.splitext(RECORDED_AUDIO_FILES[i])[0]

    return best_match

def main():
    # Load recorded audio features
    recorded_features = []
    for audio_file in RECORDED_AUDIO_FILES:
        audio_path = os.path.join(RECORDED_AUDIO_FOLDER, audio_file)
        audio_data = preprocess_audio(audio_path)
        features = calculate_features(audio_data)
        recorded_features.append(features)

    # Set up audio recording using pyAudioAnalysis
    fs, audio_data = audioBasicIO.read_audio_file("", "live_audio.wav")

    # Calculate segment sizes based on signal and no signal durations
    signal_samples = int(fs * SIGNAL_DURATION)
    no_signal_samples = int(fs * 4)  # Assuming 4 seconds of no signal

    # Process each audio segment
    recognized_sequence = ""
    index = 0
    while index + signal_samples <= len(audio_data):
        # Extract the signal segment
        segment = audio_data[index : index + signal_samples]

        # Preprocess live audio segment
        live_features = calculate_features(segment)

        # Match live audio features with recorded audio
        match = match_audio_features(live_features, recorded_features)

        # Append the recognized match to the sequence
        recognized_sequence += match

        # Move to the next signal segment
        index += signal_samples + no_signal_samples

    print("Recognized Sequence:", recognized_sequence)

if __name__ == "__main__":
    main()
