from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import tensorflow as tf
import librosa.feature
import pandas as pd
import numpy as np

# Initialize flask server
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_file = 'DeepVoiceModel.pkl'

# load model from pickle file
with open(model_file, 'rb') as file:  
    model = pickle.load(file)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def LoadWav16kMono(filename):
    """
    LoadWav16kMono - loads audio file and resamples it to 16k Hz
    filename: path to the audio file
    return: waveform sampled at 16k Hz
    """
    # Loads encoded wav file, and resizes to beginning first seconds (16k Hz)
    # Change sr=16000 to sr=None to use entire .wav file
    wav, sample_rate = librosa.load(filename, sr=None, mono=True)
    return wav, sample_rate

def extract_features(wav, sample_rate):
    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=wav, sr=sample_rate)
    rms = librosa.feature.rms(y=wav)
    spectral_centroid = librosa.feature.spectral_centroid(y=wav, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=wav)
    mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=20)

    # Concatenate features into a single array
    features = [
        chroma_stft.mean(),
        rms.mean(),
        spectral_centroid.mean(),
        spectral_bandwidth.mean(),
        rolloff.mean(),
        zero_crossing_rate.mean(),
    ]
    features.extend(mfcc.mean(axis=1))

    return features

def process_audio(file_path, segment_duration=10, overlap=0.5):
    """
    Process all audio files in a folder.

    Parameters:
        folder_path (str): Path to the folder containing audio files.
        label (str): Label to assign to the audio files in this folder.

    Returns:
        list: List of extracted features from all audio files in the folder.
    """
    # Initialize an empty list to store features and labels
    data = []

    # loads the .wav data
    wav, sample_rate = LoadWav16kMono(file_path)
    segment_length = int(segment_duration * sample_rate)
    hop_length = int(segment_length * (1 - overlap))
    # Split audio into segments
    for i in range(0, len(wav) - segment_length + 1, hop_length):
        wav_segment = wav[i:i + segment_length]
        features = extract_features(wav_segment, sample_rate)
        for f in range(len(features)):
            features[f] = abs(features[f])
        data.append(features)


    return data

# API route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audioFile']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Preprocess the uploaded file
        processed_data = process_audio(filename)

        predicted = model.predict(np.array(processed_data))
        predicted = tf.squeeze(predicted)
        predicted = [1 if x >= 0.5 else 0 for x in predicted]

        if predicted.count(1) >= predicted.count(0):
            return jsonify({'result': "No Deep Fake Voice Detected"}), 200
        else:
            return jsonify({'result': "Deep Fake Voice Detected"}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == "__main__":
    app.run(debug=True)