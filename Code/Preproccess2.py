import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
import librosa.feature
import pandas as pd


def LoadWav16kMono(filename):
    """
    LoadWav16kMono - loads audio file and resamples it to 16k Hz
    filename: path to the audio file
    return: waveform sampled at 16k Hz
    """
    # Loads encoded wav file, and resizes to beginning first seconds (16k Hz)
    # Change sr=16000 to sr=None to use entire .wav file
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav, sample_rate

def PlotWav(input_wav):
    """
    Plots the waveform
    """
    plt.plot(input_wav)
    plt.show()

def preprocess(file_path, label):
    """
    loads and preprocesses audio file
    file_path: path to the audio file
    label: label associated with the audio file
    returns: spectrogram and label

    This is only for displaying a spectrogram
    """
    # Loads wav data
    wav = LoadWav16kMono(file_path)
    # adds zero padding for missing/empty files that are smaller than 48000
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    # Converts to spectrogram
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

def extract_features(wav, sample_rate, label):
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
    features.append(label)  # Append label

    return features

def process_audio_folder(folder_path, label, segment_duration=10, overlap=0.5):
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

    # Iterate through audio files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            # Get file path to a .wav file
            file_path = os.path.join(folder_path, filename)
            # loads the .wav data
            wav, sample_rate = LoadWav16kMono(file_path)
            segment_length = int(segment_duration * sample_rate)
            hop_length = int(segment_length * (1 - overlap))
            # Split audio into segments
            for i in range(0, len(wav) - segment_length + 1, hop_length):
                wav_segment = wav[i:i + segment_length]
                features = extract_features(wav_segment, sample_rate, label)
                data.append(features)

    return data

# Change to your path
DATASET_PATH = 'c:/Users/redsl//PycharmProjects/pythonProject8/CS433_Project/KAGGLE/'
REAL_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'REAL')
FAKE_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'FAKE')

# Initialize an empty list to store data
all_data = []

# preprocess all real audio
folder_data = process_audio_folder(REAL_AUDIO, 'REAL')
all_data.extend(folder_data)

# preprocess all fake audio
folder_data = process_audio_folder(FAKE_AUDIO, 'FAKE')
all_data.extend(folder_data)

# Convert data to DataFrame
columns = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
           'rolloff', 'zero_crossing_rate'] + [f'mfcc{i}' for i in range(1, 21)] + ['LABEL']
df = pd.DataFrame(all_data, columns=columns)

# Save DataFrame to CSV
df.to_csv('audio_features.csv', index=False)

# REAL_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'REAL', 'biden-original.wav')
# FAKE_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'FAKE', 'biden-to-linus.wav')

# wave = LoadWav16kMono(REAL_AUDIO)
# nwave = LoadWav16kMono(FAKE_AUDIO)
# plt.plot(nwave)
# plt.plot(wave)
# plt.show()

# spectro, label = preprocess(REAL_AUDIO, 0.0)
# plt.figure(figsize=(10, 5))
# plt.imshow(tf.transpose(spectro)[0])
# plt.show()
