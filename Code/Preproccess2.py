import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
import librosa


def LoadWav16kMono(filename):
    """
    LoadWav16kMono - loads audio file and resamples it to 16k Hz
    filename: path to the audio file
    return: waveform sampled at 16k Hz
    """
    # Load encoded wav file
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

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
    """
    wav = LoadWav16kMono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# Change to your path
DATASET_PATH = 'c:/Users/redsl//PycharmProjects/pythonProject8/CS433_Project/KAGGLE/'
REAL_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'REAL', 'biden-original.wav')
FAKE_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'FAKE', 'biden-to-linus.wav')

wave = LoadWav16kMono(REAL_AUDIO)
nwave = LoadWav16kMono(FAKE_AUDIO)
plt.plot(nwave)
plt.plot(wave)
plt.show()

spectro, label = preprocess(REAL_AUDIO, 0.0)
plt.figure(figsize=(10, 5))
plt.imshow(tf.transpose(spectro)[0])
plt.show()
