import os
import librosa
import matplotlib.pyplot as plt 
import tensorflow as tf

def load_wav_16k_mono(filename):
    """
    LoadWav16kMono - loads audio file and resamples it to 16k Hz
    
    filename: path to the audio file
    return: waveform sampled at 16k Hz
    """
    wav, _ = librosa.load(filename, sr=16000, mono=True)
    return wav

def plot_wav(input_wav):
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

    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000 - tf.shape(wav)[0]], dtype=tf.float32) 
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# Change to your path
DATASET_PATH = '/Users/ethanhyde/Downloads/archive/KAGGLE'
REAL_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'REAL', 'biden-original.wav')
FAKE_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'FAKE', 'biden-to-linus.wav')

wave = load_wav_16k_mono(REAL_AUDIO)
nwave = load_wav_16k_mono(FAKE_AUDIO)

spectro, label = preprocess(REAL_AUDIO, 0.0)
plt.figure(figsize=(30, 20))
plt.imshow(tf.transpose(spectro)[0])
plt.show()

plt.plot(nwave)
plt.plot(wave)
plt.show()
