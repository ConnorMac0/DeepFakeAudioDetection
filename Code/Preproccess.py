"""this file will contain functions that preprocess audio data before training"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

"""
LoadWav16kMono - converts audio file to 16k hertz wav

:arguments
    filename -audio file to be converted
    
The resulting wav will be used with tensorflow
"""
def LoadWav16kMono(filename):
    # Load encoded wav file as wav string "file_contents"
    file_contents = tf.io.read_file(filename)
    # Decode wav string
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Remove trailing axis
    wav = tf.squeeze(wav, axis=[-1])

    # RESIZING STILL PROBLEMATIC
    # sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Convert to 16000 hertz, used to reduce the size of final audio
    # Depending on the audio files the WILL NEED TO BE FINE TUNED
    # wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

    return wav


"""
PLotWav - takes input wav and plots it on visual graph

:arguments
    input_wav - preprocessed wav

The resulting graph will be displayed with matplotlib
"""
def PlotWav(input_wav):
    # This function can be expanded to show overlapping waves
    # and potentially save visual graphs later on
    plt.plot(input_wav)
    plt.show()


"""
preprocess - takes filepath and converts wav file to
                spectrogram

:arguments
    filepath - wav file
    label
The resulting spectrogram will be displayed with matplotlib
"""
def preprocess(file_path, label):
    wav = LoadWav16kMono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


""" MAIN TEST LAUNCH"""
# Change this to your own path to the dataset
DATASET_PATH = 'c:/Users/redsl//PycharmProjects/pythonProject8/CS433_Project/KAGGLE/'

REAL_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'REAL', 'biden-original.wav')
FAKE_AUDIO = os.path.join(DATASET_PATH, 'AUDIO', 'FAKE', 'biden-to-linus.wav')

# wave = LoadWav16kMono(REAL_AUDIO)
# nwave = LoadWav16kMono(FAKE_AUDIO)

spectro, label = preprocess(REAL_AUDIO, 0.0)
plt.figure(figsize=(30, 20))
plt.imshow(tf.transpose(spectro)[0])
plt.show()


# plt.plot(nwave)
# plt.plot(wave)
plt.show()
