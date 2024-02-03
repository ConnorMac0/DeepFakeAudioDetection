"""this file will contain functions that we preprocess audio data before training"""
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt

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
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    # Convert to 16000 hertz, used to reduce the size of final audio
    # Depending on the audio files the WILL NEED TO BE FINE TUNED
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


"""
PLotWav - takes input wav and plots it on visual graph

:arguments
    input_wav - preprocessed wav

The resulting graph will be displayed with matplotlib
"""
def PlotWav(input_wav):
    # This function can be expanded to sho overlapping waves
    # and potentially save visual graphs later on
    plt.plot(input_wav)
    plt.show()
