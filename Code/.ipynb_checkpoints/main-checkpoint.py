import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Change this to your own path to the dataset
DATASET_PATH = "/Users/ethanhyde/Downloads/archive/KAGGLE"

AUDIO = "AUDIO"
REAL_FOLDER = os.path.join(DATASET_PATH, AUDIO, "REAL")
FAKE_FOLDER = os.path.join(DATASET_PATH, AUDIO, "FAKE")

# For testing that the directory is correct
# print("REAL_FOLDER contents:", os.listdir(REAL_FOLDER))
# print("FAKE_FOLDER contents:", os.listdir(FAKE_FOLDER))

# Dataframe for processing. We can get rid of this if not needed
# Similar to a spreadsheet layout
df = pd.DataFrame(columns=['audio_file', 'LABEL'])

# Populate the DataFrame with all files in the 'REAL' folder
realFiles = [f for f in os.listdir(REAL_FOLDER) if os.path.isfile(os.path.join(REAL_FOLDER, f))]
dfReal = pd.DataFrame({'audio_file': realFiles, 'LABEL': 'REAL'})

# Appending
df = pd.concat([df, dfReal])

# Populate the DataFrame with all files in the 'FAKE' folder
fakeFiles = [f for f in os.listdir(FAKE_FOLDER) if os.path.isfile(os.path.join(FAKE_FOLDER, f))]
dfFake = pd.DataFrame({'audio_file': fakeFiles, 'LABEL': 'FAKE'})

# Appending
df = pd.concat([df, dfFake])

# Split the dataset into training and testing sets
# The 0.2 means that 20% will be used for testing, and the other 80% is used for training
# 42 is a seed value for RNG
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(train_df.columns)


# For processing audio files. Changing sample rate, getting audio data, etc.
def process_audio_files(dataframe, outputFolder):
    for index, row in dataframe.iterrows():
        audio_file = os.path.join(DATASET_PATH, AUDIO, row['audio_file'])
        # Code can go here to process the audio file

        # Save Processed data to the output folder
        output_file = os.path.join(outputFolder, f"{row['audio_file']}.npy")
        # Save the processed data

        print(f"Processing {audio_file} and saving to {output_file}")

# Process real audio files and save to REAL_FOLDER
process_audio_files(train_df[train_df['LABEL'] == 'REAL'], REAL_FOLDER)

# Process fake audio files and save to FAKE_FOLDER
process_audio_files(train_df[train_df['LABEL'] == 'FAKE'], FAKE_FOLDER)
