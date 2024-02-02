"""we will import all data here and launch all functionality"""
# import tensorflow_io as tfio # for some reason this line isn't working for me
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

DATASET_PATH = "/Users/ethanhyde/Downloads/archive/KAGGLE"  # Change this to your path
# Insert path here:


# Constants to hold paths to the dataset
AUDIO = "/AUDIO"
LABELS_FILE = os.path.join(DATASET_PATH, "DATASET-balanced.csv")
REAL_FOLDER = os.path.join(DATASET_PATH, AUDIO, "REAL")
FAKE_FOLDER = os.path.join(DATASET_PATH, AUDIO, "FAKE")

# Load labels from CSV
labels_df = pd.read_csv(LABELS_FILE)

# Use the index as an identifier, not sure if this is right, may have to look for a better way to do this
file_paths = [os.path.join(REAL_FOLDER if label == 'REAL' else FAKE_FOLDER, f"{index}.png") for index, label in labels_df[['LABEL']].itertuples(index=True, name=None)]
class_labels = labels_df['LABEL'].values

# Convert class labels to numerical labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(class_labels)

# Function to load and preprocess audio data
def load_and_preprocess_data(file_paths, numeric_labels):
    audio_data = []

    for file_path in file_paths:
        # Add code to load and preprocess audio data
     

    # Convert the list of audio data to numpy array
        audio_data = np.array(audio_data)

    return audio_data, numeric_labels

# Load and preprocess audio data
audio_data, numeric_labels = load_and_preprocess_data(file_paths, numeric_labels)

# Split the data into training and testing sets goes here


# Define the NN model
model = models.Sequential([
 
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
#model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
#test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: ")

