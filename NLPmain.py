import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
dataset = pd.read_csv("preprocessed_dataset.csv")

# Assuming you have text_messages and department_labels arrays
text_messages = dataset["Message"]
department_labels = dataset["Departement"]

# Convert department labels to numerical format
label_map = {label: index for index, label in enumerate(set(department_labels))}
numerical_labels = [label_map[label] for label in department_labels]

# Split dataset into training, validation, and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(
    text_messages, numerical_labels, test_size=0.3, random_state=42
)
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, test_size=0.2, random_state=42
)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_text, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_text, test_labels))

print("Train len: ", len(train_text))
print("Val len: ", len(val_text))
print("Test len: ", len(test_text))