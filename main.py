# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
!pip install -q seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')

# Use log transformation to normalize data
columns_to_transform = ['bmi', 'expenses']
transformed_columns = np.log(dataset[columns_to_transform])

# Replace the original columns with the transformed values
dataset[columns_to_transform] = transformed_columns

dataset.tail()

# Change catagorical variables to numeric

# Perform one-hot encoding
dataset = pd.get_dummies(dataset, columns=['sex'], prefix='', prefix_sep='')
dataset = pd.get_dummies(dataset, columns=['smoker'], prefix='', prefix_sep='')
dataset = pd.get_dummies(dataset, columns=['region'], prefix='', prefix_sep='')

# Split into training and test data

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Create features and labels

train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Create normalization layer

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_dataset.to_numpy())

train_dataset = normalizer(train_dataset.to_numpy())
test_dataset = normalizer(test_dataset.to_numpy())

# Linear regression model

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

model.layers[1].kernel
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

history = model.fit(
    train_dataset,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

loss = model.evaluate(test_dataset, test_labels, verbose=2)