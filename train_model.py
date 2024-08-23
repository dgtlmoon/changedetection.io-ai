#!/usr/bin/python3

import json
import numpy as np

from app import helpers
from app.model import build_model, compile_model
import tensorflow as tf
from app.extract_features import extract_features  # Import the extract_features function
import glob
import os

elements = []


json_files = glob.glob(os.path.join('elements/', '*.json'), recursive=True)

print(f"TF Version: {tf.__version__} training on {len(json_files)} JSON scraped elements files")

# Iterate through each JSON file found
for file_path in json_files:
    with open(file_path, 'r') as f:
        print (f"Parsing {file_path} ...")
        data = json.loads(f.read())
        # Append the 'size_pos' from each file to the elements list
        if 'size_pos' in data:
            elements.extend(data['size_pos'])

print (f"Found {len(elements)} elements initially..")

import random
from copy import deepcopy

### augment more data
# Extract price elements
price_elements = [element for element in elements if element.get('label','') == 'price']

# Number of new elements to generate
num_new_elements = len(elements) - int(len(price_elements)/2)

for i in range(num_new_elements):
    # Make a deep copy of a random price element
    random_price_element = deepcopy(random.choice(price_elements))

    # Randomly modify the copied element
    random_price_element['height'] += random.randint(-5, 20)
    random_price_element['top'] += random.randint(-5, 200)
    random_price_element['left'] += random.randint(-250, 200)

    # Ensure fontWeight is an integer and then modify it
    random_price_element['fontWeight'] = int(random_price_element['fontWeight'])
    random_price_element['fontWeight'] += random.randint(1, 200)


    # Append the modified element back to the elements list
    elements.append(random_price_element)


print (f"Found {len(elements)} elements after augmenting more data with price label..")

# find the element with label='price' and then make the same amount of entries again


# Filter elements where 'top' <= 800
filtered_elements = helpers.get_filtered_elements(elements)

# Prepare the training data
X_train = np.array([extract_features(element) for element in filtered_elements])
y_train = np.array([1 if element['label'] == 'price' else 0 for element in filtered_elements])

# Reshape X_train for the Conv1D layer
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Reshape y_train to match model output shape (None, 1)
y_train = y_train.reshape(-1, 1)

# Define the input shape for the model
input_shape = (X_train.shape[1], 1)

# Build and compile the model
model = build_model(input_shape)
model = compile_model(model, learning_rate=0.00001)

# Train the model
model.fit(X_train_reshaped, y_train, epochs=75, batch_size=32, validation_split=0.4)

# Save the trained model
model.save('../trained_model.keras')


# Convert the model to TensorFlow Lite format also for future use
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('trained_model.tflite', 'wb') as f:
    f.write(tflite_model)
