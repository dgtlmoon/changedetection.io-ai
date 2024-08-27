#!/usr/bin/python3


from app import helpers
from app.extract_features import extract_features  # Import the extract_features function
from app.model import build_model, compile_model
from loguru import logger
import argparse
import glob
import json
import numpy as np
import os
import tempfile


def retrain(json_elements_files: []):
    import tensorflow as tf

    elements = []
    logger.info(f"TF Version: {tf.__version__}")
    logger.info(f"Training on {len(json_elements_files)} JSON scraped elements files")

    # Iterate through each JSON file found
    for file_path in json_elements_files:
        with open(file_path, 'r') as f:
            logger.info(f"Parsing {file_path} ...")
            data = json.loads(f.read())
            # Append the 'size_pos' from each file to the elements list
            if 'size_pos' in data:
                elements.extend(data['size_pos'])

    logger.info(f"Found {len(elements)} elements initially..")

    import random
    from copy import deepcopy

    ### augment more data
    # Extract price elements
    price_elements = [element for element in elements if element.get('label', '') == 'price']

    # Number of new elements to generate
    # num_new_elements = len(elements) - int(len(price_elements) / 10)
    num_new_elements = 500

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

    logger.info(f"Found {len(elements)} elements after augmenting more data with 'price' label, added {num_new_elements} new elements")

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
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.4)

    # Save the trained model
    model.save('../trained_model.keras')

    # Convert the model to TensorFlow Lite format also for future use
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('trained_model.tflite', 'wb') as f:
        f.write(tflite_model)


def validate_directory(path):
    """Validate that the given path is a directory."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"The path {path} is not a valid directory.")

    if not os.path.isfile(os.path.join(path, 'url-watches.json')):
        raise argparse.ArgumentTypeError(f"The path {path} does not contain a url-watches.json data file")

    return path


def find_watch_element_files_with_visualselector_set(path_to_datapath):
    element_files = []
    labels_set = 0

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Relabelling JSON elements file where the include_filter is set to 'price' at {temp_dir}...")
        with open(os.path.join(path_to_datapath, 'url-watches.json'), 'r') as f:
            data = json.load(f)
            for uuid, watch in data['watching'].items():

                if watch.get('last_error'):
                    continue

                if watch.get('include_filters'):
                    elements_json_file = os.path.join(path_to_datapath, uuid, 'elements.json')

                    # rewrite (!) the elements.json file and set the label="price" on any xpath that matches this include_filters
                    with open(elements_json_file, 'r') as f:
                        data = json.load(f)
                        for idx, elem in enumerate(data.get('size_pos', [])):
                            if not elem.get('textWidth'):
                                continue
                            if any(elem.get('xpath', '') in item for item in watch.get('include_filters')):
                                elem['label'] = "price"
                                labels_set += 1

                        with open(os.path.join(temp_dir, f"{uuid}-elements.json"), 'w') as file:
                            json.dump(data, file, indent=4)

        logger.info(f"Done, found {labels_set} matching selectors from the custom filters, now starting training..")
        json_files = glob.glob(os.path.join(temp_dir, '*.json'), recursive=False)

        retrain(json_elements_files=json_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some training paths.')

    parser.add_argument('-d', '--datadir', type=validate_directory, default='',
                        help='Path to data-dir which contains url-watches.json, must be path only')

    # Parse the arguments
    args = parser.parse_args()
    json_files = []

    if args.datadir:
        json_files = find_watch_element_files_with_visualselector_set(path_to_datapath=args.datadir)
    else:
        json_files = glob.glob(os.path.join('elements/', '*.json'), recursive=False)
        retrain(json_elements_files=json_files)

#
