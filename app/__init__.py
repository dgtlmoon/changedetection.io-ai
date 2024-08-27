#!/usr/bin/python3

from fastapi import FastAPI, Request
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import inotify.adapters
from loguru import logger

from app import helpers
from app.extract_features import extract_features

logger.info(f"TF Version: {tf.__version__} (Tested with 2.17.0)")

# Create a threading lock for model reloading
model_lock = threading.Lock()

# Load the trained model initially
model = load_model('trained_model.keras')

# Create a FastAPI app
app = FastAPI()

def monitor_model_file():
    """Monitor the model file for changes and reload the model."""
    i = inotify.adapters.Inotify()
    i.add_watch('trained_model.keras')

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        if 'IN_MODIFY' in type_names:
            logger.info(f"Model file '{filename}' modified, reloading model...")
            with model_lock:
                global model
                model = load_model('trained_model.keras')
            logger.info("Model reloaded successfully.")

# Start the file monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_model_file, daemon=True)
monitor_thread.start()

@app.post("/price-element")
async def process_data(request: Request):
    data = await request.json()

    # Extract the 'size_pos' list, filter small, big, out of size elements and add `original_idx` for future use
    elements = helpers.get_filtered_elements(data['size_pos'])

    # Prepare the data for prediction
    X_new = np.array([extract_features(element) for element in elements])
    X_new_reshaped = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))

    # Wait if the model is being reloaded
    with model_lock:
        # Make predictions
        predictions = model.predict(X_new_reshaped)

    # Combine predictions with corresponding elements and scores
    # BINARY CLASSIFICATION
    results = []
    for i, (binary_prediction, score) in enumerate(zip((predictions > 0.5).astype(int), predictions)):
        binary_prediction_value = binary_prediction[0]  # Access the first element
        score_value = score[0]  # Access the first element
        results.append((binary_prediction_value, score_value, elements[i]['original_idx']))

    # Sort results by score_value in descending order
    results_sorted = sorted(results, key=lambda x: x[1], reverse=False)

    return {
        'bin': int(results_sorted[-1][0]),
        'score': float(results_sorted[-1][1]),
        'idx': int(results_sorted[-1][2])
    }

if __name__ == "__main__":
    # Run the server with multiple workers and threads
    uvicorn.run(app, host="0.0.0.0", port=5005, workers=8, loop="asyncio")
