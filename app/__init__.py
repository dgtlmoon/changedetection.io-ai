#!/usr/bin/python3

from fastapi import FastAPI, Request
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from app import helpers
from app.extract_features import extract_features

print(f"TF Version: {tf.__version__} (Tested with 2.17.0)")

# Load the trained model
model = load_model('trained_model.keras')
app = FastAPI()

@app.post("/price-element")
async def process_data(request: Request):

    data = await request.json()

    # Extract the 'size_pos' list, filter small, big, out of size elements and add `original_idx` for future use
    elements = helpers.get_filtered_elements(data['size_pos'])

    # Prepare the data for prediction
    X_new = np.array([extract_features(element) for element in elements])
    X_new_reshaped = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))

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

    return {'bin': int(results_sorted[-1][0]),
            'score': float(results_sorted[-1][1]),
            'idx': int(results_sorted[-1][2])
            }

if __name__ == "__main__":
    # Run the server with multiple workers and threads
    uvicorn.run(app, host="0.0.0.0", port=5005, workers=8, loop="asyncio")
