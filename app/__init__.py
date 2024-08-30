#!/usr/bin/python3

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from loguru import logger

from app import helpers
from app.predict import perform_prediction

# Create a FastAPI app
app = FastAPI()


@app.post("/price-element")
async def process_data(request: Request):
    try:
        data = await request.json()
    except ValueError:
        s = "Invalid JSON format"
        logger.warning(s)
        raise HTTPException(status_code=400, detail=s)

    size_pos = data.get('size_pos')
    if not isinstance(size_pos, list):
        s = "Expected 'size_pos' to be a list of scraped elements from changedetection.io"
        logger.warning(s)
        raise HTTPException(status_code=400, detail=s)

    elements = helpers.get_filtered_elements(size_pos)

    if not elements:
        s = "No valid elements found after filtering - see helpers.get_filtered_elements (Maybe none had text set/numerical data etc)"
        logger.warning(s)
        return {}

    ret = perform_prediction(elements=elements)
    return ret


if __name__ == "__main__":
    # Run the server with multiple workers and threads
    uvicorn.run(app, host="0.0.0.0", port=5005, workers=8, loop="asyncio")
