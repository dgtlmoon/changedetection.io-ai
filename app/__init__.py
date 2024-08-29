#!/usr/bin/python3

from fastapi import FastAPI, Request
import uvicorn

from app import helpers


# Create a FastAPI app
app = FastAPI()


@app.post("/price-element")
async def process_data(request: Request):
    data = await request.json()
    from app.predict import perform_prediction
    ret = perform_prediction(data)
    return ret

if __name__ == "__main__":
    # Run the server with multiple workers and threads
    uvicorn.run(app, host="0.0.0.0", port=5005, workers=8, loop="asyncio")
