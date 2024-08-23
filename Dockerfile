# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Create a user called cdio and set up the environment
RUN useradd -ms /bin/bash cdio

# Set the working directory in the container and change ownership to the new user
WORKDIR /src

COPY app /src/app
COPY docs /src/app/docs
COPY trained_model.keras /src/trained_model.keras
COPY trained_model.tflite /src/trained_model.tflite
COPY requirements.txt /src/requirements.txt

RUN chown -R cdio:cdio /src


# Switch to the new user
USER cdio

# Create a virtual environment within the working directory
RUN python -m venv venv

# Install dependencies using the virtual environment's pip
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Make port 5005 available to the world outside this container
EXPOSE 5005


CMD ["venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
