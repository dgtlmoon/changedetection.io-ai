# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Make port 5005 available to the world outside this container
EXPOSE 5005

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "2"]
