FROM python:3.12.7

ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_NO_CACHE_DIR=true

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy necessary files
COPY params.yaml params.yaml
COPY /digits_recognition /digits_recognition

# Run the app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "digits_recognition.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
