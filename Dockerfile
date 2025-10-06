FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies commonly needed by Pillow / OpenCV and for building wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
       libjpeg62-turbo-dev \
       zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY . /app

# Create expected runtime directories
RUN mkdir -p /app/process-images /app/fonts

# Default command runs the single-file pipeline. You can override CMD at runtime.
CMD ["python", "image_translator.py"]
