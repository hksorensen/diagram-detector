FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install package
COPY . /app
RUN pip install --no-cache-dir /app

# Create directories for data
RUN mkdir -p /data/input /data/output

WORKDIR /data

ENTRYPOINT ["diagram-detect"]
CMD ["--help"]
