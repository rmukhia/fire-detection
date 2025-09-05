# Fire Detection System Docker Image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install GDAL and other geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Create directories for data and outputs
RUN mkdir -p data output models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/output
ENV MODEL_DIR=/app/models

# Expose port for Jupyter/dashboards
EXPOSE 8888 8787

# Default command
CMD ["python", "-c", "from aad.common.performance import log_system_info; log_system_info()"]