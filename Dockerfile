FROM python:3.10-slim

# Set proper working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 --timeout 120 --retries 10 && \
    pip install --no-cache-dir python-dotenv && \
    pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p /app/models

# Copy application files
COPY . .

# Set environment variables
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    MAX_REQUESTS_PER_MINUTE=100 \
    ALLOWED_ORIGINS=http://localhost:3000 \
    PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Reasonable health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]