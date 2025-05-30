FROM python:3.10-slim

# Set working directory
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
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install python-dotenv if not in requirements.txt
RUN pip install --no-cache-dir python-dotenv

# Copy application files
COPY . .

# Create directory for models
RUN mkdir -p /app/models

# Expose FastAPI port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=3600s --timeout=3600s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default environment variables
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV MAX_REQUESTS_PER_MINUTE=100
ENV ALLOWED_ORIGINS=http://localhost:3000
ENV ML_API_KEY=your_secure_api_key_here
ENV PYTHONPATH=/app


# Start FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
