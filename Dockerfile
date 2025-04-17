FROM python:3.10-slim

# Create a non-root user
# RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Add python-dotenv to requirements if not already there
RUN pip install --no-cache-dir python-dotenv

# Create directory for models if needed
RUN mkdir -p /app/models

# Change ownership to non-root user
# RUN chown -R appuser:appuser /app

# Switch to non-root user
# USER appuser

# Expose the port that FastAPI will run on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables (can be overridden)
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV MAX_REQUESTS_PER_MINUTE=100
ENV ALLOWED_ORIGINS=http://localhost:3000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]