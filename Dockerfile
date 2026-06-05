# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for compiling python packages (e.g. gcc, pg_config for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup files first for layer caching
COPY requirements.txt setup.py /app/

# Install python dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the package source code
COPY shopsense /app/shopsense
COPY README.md /app/

# Install shopsense package in editable mode
RUN pip install -e .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run uvicorn to serve the API
CMD ["uvicorn", "shopsense.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
