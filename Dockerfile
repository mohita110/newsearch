FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Extract dataset (expects twenty_newsgroups.zip in /app)
RUN if [ -f twenty_newsgroups.zip ]; then \
        unzip twenty_newsgroups.zip -d newsgroups_raw_zip && \
        cd newsgroups_raw_zip && \
        tar -xzf 20_newsgroups.tar.gz && \
        mv 20_newsgroups ../newsgroups_raw/20_newsgroups && \
        cd .. && rm -rf newsgroups_raw_zip; \
    fi

# Pre-build the embeddings and models (baked into the image)
# Skip this if you want to mount pre-built embeddings at runtime
RUN python run_pipeline.py

# Expose API port
EXPOSE 8000

# Start the FastAPI service
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
