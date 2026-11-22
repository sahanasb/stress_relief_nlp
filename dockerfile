# Use an official lightweight Python image
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /serenai_app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set a fixed cache folder for Hugging Face
ENV HF_HOME=/serenai_app/hf_cache

# Download all 3 models during the Docker build
RUN python -c "from transformers import pipeline; \
    pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base'); \
    pipeline('text-classification', model='abdulmuksith/textthreat-distilbert-dreaddit'); \
    pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# Force offline mode at runtime so it loads from disk instead of downloading
ENV HF_HUB_OFFLINE=1

# Copy the rest of the application code
COPY . .

# Expose the port Flask runs on
ENV PORT=5000

# Command to run the application
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app"]