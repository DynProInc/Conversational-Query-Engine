FROM python:3.13.3-slim

WORKDIR /app

# Create directory structure first
RUN mkdir -p "/app/LLM Query Engine"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ["LLM Query Engine/requirements.txt", "/app/LLM Query Engine/"]

# Install dependencies
RUN pip install --no-cache-dir -r "/app/LLM Query Engine/requirements.txt" \
    && pip install pymilvus==2.5.14

# Pre-download models to avoid runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-large-en-v1.5')" \
    && python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large'); model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')"

# Note: We don't copy the application code here because it will be mounted as a volume
# This allows us to maintain the exact same directory structure

# Create cache directory for downloaded models
RUN mkdir -p /root/.cache/huggingface

# Expose the port the app runs on
EXPOSE 8002

# Command to run the application
CMD ["python", "LLM Query Engine/api_server.py", "--port", "8002", "--with-rag"]
