FROM python:3.13.3-slim

WORKDIR /app

# Create directory structure first
RUN mkdir -p /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ["LLM Query Engine/requirements.txt", "/app/"]

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install pymilvus==2.5.14

# Copy application code
COPY ["LLM Query Engine/", "/app/"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "api_server.py", "--port", "8000"]
