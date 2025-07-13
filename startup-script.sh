#!/bin/bash

# Update packages and install required software
apt-get update
apt-get install -y docker.io git

# Start Docker service
systemctl start docker
systemctl enable docker

# Set up application directory
mkdir -p /app

# Clone the repository from GitHub
git clone https://github.com/YOUR_USERNAME/Conversational-Query-Engine.git /app
cd /app

# Create .env file if it doesn't exist (will be populated later)
if [ ! -f .env ]; then
    touch .env
fi

# Build the Docker image
docker build -t cqe-api .

# Stop any running container
docker stop cqe-container || true
docker rm cqe-container || true

# Run the container
docker run -d --name cqe-container -p 80:8000 \
  --restart always \
  --env-file .env \
  cqe-api

# Set up logging
mkdir -p /var/log/cqe-api
docker logs cqe-container > /var/log/cqe-api/startup.log 2>&1
