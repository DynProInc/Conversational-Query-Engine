#!/bin/bash

# Script to update the application from GitHub and restart the container
# Run this script when you want to pull the latest changes

# Navigate to app directory
cd /app

# Pull latest changes from GitHub
git pull

# Rebuild Docker image
docker build -t cqe-api .

# Stop and remove existing container
docker stop cqe-container
docker rm cqe-container

# Run new container
docker run -d --name cqe-container -p 80:8000 \
  --restart always \
  --env-file .env \
  cqe-api

# Display logs
echo "Application updated and restarted"
docker logs cqe-container --tail 20
