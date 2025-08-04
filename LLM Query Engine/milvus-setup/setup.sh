#!/bin/bash
# Milvus Multi-Client RAG Setup Script
# ------------------------------------

# Set color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Milvus Multi-Client RAG Setup Script${NC}"
echo -e "${GREEN}=====================================${NC}"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not installed${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}[INFO] Docker is installed. Checking if Docker is running...${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not running. Please start Docker daemon.${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] Docker is running. Setting up Milvus environment...${NC}"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}[WARNING] Docker Compose not found as standalone command.${NC}"
    echo -e "${YELLOW}Trying Docker Compose plugin...${NC}"
    
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}[ERROR] Docker Compose is not installed${NC}"
        echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
        exit 1
    else
        echo -e "${GREEN}[INFO] Docker Compose plugin is available.${NC}"
        COMPOSE_CMD="docker compose"
    fi
else
    echo -e "${GREEN}[INFO] Docker Compose is installed.${NC}"
    COMPOSE_CMD="docker-compose"
fi

# Create directories if they don't exist
mkdir -p volumes/etcd volumes/minio volumes/milvus logs

# Set Docker volume directory
export DOCKER_VOLUME_DIRECTORY=$(pwd)

echo -e "${GREEN}[INFO] Starting Milvus containers with Docker Compose...${NC}"
${COMPOSE_CMD} down
${COMPOSE_CMD} up -d

echo
echo -e "${YELLOW}[INFO] Waiting for Milvus to start up...${NC}"
sleep 10

# Check if containers are running
echo -e "${GREEN}[INFO] Checking container status...${NC}"
${COMPOSE_CMD} ps

# Install Python dependencies
echo -e "${GREEN}[INFO] Installing Python dependencies...${NC}"
pip install -r requirements.txt

echo
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Milvus Multi-Client RAG Setup Complete!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo
echo -e "Milvus is now running at ${YELLOW}localhost:19530${NC}"
echo -e "Minio Console: ${YELLOW}http://localhost:9001${NC} (minioadmin/minioadmin)"
echo
echo "To initialize the RAG system:"
echo -e "${YELLOW}python multi_client_rag.py${NC}"
echo
echo "To stop Milvus:"
echo -e "${YELLOW}${COMPOSE_CMD} down${NC}"
echo -e "${GREEN}===============================================${NC}"
