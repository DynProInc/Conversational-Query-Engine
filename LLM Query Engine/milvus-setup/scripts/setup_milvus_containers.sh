#!/bin/bash
# setup_milvus_containers.sh
# Script to set up Milvus containers for RAG system
# This script can be run from the scripts directory

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Milvus containers for RAG system...${NC}"

# Get the parent directory (milvus-setup directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MILVUS_SETUP_DIR="$(dirname "$SCRIPT_DIR")"

# Change to milvus-setup directory
cd "$MILVUS_SETUP_DIR"
echo -e "${CYAN}Running from directory: $MILVUS_SETUP_DIR${NC}"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}docker-compose is not installed. Please install docker-compose first.${NC}"
    exit 1
fi

# Check if docker-compose.yml exists in milvus-setup directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}docker-compose.yml not found in milvus-setup directory.${NC}"
    echo -e "${YELLOW}Please ensure this script is in the milvus-setup/scripts directory.${NC}"
    exit 1
fi

# Check if containers exist and their status
echo -e "${CYAN}Checking existing Milvus containers...${NC}"
CONTAINERS_EXIST=false
CONTAINERS_RUNNING=true

for CONTAINER in "milvus-standalone" "milvus-etcd" "milvus-minio"; do
    if docker ps -a --filter "name=$CONTAINER" --format "{{.Status}}" | grep -q .; then
        CONTAINERS_EXIST=true
        CONTAINER_STATUS=$(docker ps -a --filter "name=$CONTAINER" --format "{{.Status}}")
        echo -e "${CYAN}$CONTAINER exists. Status: $CONTAINER_STATUS${NC}"
        
        if ! echo "$CONTAINER_STATUS" | grep -q "Up"; then
            CONTAINERS_RUNNING=false
        fi
    else
        echo -e "${YELLOW}$CONTAINER does not exist.${NC}"
        CONTAINERS_EXIST=false
        CONTAINERS_RUNNING=false
    fi
done

# Handle different scenarios
if $CONTAINERS_EXIST && $CONTAINERS_RUNNING; then
    echo -e "${GREEN}All Milvus containers are already running.${NC}"
elif $CONTAINERS_EXIST && ! $CONTAINERS_RUNNING; then
    # Containers exist but not running - start them
    echo -e "${CYAN}Starting existing Milvus containers...${NC}"
    for CONTAINER in "milvus-etcd" "milvus-minio" "milvus-standalone"; do
        docker start $CONTAINER
        echo -e "${GREEN}Started $CONTAINER${NC}"
    done
else
    # Containers don't exist or some are missing - recreate all
    echo -e "${CYAN}Creating Milvus containers using docker-compose...${NC}"
    
    # Remove any existing containers first
    docker-compose down -v
    
    # Create and start containers
    docker-compose up -d
    
    echo -e "${CYAN}Waiting for containers to initialize...${NC}"
    sleep 10
fi

# Verify containers are running
ALL_RUNNING=true
for CONTAINER in "milvus-standalone" "milvus-etcd" "milvus-minio"; do
    CONTAINER_STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    
    if ! echo "$CONTAINER_STATUS" | grep -q "Up"; then
        echo -e "${RED}$CONTAINER is not running.${NC}"
        ALL_RUNNING=false
    else
        echo -e "${GREEN}$CONTAINER is running: $CONTAINER_STATUS${NC}"
    fi
done

if $ALL_RUNNING; then
    echo -e "${GREEN}All Milvus containers are now running successfully!${NC}"
    
    # Display container information
    echo -e "\n${CYAN}Container Details:${NC}"
    docker ps --filter "name=milvus"
else
    echo -e "${RED}Some containers failed to start. Please check Docker logs:${NC}"
    echo -e "${YELLOW}docker logs milvus-standalone${NC}"
fi

echo -e "\n${GREEN}Milvus container setup completed!${NC}"
echo -e "${YELLOW}You can now run the API server with: python api_server.py --with-rag --port 8002${NC}"

# Script completed
echo -e "\n${CYAN}Press Enter to exit...${NC}"
read
