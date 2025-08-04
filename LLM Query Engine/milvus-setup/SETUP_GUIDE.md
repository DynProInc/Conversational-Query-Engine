# Milvus RAG System Setup Guide

This guide provides detailed instructions for setting up the Milvus vector database for the Conversational Query Engine RAG (Retrieval-Augmented Generation) system. It covers everything from initial Docker installation to running the system in production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Setup](#docker-setup)
3. [Milvus Container Setup](#milvus-container-setup)
4. [Verifying Milvus Installation](#verifying-milvus-installation)
5. [RAG System Configuration](#rag-system-configuration)
6. [Running the API Server with RAG](#running-the-api-server-with-rag)
7. [Production Deployment Considerations](#production-deployment-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Key Files Reference](#key-files-reference)

## Prerequisites

Before setting up the Milvus RAG system, ensure you have:

- **Operating System**: Windows, Linux, or macOS
- **Hardware Requirements**:
  - CPU: 4+ cores recommended
  - RAM: 8GB+ recommended (16GB+ for production)
  - Storage: 10GB+ free space

## Docker Setup

### 1. Install Docker

#### Windows
1. Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Ensure WSL2 is enabled (for Windows users)
3. Start Docker Desktop and verify it's running

#### Linux
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-compose

# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in for changes to take effect
```

### 2. Verify Docker Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version
```

## Milvus Container Setup

Our system uses Docker Compose to manage three containers:
- **milvus-standalone**: The main Milvus service
- **milvus-etcd**: For metadata storage
- **milvus-minio**: For vector data storage

### Method 1: Using the Setup Scripts (Recommended)

#### Windows (PowerShell)

```powershell
# Navigate to the milvus-setup directory
cd "c:\path\to\Conversational-Query-Engine\LLM Query Engine\milvus-setup"

# Run the setup script
powershell -ExecutionPolicy Bypass -File "setup_milvus_containers.ps1"
```

#### Linux/macOS

```bash
# Navigate to the milvus-setup directory
cd /path/to/Conversational-Query-Engine/LLM\ Query\ Engine/milvus-setup

# Make the script executable
chmod +x scripts/setup_milvus_containers.sh

# Run the setup script
./scripts/setup_milvus_containers.sh
```

### Method 2: Manual Setup with Docker Compose

```bash
# Navigate to the milvus-setup directory
cd /path/to/Conversational-Query-Engine/LLM\ Query\ Engine/milvus-setup

# Start the containers
docker-compose up -d

# Verify containers are running
docker-compose ps
```

## Verifying Milvus Installation

### 1. Check Container Status

```bash
# Check if all containers are running
docker ps --filter "name=milvus"
```

You should see three containers running:
- milvus-standalone
- milvus-etcd
- milvus-minio

### 2. Test Milvus Connection

Run the connection test script:

```bash
# Navigate to the milvus-setup directory
cd /path/to/Conversational-Query-Engine/LLM\ Query\ Engine/milvus-setup

# Run the connection test
python check_milvus_connection.py
```

If successful, you should see:
```
✅ Successfully connected to Milvus!
✅ Milvus server version: v2.2.11
```

## RAG System Configuration

### 1. Install Python Dependencies

```bash
# Navigate to the milvus-setup directory
cd /path/to/Conversational-Query-Engine/LLM\ Query\ Engine/milvus-setup

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Client Embeddings

The system uses SentenceTransformer embeddings stored in Milvus with the following configuration:
- **Metric Type**: IP (Inner Product)
- **Index Type**: IVF_FLAT
- **Parameters**: {"nlist": 64}

To generate embeddings for a client:

```bash
# Navigate to the LLM Query Engine directory
cd /path/to/Conversational-Query-Engine/LLM\ Query\ Engine

# Run the embedding generation script
python -m milvus-setup.generate_client_embeddings --client_id CLIENT_ID --dict_path path/to/dictionary.csv
```

## Running the API Server with RAG

To start the API server with RAG enabled:

```bash
# Navigate to the LLM Query Engine directory
cd /path/to/Conversational-Query-Engine/LLM\ Query\ Engine

# Start the API server with RAG enabled
python api_server.py --with-rag --port 8002
```

The `--with-rag` flag triggers:
1. Checking if Milvus containers are running (using `milvus_container_utils.py`)
2. Starting containers if needed
3. Initializing the RAG system
4. Including RAG API endpoints in the FastAPI server

## Production Deployment Considerations

For production deployments, consider the following:

### 1. Environment Variables

Create a `.env` file in the milvus-setup directory with appropriate settings:

```
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 2. Data Persistence

The docker-compose.yml file uses named volumes for data persistence:
- etcd_data
- minio_data
- milvus_data

These volumes persist data even when containers are stopped or removed.

### 3. Backup Strategy

Regularly backup your named Docker volumes:

```bash
# Create a backup directory
mkdir -p ~/milvus-backups

# Backup volumes
docker run --rm -v etcd_data:/data -v ~/milvus-backups:/backup alpine tar -czf /backup/etcd_data.tar.gz /data
docker run --rm -v minio_data:/data -v ~/milvus-backups:/backup alpine tar -czf /backup/minio_data.tar.gz /data
docker run --rm -v milvus_data:/data -v ~/milvus-backups:/backup alpine tar -czf /backup/milvus_data.tar.gz /data
```

### 4. Resource Allocation

For production, allocate sufficient resources to Docker:
- CPU: 4+ cores
- RAM: 16GB+
- Disk: SSD storage recommended

## Troubleshooting

### Common Issues

1. **Containers not starting**
   - Check Docker logs: `docker logs milvus-standalone`
   - Ensure ports 19530 and 9091 are not in use by other applications

2. **Connection issues**
   - Verify Milvus is running: `docker ps --filter "name=milvus"`
   - Check network configuration: `docker network inspect bridge`

3. **Embedding generation failures**
   - Check Python dependencies are installed
   - Verify SentenceTransformer model is accessible
   - Ensure client dictionary files exist and are properly formatted

### Logs

Check logs for detailed error information:

```bash
# Docker container logs
docker logs milvus-standalone
docker logs milvus-etcd
docker logs milvus-minio

# Application logs
cat milvus-setup/logs/rag_system.log
```

## Key Files Reference

| File | Purpose | Usage |
|------|---------|-------|
| `docker-compose.yml` | Defines Milvus container configuration | Used by setup scripts and manual docker-compose commands |
| `setup_milvus_containers.ps1` | PowerShell script for Windows setup | Main setup script for Windows users |
| `scripts/setup_milvus_containers.sh` | Shell script for Linux/macOS setup | Main setup script for Linux/macOS users |
| `milvus_container_utils.py` | Python utilities for container management | Used by API server to check/start containers |
| `rag_embedding.py` | Main RAG implementation | Core RAG functionality with Milvus integration |
| `schema_processor.py` | Processes client database schemas | Prepares data for embedding generation |
| `generate_client_embeddings.py` | Creates and updates embeddings | Used to generate vector embeddings for clients |
| `check_milvus_connection.py` | Tests Milvus connectivity | Utility to verify Milvus is accessible |
| `requirements.txt` | Python dependencies | Lists required packages |

### Docker Compose File Details

The `docker-compose.yml` file defines three services:

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    ports:
      - "9000:9000"
      - "9001:9001"

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.2.11
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

This configuration uses named Docker volumes for data persistence and proper container orchestration.
