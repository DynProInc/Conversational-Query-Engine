# Docker Commands Reference for Conversational Query Engine

This document provides a comprehensive list of Docker commands for daily development and troubleshooting of the Conversational Query Engine project.

## Basic Docker Operations

```bash
# Start all containers in detached mode
docker-compose up -d

# Stop all containers
docker-compose down

# Restart all containers
docker-compose restart

# Restart a specific container
docker restart conversational-query-engine-app-1
```

## Updating Code Without Rebuilding

```bash
# Copy updated file to container (much faster than rebuilding)
docker cp "c:\Users\git_manoj\Conversational-Query-Engine\LLM Query Engine\embedding_api.py" conversational-query-engine-app-1:"/app/LLM Query Engine/embedding_api.py"

# Copy multiple files using PowerShell
Get-ChildItem -Path "c:\Users\git_manoj\Conversational-Query-Engine\LLM Query Engine\*.py" | ForEach-Object {
    docker cp $_.FullName conversational-query-engine-app-1:"/app/LLM Query Engine/$($_.Name)"
}
```

## Monitoring and Debugging

```bash
# View logs of a specific container
docker-compose logs app

# Follow logs in real-time
docker-compose logs -f app

# View logs with timestamps
docker-compose logs --timestamps app

# View last 100 lines of logs
docker-compose logs --tail=100 app

# Check container status
docker-compose ps

# Get detailed container info
docker inspect conversational-query-engine-app-1
```

## Accessing Container Shell

```bash
# Get an interactive shell in the app container
docker exec -it conversational-query-engine-app-1 bash

# Run a specific command in the container
docker exec conversational-query-engine-app-1 python -c "import sys; print(sys.path)"
```

## Managing Images and Builds

```bash
# Rebuild a specific service
docker-compose build app

# Force rebuild without cache
docker-compose build --no-cache app

# List all images
docker images

# Remove unused images
docker image prune
```

## Troubleshooting

```bash
# Check container networking
docker network ls
docker network inspect conversational-query-engine_default

# Check container resource usage
docker stats

# View container environment variables
docker exec conversational-query-engine-app-1 env

# Check volume mounts
docker inspect -f '{{ .Mounts }}' conversational-query-engine-app-1
```

## Data Management

```bash
# Create a backup of Milvus data
docker run --rm --volumes-from conversational-query-engine-milvus-standalone-1 -v $(pwd):/backup alpine tar -czvf /backup/milvus-data-backup.tar.gz /var/lib/milvus

# Copy files from container to host
docker cp conversational-query-engine-app-1:"/app/LLM Query Engine/logs/app.log" ./app-logs.log
```

## Milvus-Specific Commands

```bash
# Check Milvus status
docker-compose ps milvus-standalone

# View Milvus logs
docker-compose logs milvus-standalone

# Restart just Milvus
docker-compose restart milvus-standalone

# Check Milvus configuration
docker exec conversational-query-engine-app-1 env | grep MILVUS
```

## Quick Reference for Common Workflows

### Code Update Workflow

```bash
# 1. Edit files locally
# 2. Copy to container
docker cp "c:\Users\git_manoj\Conversational-Query-Engine\LLM Query Engine\updated_file.py" conversational-query-engine-app-1:"/app/LLM Query Engine/updated_file.py"
# 3. Restart container
docker restart conversational-query-engine-app-1
# 4. Check logs
docker-compose logs -f app
```

### Troubleshooting Connection Issues

```bash
# 1. Check if containers are running
docker-compose ps
# 2. Check environment variables
docker exec conversational-query-engine-app-1 env | grep MILVUS
# 3. Check logs for errors
docker-compose logs --tail=100 app
# 4. Restart the service if needed
docker-compose restart app
```

### API Testing Workflow

```bash
# 1. Start all containers
docker-compose up -d
# 2. Check health endpoint
curl http://localhost:8002/health
# 3. Test specific endpoints
curl http://localhost:8002/embeddings/stats
# 4. View logs for any errors
docker-compose logs -f app
```

## Notes

- The Docker setup uses port 8002 for the application
- The `.env` file is mounted into the container at runtime
- Milvus connection uses environment variables `MILVUS_HOST=milvus-standalone` and `MILVUS_PORT=19530`
- Using `docker cp` to update files is much faster than rebuilding the entire image (30-40 minutes)
- Two Milvus instances can run simultaneously: local on port 19530:19530 and Docker on port 19531:19530
