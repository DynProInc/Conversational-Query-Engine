# Deployment Guide for Conversational Query Engine

This guide provides instructions for deploying the Conversational Query Engine application using Docker with full RAG (Retrieval-Augmented Generation) functionality.

## Prerequisites

- Docker and Docker Compose installed on your deployment machine
- Environment variables configured in `.env` file
- Snowflake credentials and API keys for OpenAI, Claude, and/or Gemini
- At least 4GB of free RAM for Milvus vector database

## Deployment Options

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd Conversational-Query-Engine
   ```

2. **Configure environment variables**:
   Create a `.env` file in the `LLM Query Engine` directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   SNOWFLAKE_ACCOUNT=your_snowflake_account
   SNOWFLAKE_USER=your_snowflake_user
   SNOWFLAKE_PASSWORD=your_snowflake_password
   SNOWFLAKE_DATABASE=your_snowflake_database
   SNOWFLAKE_SCHEMA=your_snowflake_schema
   SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
   SNOWFLAKE_ROLE=your_snowflake_role
   ```
   
   > **IMPORTANT SECURITY NOTE**: The `.env` file contains sensitive information and is **not** included in the Docker image. When deploying to a new machine, you must manually create this file in the `LLM Query Engine` directory before running the application. Never commit this file to version control.

3. **Build and start the application**:
   ```bash
   docker-compose up -d
   ```

   This will:
   - Build the application container
   - Start the Milvus vector database services (etcd, minio, milvus-standalone)
   - Start the application with RAG functionality enabled
   - Configure the application to connect to Milvus using the service name `milvus-standalone`

4. **Access the application**:
   The API will be available at `http://localhost:8002`
   - API documentation: `http://localhost:8002/docs`
   - Redoc documentation: `http://localhost:8002/redoc`
   - Health check: `http://localhost:8002/health`
   - Client status: `http://localhost:8002/health/client`
   - RAG statistics: `http://localhost:8002/rag/stats`

5. **Monitor the logs**:
   ```bash
   docker-compose logs -f app
   ```

6. **Testing RAG functionality**:
   ```bash
   # Check RAG status and collections
   curl -X GET "http://localhost:8002/rag/stats"
   
   # Query the RAG system
   curl -X POST "http://localhost:8002/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"client_id": "your_client_id", "query": "your search query", "top_k": 5}'
   ```

### Option 2: Manual Deployment

1. **Install Python 3.13.3**:
   Download and install from [python.org](https://www.python.org/downloads/)

2. **Install dependencies**:
   ```bash
   cd "LLM Query Engine"
   pip install -r requirements.txt
   ```

3. **Start Milvus services** (if using RAG functionality):
   ```bash
   cd "LLM Query Engine/milvus-setup"
   docker-compose up -d
   ```

4. **Run the application**:
   ```bash
   cd "LLM Query Engine"
   python api_server.py --port 8002 --with-rag
   ```

## Troubleshooting

### Milvus Connection Issues

- **Problem**: Application cannot connect to Milvus
- **Solution**: Verify that the Milvus container is running and that the application is using the correct host and port
  ```bash
  # Check if Milvus is running
  docker ps | grep milvus
  
  # For Docker deployment, ensure MILVUS_HOST=milvus-standalone in docker-compose.yml
  # For local deployment, ensure MILVUS_HOST=localhost
  ```

### Port Conflicts

- **Problem**: Port 8002 or Milvus ports (19530, 9091) are already in use
- **Solution**: Change the port mapping in docker-compose.yml
  ```yaml
  ports:
    - "8003:8002"  # Map container port 8002 to host port 8003
  ```

### Docker Image Size

- **Problem**: Docker image is large (>10GB)
- **Solution**: Consider using the manual deployment option if disk space is limited

## Advanced Configuration

### Running Multiple Instances

You can run both a local instance and a Docker instance simultaneously by using different port mappings:

```yaml
# In docker-compose.yml
ports:
  - "8003:8002"  # Docker instance on port 8003
  - "19531:19530"  # Milvus on different port
```

Then run the local instance on the default port:
```bash
python api_server.py --port 8002 --with-rag
```

### Missing Dependencies

If you encounter errors about missing Python modules, install them manually:
```bash
pip install pymilvus==2.5.14
pip install sentence-transformers
```

Add these dependencies to the `requirements.txt` file for future deployments.

### Port Conflicts

If you encounter port conflicts, modify the port mappings in `docker-compose.yml`:

- Changed app port from 8000 to 8002
- Changed Milvus port from 19530 to 19531
- Changed Milvus web service port from 9091 to 9092

### Volume Mount Errors

If you see errors related to volume mounts, ensure the directories exist and have proper permissions.

## Environment Variables

Ensure your `.env` file includes:

```
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Claude Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key

# Gemini Configuration
GOOGLE_API_KEY=your_google_api_key

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_snowflake_account
SNOWFLAKE_USER=your_snowflake_user
SNOWFLAKE_PASSWORD=your_snowflake_password
SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
SNOWFLAKE_DATABASE=your_snowflake_database
SNOWFLAKE_SCHEMA=your_snowflake_schema

# Client-specific configurations
CLIENT_MTS_OPENAI_MODEL=gpt-4o
CLIENT_MTS_CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLIENT_MTS_GEMINI_MODEL=gemini-1.5-pro
CLIENT_PENGUIN_OPENAI_MODEL=gpt-4-turbo
```

## Health Check

Use the following endpoints to verify the application is running correctly:

```bash
# Basic health check
curl http://localhost:8002/health

# Client status check
curl http://localhost:8002/health/client
```

## Docker Commands Reference

### Building the Docker Image

```bash
# Build the Docker image
docker-compose build

# Build without using cache (for clean rebuild)
docker-compose build --no-cache

# Build a specific service
docker-compose build app
```

### Starting Containers

```bash
# Start all containers in detached mode
docker-compose up -d

# Start a specific container
docker-compose up -d app

# Start with rebuilding
docker-compose up -d --build
```

### Stopping Containers

```bash
# Stop all containers
docker-compose stop

# Stop a specific container
docker-compose stop app

# Stop and remove containers
docker-compose down

# Stop and remove containers, networks, images, and volumes
docker-compose down --rmi all --volumes
```

### Managing Containers

```bash
# View running containers
docker-compose ps

# View all containers (including stopped)
docker-compose ps -a

# View container logs
docker-compose logs -f app

# Execute command in a running container
docker-compose exec app bash

# Copy files to/from container
docker cp "LLM Query Engine/rag_api.py" conversational-query-engine-app-1:/app/rag_api.py
docker cp conversational-query-engine-app-1:/app/logs/app.log ./app_logs.log

# Restart containers
docker-compose restart app
```

### Monitoring Resources

```bash
# View container resource usage
docker stats

# View container details
docker inspect conversational-query-engine-app-1
```

## Updating the Application

### Quick Updates Without Rebuilding

For small code changes, you can update the running Docker container without rebuilding the entire image (which can take 30-40 minutes):

```bash
# Copy updated files to the container
docker cp "LLM Query Engine/rag_api.py" conversational-query-engine-app-1:/app/rag_api.py

# Restart just the app container
docker-compose restart app
```

### Monitoring Changes

Check the logs to verify your changes were applied:

```bash
docker-compose logs -f app
```

## Monitoring and Debugging

### Live Log Streaming

To view real-time logs from the container (equivalent to seeing terminal output in local development):

```bash
# Stream logs in real-time (follow mode)
docker-compose logs -f app

# Stream logs with timestamps
docker-compose logs -f --timestamps app

# Show only the last 100 lines and then follow
docker-compose logs -f --tail=100 app
```

### Accessing Container Shell

To get an interactive shell inside the container for debugging:

```bash
# Get a bash shell
docker-compose exec app bash

# Or if bash is not available
docker-compose exec app sh
```

Once inside the container, you can:
- Check files: `ls -la /app`
- View logs: `cat /app/logs/app.log`
- Check environment variables: `env | grep MILVUS`
- Run Python interactively: `python`

### Debugging API Endpoints

From inside the container:

```bash
# Test internal connectivity
curl http://localhost:8002/health
curl http://milvus-standalone:19530/health
```

From your host machine:

```bash
# Test external connectivity
curl http://localhost:8002/health
```

### Checking Application Status

```bash
# Check if the application process is running
docker-compose exec app ps aux | grep python

# Check port bindings
docker-compose exec app netstat -tulpn | grep LISTEN
```

### Full Rebuild

For major changes or dependency updates, rebuild the entire image:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```
After deployment, verify the application is running correctly:

```bash
curl http://localhost:8000/health
```

You should receive a JSON response with health status information for all components.

## Client-Specific Health Check

To check health for a specific client:

```bash
curl http://localhost:8000/health/client/mts
```

## Troubleshooting

1. **Milvus Connection Issues**:
   - Check if Milvus containers are running: `docker ps`
   - Restart Milvus containers: `docker-compose restart etcd minio milvus-standalone`

2. **API Key Issues**:
   - Verify API keys in the `.env` file
   - Check client-specific health endpoint for detailed diagnostics

3. **Snowflake Connection Issues**:
   - Verify Snowflake credentials in the `.env` file
   - Ensure the Snowflake warehouse is active

4. **Container Resource Issues**:
   - Increase Docker resource allocation (memory/CPU) in Docker Desktop settings
