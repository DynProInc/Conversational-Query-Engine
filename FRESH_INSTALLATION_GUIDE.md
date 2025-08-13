# Fresh Installation Guide for Conversational Query Engine

This guide provides detailed step-by-step instructions for setting up the Conversational Query Engine on a fresh machine (Windows, Linux, or macOS) using Docker containers.

## Prerequisites

- Git installed
- Docker and Docker Compose installed
  - [Docker Desktop for Windows/Mac](https://www.docker.com/products/docker-desktop/)
  - [Docker Engine for Linux](https://docs.docker.com/engine/install/)
- At least 4GB of free RAM for Milvus vector database
- At least 15GB of free disk space

## Step 1: Clone the Repository

```bash
# Windows
git clone https://github.com/your-repo/Conversational-Query-Engine.git
cd Conversational-Query-Engine

# Linux/macOS
git clone https://github.com/your-repo/Conversational-Query-Engine.git
cd Conversational-Query-Engine
```

## Step 2: Create Environment Variables File

Create a `.env` file in the `LLM Query Engine` directory:

```bash
# Windows
cd "LLM Query Engine"
notepad .env

# Linux/macOS
cd "LLM Query Engine"
nano .env
```

Add the following content to the `.env` file:

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
SNOWFLAKE_DATABASE=your_snowflake_database
SNOWFLAKE_SCHEMA=your_snowflake_schema
SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
SNOWFLAKE_ROLE=your_snowflake_role

# Client-specific configurations (optional)
CLIENT_MTS_OPENAI_MODEL=gpt-4o
CLIENT_MTS_CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLIENT_PENGUIN_OPENAI_MODEL=gpt-4-turbo
```

Save and close the file. This `.env` file will be mounted into the Docker container at runtime.

## Step 3: Build and Start Docker Containers

Navigate back to the root directory and start the Docker containers:

```bash
# Windows
cd ..
docker-compose up -d

# Linux/macOS
cd ..
docker-compose up -d
```

This command will:
1. Build the application Docker image
2. Start the Milvus vector database services (etcd, minio, milvus-standalone)
3. Start the application with RAG functionality enabled
4. Configure the application to connect to Milvus using the service name `milvus-standalone`

> **Note**: The first build may take 30-40 minutes depending on your internet connection and machine performance. The system will download large language model files for the reranker during the first startup.

## Step 4: Verify the Installation

Check if all containers are running:

```bash
docker-compose ps
```

You should see four containers running:
- `conversational-query-engine-app-1`
- `conversational-query-engine-milvus-standalone-1`
- `conversational-query-engine-etcd-1`
- `conversational-query-engine-minio-1`

## Step 5: Test the API

Test the API endpoints to ensure everything is working correctly:

```bash
# Basic health check
curl http://localhost:8002/health

# Client status check
curl http://localhost:8002/health/client

# Client dictionary check
curl http://localhost:8002/client/dictionary/mts

# RAG status check
curl http://localhost:8002/rag/stats

# Embedding stats check
curl http://localhost:8002/embeddings/stats
```

You can also access the API documentation at:
- http://localhost:8002/docs

## Step 6: Monitor the Application

View the application logs to ensure everything is running correctly:

```bash
docker-compose logs -f app
```

## Step 7: Using the API Endpoints

The Conversational Query Engine provides several API endpoints for different functionalities:

### RAG (Retrieval-Augmented Generation) Endpoints

```bash
# Build RAG collection for a client
curl -X POST http://localhost:8002/rag/build -H "Content-Type: application/json" -d '{"client_id": "mts"}'

# Query RAG with standard retrieval
curl -X POST http://localhost:8002/rag/query -H "Content-Type: application/json" -d '{"query": "What are the top selling stores?", "client_id": "mts", "top_k": 5}'

# Query RAG with enhanced retrieval (reranking)
curl -X POST http://localhost:8002/rag/enhanced -H "Content-Type: application/json" -d '{"query": "What are the top selling stores?", "client_id": "mts", "top_k": 8, "rerank_top_k": 5}'

# Get RAG statistics
curl http://localhost:8002/rag/stats
```

### Embedding Endpoints

```bash
# Build embeddings for a client
curl http://localhost:8002/embeddings/build?client_id=mts&model_type=gemini

# Query embeddings
curl http://localhost:8002/embeddings/query?client_id=mts&query=sales%20data&model_type=gemini&top_k=5

# Get embedding statistics
curl http://localhost:8002/embeddings/stats

# Drop embeddings for a client
curl http://localhost:8002/embeddings/drop?client_id=mts&model_type=all
```

### Client Dictionary Endpoints

```bash
# Get client dictionary
curl http://localhost:8002/client/dictionary/mts
```

## Troubleshooting

### Container Startup Issues

If containers fail to start:

```bash
# Check container logs
docker-compose logs app
docker-compose logs milvus-standalone
docker-compose logs --follow app

# Restart containers
docker-compose restart
```

### Port Conflicts

If you encounter port conflicts:

1. Edit `docker-compose.yml`
2. Change the port mappings:
   ```yaml
   ports:
     - "8003:8002"  # Change 8002 to another port like 8003
     - "19531:19530"  # Change 19530 to another port like 19531
   ```
3. Restart the containers:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Milvus Connection Issues

If the application cannot connect to Milvus:

1. Check if Milvus is running:
   ```bash
   docker-compose ps milvus-standalone
   ```

2. Check Milvus logs:
   ```bash
   docker-compose logs milvus-standalone
   ```

3. Verify environment variables in `docker-compose.yml`:
   ```yaml
   environment:
     - MILVUS_HOST=milvus-standalone
     - MILVUS_PORT=19530
   ```

### Updating Code Without Rebuilding Docker Image

For small code changes, you can copy files directly into the running container:

```bash
# Copy updated file to container
docker cp "path/to/updated/file.py" conversational-query-engine-app-1:"/app/LLM Query Engine/file.py"

# Restart the app container
docker restart conversational-query-engine-app-1
```

This is much faster than rebuilding the entire Docker image (which can take 30-40 minutes).

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt for running commands
- Ensure Docker Desktop is running with WSL 2 backend for better performance
- Use double quotes for paths with spaces: `cd "LLM Query Engine"`
- If using multiple Milvus instances, ensure they use different ports (e.g., 19530:19530 for local and 19531:19530 for Docker)

### Linux

- You may need to use `sudo` for Docker commands if your user is not in the docker group
- Add your user to the docker group to avoid using sudo:
  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

### macOS

- Ensure Docker Desktop has enough resources allocated (at least 4GB RAM)
- Use Terminal.app or iTerm for running commands

## Docker Environment Details

### Container Structure

The Docker setup consists of four containers:
1. **app** - The main application container running the FastAPI server
2. **milvus-standalone** - Milvus vector database for storing embeddings
3. **etcd** - Key-value store used by Milvus
4. **minio** - Object storage used by Milvus

### Environment Variables

The Docker container uses these environment variables for Milvus connection:
- `MILVUS_HOST=milvus-standalone` - Uses the service name for internal Docker network communication
- `MILVUS_PORT=19530` - Default Milvus port

### Volume Mounts

The Docker setup includes these important volume mounts:
- `.env` file is mounted into the container for API keys and configuration
- Client data dictionaries are mounted from the host machine
- Milvus data is persisted in Docker volumes

## Alternative: Manual Installation (Without Docker)

If you prefer not to use Docker for the application (but still use Docker for Milvus):

1. Install Python 3.13.3
2. Install dependencies:
   ```bash
   cd "LLM Query Engine"
   pip install -r requirements.txt
   ```
3. Start Milvus services using Docker:
   ```bash
   cd "milvus-setup"
   docker-compose up -d
   ```
4. Run the application:
   ```bash
   cd ..
   python api_server.py --port 8002 --with-rag
   ```

## Next Steps

After successful installation:

1. Create RAG collections for your data
2. Configure client-specific settings
3. Integrate with your frontend applications
4. Test the API endpoints using the Swagger UI at http://localhost:8002/docs

For more detailed information, refer to the main `DEPLOYMENT.md` file.
