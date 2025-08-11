# Fresh Installation Guide for Conversational Query Engine

This guide provides detailed step-by-step instructions for setting up the Conversational Query Engine on a fresh machine (Windows, Linux, or macOS).

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

Save and close the file.

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

> **Note**: The first build may take 30-40 minutes depending on your internet connection and machine performance.

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

# RAG status check
curl http://localhost:8002/rag/stats
```

You can also access the API documentation at:
- http://localhost:8002/docs

## Step 6: Monitor the Application

View the application logs to ensure everything is running correctly:

```bash
docker-compose logs -f app
```

## Troubleshooting

### Container Startup Issues

If containers fail to start:

```bash
# Check container logs
docker-compose logs app
docker-compose logs milvus-standalone

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

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt for running commands
- Ensure Docker Desktop is running with WSL 2 backend for better performance
- Use double quotes for paths with spaces: `cd "LLM Query Engine"`

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

## Alternative: Manual Installation (Without Docker)

If you prefer not to use Docker:

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

For more detailed information, refer to the main `DEPLOYMENT.md` file.
