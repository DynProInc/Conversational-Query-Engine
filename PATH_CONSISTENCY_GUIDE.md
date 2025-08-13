# Path and Environment Consistency Guide

This guide explains how we've solved the file path consistency issues and environment-specific configuration between Docker and local environments in the Conversational Query Engine.

## Problems Addressed

1. **Path Inconsistency**: Different file paths between environments
   - **Docker environment**: Files are located at `/app/LLM Query Engine/...`
   - **Local environment**: Files are located at `C:/Users/git_manoj/Conversational-Query-Engine/LLM Query Engine/...`

2. **Milvus Connection Issues**: Connection parameters not properly configured for Docker
   - Docker containers need to use service names instead of localhost
   - Environment variables not being properly utilized

3. **Docker Container Management**: The app tries to manage Docker containers from within a container
   - This causes errors as Docker commands aren't available inside containers

4. **Client Data Dictionary Paths**: Absolute Windows paths in client_registry.csv causing issues in Docker
   - Windows paths like `C:/Users/git_manoj/...` don't exist in Docker containers

## Solution

We've implemented a comprehensive solution with these components:

1. **Path Resolver Modules**: Utilities that automatically detect the environment and provide consistent paths
2. **Docker Volume Mounts**: Updated Docker configuration to mount the entire directory structure
3. **Environment Variable Configuration**: Properly configured environment variables for Milvus connection
4. **Code Updates**: Modified code to use the path resolvers and environment variables
5. **Client Path Handling**: Special handling for client data dictionary paths in Docker

## 1. Path Resolver Modules

We created two path resolver modules:

### 1.1 Main Path Resolver (`path_resolver.py`)

- Located in the `milvus-setup` directory
- Handles general file paths for the application
- Provides methods for client registry and column mappings

```python
# Example usage
from path_resolver import get_path_resolver

path_resolver = get_path_resolver()
client_registry_path = path_resolver.get_client_registry_path()
column_mappings_path = path_resolver.get_column_mappings_path()
```

### 1.2 Client Path Resolver (`path_resolver_client.py`)

- Located in the `config` directory
- Specifically handles client data dictionary paths
- Detects Windows absolute paths and converts them to Docker paths

```python
# Example usage
from config.path_resolver_client import resolve_path

client_dict_path = resolve_path(relative_path)
```

## 2. Docker Volume Mounts

We updated the `docker-compose.yml` file to mount the entire directory structure:

```yaml
volumes:
  - "./LLM Query Engine:/app/LLM Query Engine"
  - "./LLM Query Engine/.env:/app/.env"
```

This ensures that the Docker container has access to the same configuration files as the local environment with the exact same directory structure.

## 3. Client Data Dictionary Path Handling

We've implemented a special solution for client data dictionary paths in the `client_manager.py` file:

```python
def get_data_dictionary_path(self, client_id: str) -> str:
    """
    Get the data dictionary path for a client
    """
    if client_id not in self.clients:
        raise ValueError(f"Client ID '{client_id}' not found")
    
    logger = logging.getLogger(__name__)
    
    # Check if we're running in Docker
    in_docker = os.path.exists('/.dockerenv') or os.path.exists('/app')
    
    if in_docker:
        # In Docker, use the fixed path structure
        return f"/app/LLM Query Engine/config/clients/data_dictionaries/{client_id}/{client_id}_dictionary.csv"
    else:
        # Get the relative path from client configuration for local environment
        relative_path = self.clients[client_id]['data_dictionary_path']
        
        # Use path resolver if available, otherwise return the path as is
        if path_resolver_available:
            resolved_path = resolve_path(relative_path)
            logger.info(f"Resolved path for client {client_id}: {resolved_path}")
            return resolved_path
        else:
            return relative_path
```

This approach:

1. Detects if the application is running in Docker
2. If in Docker, uses a fixed path structure based on client ID
3. If running locally, uses the path from client_registry.csv with path resolution

## 4. Docker Environment Detection

We've improved Docker environment detection in `api_server.py` to skip Docker container management when running in Docker:

```python
def check_if_docker():
    return os.path.exists('/.dockerenv') or os.path.exists('/app')

in_docker = check_if_docker()

if in_docker:
    print("ℹ️ Running in Docker environment, skipping Milvus container checks")
else:
    # container management logic
```

## 5. Milvus Connection Configuration

We've updated the Milvus connection in `rag_api.py` to use environment variables:

```python
milvus_host = os.environ.get("MILVUS_HOST", "localhost")
milvus_port = os.environ.get("MILVUS_PORT", "19530")
rag_manager = rag_module.RAGManager(
    milvus_host=milvus_host,
    milvus_port=milvus_port,
    enable_reranking=enable_reranking
)
```

And set these environment variables in `docker-compose.yml`:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - MILVUS_HOST=milvus-standalone
  - MILVUS_PORT=19530
```

## 3. Environment Variable Configuration

We've properly configured environment variables for Milvus connection in both Docker and local environments:

1. **Docker Compose Configuration**:
   ```yaml
   environment:
     - MILVUS_HOST=milvus-standalone
     - MILVUS_PORT=19530
   ```

2. **RAG API Initialization**:
   ```python
   # Get Milvus host and port from environment variables or use defaults
   milvus_host = os.environ.get("MILVUS_HOST", "localhost")
   milvus_port = os.environ.get("MILVUS_PORT", "19530")
   
   # Create RAGManager instance with environment variables
   rag_manager = rag_module.RAGManager(
       milvus_host=milvus_host,
       milvus_port=milvus_port,
       enable_reranking=enable_reranking
   )
   ```

3. **RAG Embedding Connection**:
   ```python
   def _connect_to_milvus(self):
       """Connect to Milvus server"""
       try:
           # Get host and port from environment variables if available
           milvus_host = os.environ.get("MILVUS_HOST", self.milvus_host)
           milvus_port = os.environ.get("MILVUS_PORT", self.milvus_port)
           
           logger.info(f"Attempting to connect to Milvus at {milvus_host}:{milvus_port}")
           connections.connect(alias="default", host=milvus_host, port=milvus_port)
       except Exception as e:
           logger.error(f"Failed to connect to Milvus: {e}")
           raise
   ```

## 4. Code Updates

We modified several files to ensure consistent behavior across environments:

### `rag_embedding.py`:
- Added proper path resolver import and initialization
- Updated client registry and column mappings path resolution
- Modified client dictionary path resolution in `get_active_clients()` and `process_schema_data()`
- Improved error handling and logging for path resolution failures

### `path_resolver.py`:
- Created a new utility module that detects the environment (Docker vs. local)
- Implemented methods to resolve paths consistently across environments
- Added proper exports with `__all__` to ensure module functions are accessible

## How to Use

### In Docker

1. The application will automatically detect it's running in Docker (`/app` exists)
2. It will use paths like `/app/config/clients/client_registry.csv`
3. Client dictionaries will be found at `/app/config/clients/data_dictionaries/[client_name]/[client_name]_dictionary.csv`

### Locally

1. The application will detect it's running locally (no `/app` directory)
2. It will use paths relative to the `LLM Query Engine` directory
3. Client dictionaries will be found at `[base_dir]/config/clients/data_dictionaries/[client_name]/[client_name]_dictionary.csv`

## Updating Files in Docker

After making changes to the path resolver or related files, update the Docker container:

```bash
# Copy path_resolver.py to Docker
docker cp "LLM Query Engine/milvus-setup/path_resolver.py" conversational-query-engine-app-1:/app/milvus-setup/path_resolver.py

# Copy rag_embedding.py to Docker
docker cp "LLM Query Engine/milvus-setup/rag_embedding.py" conversational-query-engine-app-1:/app/milvus-setup/rag_embedding.py

# Restart the app container
docker-compose restart app
```

## Troubleshooting

If you encounter path-related issues:

1. Check the logs to see which paths are being used:
   ```bash
   docker-compose logs -f app
   ```

2. Verify that the config directory is properly mounted:
   ```bash
   docker-compose exec app ls -la /app/config
   ```

3. Test the path resolver directly:
   ```bash
   docker-compose exec app python -c "from milvus-setup.path_resolver import get_path_resolver; p = get_path_resolver(); print(p.get_client_registry_path())"
   ```

## Next Steps

1. Consider adding more volume mounts if additional directories need to be shared
2. Update other modules to use the path resolver for consistent file access
3. Add unit tests to verify path resolution works correctly in both environments
