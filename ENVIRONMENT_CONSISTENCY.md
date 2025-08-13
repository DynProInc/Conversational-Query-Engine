# Environment Consistency Guide

This guide ensures that both Docker and local environments maintain the same structure and file naming format.

## Directory Structure

Ensure the following directory structure is maintained in both environments:

```
Conversational-Query-Engine/
├── Dockerfile
├── docker-compose.yml
├── DEPLOYMENT.md
├── FRESH_INSTALLATION_GUIDE.md
├── ENVIRONMENT_CONSISTENCY.md
├── volumes/                       # Docker volumes directory
│   ├── etcd/
│   ├── minio/
│   └── milvus/
└── LLM Query Engine/
    ├── api_server.py              # Main API server
    ├── llm_query_generator.py     # LLM query generator
    ├── rag_api.py                 # RAG API endpoints
    ├── requirements.txt           # Python dependencies
    ├── .env                       # Environment variables (not in git)
    ├── config/
    │   ├── client_manager.py
    │   ├── column_mappings.json
    │   └── clients/
    │       ├── client_registry.csv
    │       └── data_dictionaries/
    │           └── [client_name]/
    │               └── [client_name]_dictionary.csv
    └── milvus-setup/
        ├── docker-compose.yml     # Local Milvus setup
        ├── rag_embedding.py       # RAG embedding manager
        ├── rag_reranker.py        # Reranking module
        └── schema_processor.py    # Schema processing
```

## File Synchronization

To ensure both environments have identical files:

### 1. Docker to Local Sync

```bash
# Copy files from Docker container to local
docker cp conversational-query-engine-app-1:/app/milvus-setup/rag_embedding.py "./LLM Query Engine/milvus-setup/rag_embedding.py"
docker cp conversational-query-engine-app-1:/app/rag_api.py "./LLM Query Engine/rag_api.py"
```

### 2. Local to Docker Sync

```bash
# Copy files from local to Docker container
docker cp "./LLM Query Engine/milvus-setup/rag_embedding.py" conversational-query-engine-app-1:/app/milvus-setup/rag_embedding.py
docker cp "./LLM Query Engine/rag_api.py" conversational-query-engine-app-1:/app/rag_api.py
```

## Environment Variables

Ensure the same environment variables are used in both environments:

### Docker Environment

In `docker-compose.yml`:
```yaml
environment:
  - MILVUS_HOST=milvus-standalone
  - MILVUS_PORT=19530
```

### Local Environment

In `.env` file or environment variables:
```
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## Maintaining Consistency

### After Code Changes

1. Always update both environments when making changes
2. Use the sync commands above after each change
3. Restart the affected services:
   ```bash
   # Docker
   docker-compose restart app
   
   # Local
   # Stop and restart the Python process
   ```

### Version Control

1. Commit changes to both environments simultaneously
2. Use a consistent branch strategy
3. Document any environment-specific changes

## Testing Consistency

Run the same tests in both environments to ensure they behave identically:

```bash
# Test RAG stats endpoint
curl http://localhost:8002/rag/stats

# Test health endpoint
curl http://localhost:8002/health/client
```

## Troubleshooting Inconsistencies

If you notice differences between environments:

1. Compare file contents:
   ```bash
   # Extract file from Docker
   docker cp conversational-query-engine-app-1:/app/rag_api.py ./docker_rag_api.py
   
   # Compare with local
   diff "./LLM Query Engine/rag_api.py" ./docker_rag_api.py
   ```

2. Check environment variables:
   ```bash
   # Docker
   docker-compose exec app env | grep MILVUS
   
   # Local
   env | grep MILVUS
   ```

3. Verify file permissions and ownership
4. Check for any environment-specific configuration files
