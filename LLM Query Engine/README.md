# LLM Query Engine

A powerful conversational query engine that converts natural language questions into SQL queries and executes them against Snowflake databases. Supports multiple LLM providers (OpenAI, Anthropic Claude, and Google Gemini) with strict client isolation, dynamic configuration, and RAG-enhanced context retrieval.

## Overview

This system allows users to query data in Snowflake databases using natural language. The engine:
- Converts natural language questions to SQL using state-of-the-art LLMs
- Utilizes Retrieval-Augmented Generation (RAG) for efficient schema context retrieval
- Executes the generated SQL against Snowflake with strict read-only validation
- Returns the query results in a structured format with interactive visualizations
- Tracks token usage and costs for all LLM interactions
- Supports multiple clients with isolated configurations, API keys, and data dictionaries

## Architecture

The codebase follows a modular architecture to support multiple LLM providers and multiple clients:

### Core Components

1. **API Server** (`api_server.py`): FastAPI server that exposes endpoints for:
   - Converting natural language to SQL
   - Executing SQL against Snowflake with read-only validation
   - Client-specific and system-wide health checks
   - Model and client listings
   - Unified query endpoint with support for all models simultaneously
   - RAG-enhanced context retrieval

2. **LLM Query Generators**:
   - OpenAI implementation (`llm_query_generator.py`)
   - Claude implementation (`claude_query_generator.py`) 
   - Gemini implementation (`gemini_query_generator.py`)
   - All with proper escape sequence handling for SQL generation

3. **End-to-End Pipelines**:
   - OpenAI: `nlq_to_snowflake.py`
   - Claude: `nlq_to_snowflake_claude.py`
   - Gemini: `nlq_to_snowflake_gemini.py`
   - All with strict client-specific API key validation and no automatic fallbacks

4. **Database Connector** (`snowflake_runner.py`):
   - Handles Snowflake connectivity with client-specific credentials
   - Automatically activates the specified warehouse before query execution
   - Executes SQL queries with read-only validation
   - Returns results as pandas DataFrames

5. **Token Logger** (`token_logger.py`):
   - Tracks token usage across all LLM providers
   - Calculates costs based on model pricing
   - Logs to CSV with execution status flags

### Supporting Components

1. **RAG System** (`milvus-setup/`):
   - Retrieval-Augmented Generation for efficient schema context
   - Reduces token usage by 50-60% compared to full schema context
   - Uses Milvus vector database with IP (Inner Product) metric
   - Dynamic index selection based on dataset size:
     - HNSW index for smaller datasets (<100,000 rows) with M=16 and efConstruction=200
     - IVF_FLAT index for larger datasets (≥100,000 rows) with dynamic nlist values:
       - nlist=100 for <50,000 rows
       - nlist=1024 for 50,000-500,000 rows
       - nlist=4096 for >500,000 rows
   - Advanced reranking with multiple model options:
     - Primary: BAAI/bge-reranker-large for highest accuracy
     - Fallback models: BAAI/bge-reranker-base and Qwen/Qwen3-Reranker-0.6B
     - Score fusion combining vector search and reranker scores
   - Client-specific collections with isolated embeddings

2. **Health Checks** (`health_check_utils.py`):
   - System-wide health endpoint (`/health`)
   - Client-specific health checks (`/health/client/{client_id}`)
   - Comprehensive client health dashboard (`/health/client`)
   - Verifies API keys, model configurations, and Snowflake connections
   - Returns detailed health status with timestamps

3. **Client Manager** (`config/client_manager.py`):
   - Manages multiple client configurations
   - Loads client-specific environment variables
   - Enforces strict client-specific API key validation with no automatic fallbacks
   - Provides client-specific data dictionary paths
   - Prevents cross-client data leakage

4. **SQL Validation** (`sql_structure_utils.py` & frontend):
   - Frontend and backend validation for read-only operations
   - Clear UI indicators for allowed operations (SELECT, SHOW, DESCRIBE, EXPLAIN only)
   - Client-specific Snowflake account information display
   - Dynamic validation messages

5. **Query History** (`prompt_query_history_api.py` & `prompt_query_history_route.py`):
   - Maintains history of user queries
   - Provides context for subsequent queries

6. **Error Handling** (`error_hint_utils.py`):
   - Provides user-friendly error messages
   - Suggests fixes for common issues
   - Consistent error messaging format across all providers

7. **Reporting** (`generate_query_report.py`):
   - Generates usage reports from token logs

8. **Chart Rendering** (`static/chart_viewer.html`):
   - Interactive data visualizations
   - Supports multiple chart types (line, bar, scatter, pie, area, mixed)
   - Proper handling of categorical X-axis values with ensureCategoricalXAxis function
     - Ensures x-axis values are treated as true categories
     - Applies specific settings for time-based discrete intervals (quarters, years, months)
     - Implements Plotly.js settings (categoryorder, dtick, constrain) to prevent continuous scaling
     - Includes null checks to prevent errors with chart types like pie charts
   - Smart scale detection and secondary axis support
   - Enhanced RAG parameter controls in the UI:
     - Collapsible section that appears when "Use RAG for context retrieval" is checked
     - top_k: Number of top results to return from RAG (numeric input)
     - enable_reranking: Option to enable reranking of RAG results (checkbox)
     - feedback_time_window_minutes: Time window for feedback in minutes (numeric input)
   - SQL validation with clear UI indicators:
     - Prominent warning box showing allowed operations (SELECT, SHOW, DESCRIBE, EXPLAIN only)
     - Client-specific Snowflake account information display
     - Client-side validation to prevent disallowed operations
     - Clear validation messages for forbidden operations
  

## Data Flow

1. User submits a natural language question through the API
2. API server routes the request to the appropriate LLM provider based on client configuration
3. If RAG is enabled, relevant schema context is retrieved from Milvus vector database
4. LLM converts the question to SQL using the retrieved context
5. SQL is validated for read-only operations and optionally executed against Snowflake
6. Results are returned to the user with interactive visualizations
7. Token usage is logged for reporting and billing
8. Feedback can be collected to improve future queries

## Token Logging System

The system tracks token usage across all LLM providers with three key states:
- **Query Generated Only** (`query_executed = None`): SQL was generated but not executed
- **Query Successfully Executed** (`query_executed = 1`): SQL was generated and executed successfully
- **Query Execution Failed** (`query_executed = 0`): SQL was generated but execution failed

Token usage is logged to `token_usage.csv` with metrics including:
- Timestamp
- Model used
- Prompt/query text
- Generated SQL
- Token counts (prompt/completion/total)
- Cost calculations
- Execution status

## Saved Queries System

The system includes a comprehensive saved queries functionality that allows users to save, organize, retrieve, and execute queries:

### Saved Query Features

- **Query Organization**:
  - Tag-based filtering
  - Folder-based organization
  - Search by content or metadata
  - Sort by date, usage, or other attributes

- **Query Execution**:
  - Execute saved queries directly
  - View execution history
  - Compare results over time

### Saved Queries API Endpoints

- **Save a Query**:
  ```
  POST /saved_queries
  ```
  Saves a query with all associated metadata, tags, and folder information.

- **Get All Saved Queries**:
  ```
  GET /saved_queries
  ```
  Retrieves all saved queries with optional filtering by user, tags, or folders.

- **Get a Specific Saved Query**:
  ```
  GET /saved_queries/{query_id}
  ```
  Retrieves details for a specific saved query by its ID.

- **Update a Saved Query**:
  ```
  PUT /saved_queries/{query_id}
  ```
  Updates metadata, tags, folder, or notes for an existing saved query.

- **Delete a Saved Query**:
  ```
  DELETE /saved_queries/{query_id}
  ```
  Removes a saved query from the system.

- **Execute a Saved Query**:
  ```
  POST /saved_queries/execute/{query_id}
  ```
  Executes a previously saved query and returns the results.

- **Get All Tags**:
  ```
  GET /saved_queries/tags
  ```
  Retrieves all unique tags used across saved queries.

- **Get All Folders**:
  ```
  GET /saved_queries/folders
  ```
  Retrieves all unique folders used for organizing saved queries.

### Saved Queries Storage

Saved queries are stored in `saved_queries.csv` with the following structure:
- Unique query ID
- User ID
- Timestamp
- Query text (prompt)
- Generated SQL
- Model used
- Token usage statistics
- Cost calculations
- Execution status
- Tags (as JSON array)
- Folder name
- Notes
- Execution ID reference
- RAG parameters (when applicable)

This file is automatically created if it doesn't exist when the application starts and should be added to `.gitignore` to prevent it from being tracked in version control.

## Automatically Generated Files

The system automatically generates several files during operation that don't need to be tracked in version control:

- **saved_queries.csv**: Stores user-saved queries with metadata as described above
- **feedback/executions.csv**: Tracks query executions for the feedback system
- **feedback/feedback.csv**: Stores user feedback on query results
- **cache/query_cache.csv**: Caches query results to improve performance and reduce API costs

These files are automatically created if they don't exist when the application starts. They should be added to `.gitignore` to prevent them from being tracked in version control.

## Setup and Configuration

### Requirements

The application requires Python 3.13.3 and the following packages (see `requirements.txt`):

```
# API framework
fastapi==0.115.14
uvicorn==0.35.0
pydantic==2.11.7
pydantic_core==2.33.2
starlette==0.46.2

# LLM providers
openai==1.93.0
anthropic==0.56.0
google-generativeai==0.8.5

# Data processing
pandas==2.3.0
openpyxl==3.1.5
numpy==2.2.6

# Database
snowflake-connector-python==3.15.0

# Vector database and embeddings
pymilvus==2.5.14
sentence-transformers==2.6.1

# Configuration and environment
python-dotenv==1.1.0

# Additional dependencies
# See full requirements.txt for complete list
```

### RAG System Setup

The RAG system requires Milvus vector database, which can be set up using Docker:

```bash
# For Windows
./milvus-setup/scripts/setup_milvus_containers.ps1

# For Linux/Mac
./milvus-setup/scripts/setup_milvus_containers.sh
```

This will start a Milvus standalone instance with the following services:
- etcd: Configuration and metadata storage
- minio: Object storage for vector data
- milvus-standalone: Vector database service

### Environment Variables

#### Global Environment Variables

Create a `.env` file with the following variables for system-wide settings:
```
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key

SNOWFLAKE_USER=your_snowflake_user
SNOWFLAKE_PASSWORD=your_snowflake_password
SNOWFLAKE_ACCOUNT=your_snowflake_account
SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
SNOWFLAKE_DATABASE=your_snowflake_database
SNOWFLAKE_SCHEMA=your_snowflake_schema
```

#### Client-Specific Environment Variables

Create client-specific environment files in `config/clients/env/{client_id}.env` with the following naming convention:

```
# Client-specific Snowflake credentials
CLIENT_{CLIENT_ID}_SNOWFLAKE_USER=client_specific_user
CLIENT_{CLIENT_ID}_SNOWFLAKE_PASSWORD=client_specific_password
CLIENT_{CLIENT_ID}_SNOWFLAKE_ACCOUNT=client_specific_account
CLIENT_{CLIENT_ID}_SNOWFLAKE_WAREHOUSE=client_specific_warehouse
CLIENT_{CLIENT_ID}_SNOWFLAKE_DATABASE=client_specific_database
CLIENT_{CLIENT_ID}_SNOWFLAKE_SCHEMA=client_specific_schema
CLIENT_{CLIENT_ID}_SNOWFLAKE_ROLE=client_specific_role

# Client-specific OpenAI configuration
CLIENT_{CLIENT_ID}_OPENAI_API_KEY=client_specific_openai_key
CLIENT_{CLIENT_ID}_OPENAI_MODEL=gpt-4o

# Client-specific Anthropic configuration
CLIENT_{CLIENT_ID}_ANTHROPIC_API_KEY=client_specific_anthropic_key
CLIENT_{CLIENT_ID}_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Client-specific Gemini configuration
CLIENT_{CLIENT_ID}_GEMINI_API_KEY=client_specific_gemini_key
CLIENT_{CLIENT_ID}_GEMINI_MODEL=models/gemini-2.5-flash
```

The system enforces strict client-specific API key validation with no automatic fallbacks between clients. Each client's configuration is completely isolated to prevent data leakage.

## Running the Application

### Starting the API Server

Start the API server manually:
```
python api_server.py
```

Or use the provided utility script to restart the server (useful during development):
```
.\restart_server.bat
```

The restart_server.bat script:
- Automatically finds and terminates any existing process running on port 8002
- Starts a new instance of the API server
- Provides clear console output about the termination and startup process

The server will be available at `http://localhost:8002` with the following endpoints:

### API Endpoints and Examples

#### Health Check Endpoints

##### `/health` - System-wide Health Check

```
GET /health
```

Checks the system-wide health status using default API keys and connections.

##### `/health/client/{client_id}` - Client-specific Health Check

```
GET /health/client/penguin
```

Verifies the specific client's API keys and configuration, returning detailed status for:
- OpenAI API connection
- Claude API connection
- Gemini API connection
- Snowflake connection

Response example:
```json
{
  "client_id": "penguin",
  "status": "healthy",
  "models": ["openai", "claude", "gemini"],
  "details": {
    "openai": {"ok": true, "msg": "Connected to OpenAI API"},
    "claude": {"ok": true, "msg": "Connected to Claude API"},
    "gemini": {"ok": true, "msg": "Connected to Gemini API"},
    "snowflake": {"ok": true, "msg": "Connected to Snowflake"}
  },
  "timestamp": "2025-07-14T15:58:40.700514"
}
```

##### `/health/client` - All Clients Health Check

```
GET /health/client
```

Checks the health status of all configured clients, returning a consolidated report with:
- Overall system health status
- Individual client status details
- Client count
- Timestamp

Response example:
```json
{
  "status": "degraded",
  "timestamp": "2025-07-14T16:28:15.392871",
  "client_count": 2,
  "clients": {
    "penguin": { /* client health details */ },
    "mts": { /* client health details */ }
  }
}
```

#### Query Endpoints

##### `/unified_query` - Multi-Model Support

```json
POST /unified_query
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "client_id": "penguin",
  "execute_query": true,
  "model": "openai"
}
```

The unified endpoint supports:
- All LLM providers through a single interface
- Client-specific configurations and data dictionaries
- Using `all` as a model parameter to compare results from all models simultaneously
- Specific model selection (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`, `models/gemini-2.5-flash`)

##### `/query/openai` - OpenAI SQL Generation

```json
POST /query/openai
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "client_id": "penguin",
  "execute_query": true,
  "model": "gpt-4o"
}
```

#### `/query/claude` - Claude SQL Generation

```json
POST /query/claude
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "data_dictionary_path": null,
  "execute_query": true,
  "model": "claude-3-5-sonnet-20241022"
}
```

#### `/query/gemini/execute` - Gemini SQL Generation

```json
POST /query/gemini/execute
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "data_dictionary_path": null,
  "execute_query": true,
  "model": "models/gemini-1.5-flash-latest"
}
```

#### `/query/compare` - Compare All Models

```json
POST /query/compare
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "data_dictionary_path": null,
  "execute_query": true
}
```

#### `/query/unified` - Unified Endpoint

```json
POST /query/unified
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "client_id": "penguin",
  "data_dictionary_path": null,
  "execute_query": true,
  "model": "openai"  // or "claude", "gemini", "gpt-4o", "all", etc.
}
```

The unified endpoint supports:
- Client-specific routing using `client_id`
- Strict client-specific API key validation
- Using "all" as a model parameter to compare results across all available models
- Client-specific data dictionaries

#### `/health` - Health Check Endpoint

```
GET /health
```

#### `/models` - List Available Models

```
GET /models
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | **Required** | - | Natural language question to convert to SQL |
| `client_id` | string | *Optional* | "mts" | Client identifier for multi-client support |
| `limit_rows` | integer | *Optional* | 100 | Maximum number of rows to return |
| `data_dictionary_path` | string | *Optional* | null | Path to custom data dictionary (null uses default) |
| `execute_query` | boolean | *Optional* | true | Whether to execute the generated SQL |
| `model` | string | *Optional* | Depends on endpoint | Specific model to use |
| `include_charts` | boolean | *Optional* | false | Whether to include chart recommendations |
| `edited_query` | string | *Optional* | null | For user-edited SQL queries |
| `use_rag` | boolean | *Optional* | false | Whether to use RAG for context retrieval |
| `top_k` | integer | *Optional* | 10 | Number of top results to return from RAG |
| `enable_reranking` | boolean | *Optional* | false | Whether to apply reranking to RAG results |
| `feedback_enhancement_mode` | string | *Optional* | "never" | Options: "never", "client_scoped", "high_confidence", "time_bounded", "explicit", "client_exact" |
| `max_feedback_entries` | integer | *Optional* | null | Maximum number of feedback entries to include |
| `confidence_threshold` | float | *Optional* | null | Minimum similarity threshold for fuzzy matching (0.0-1.0, default: 0.85) |
| `feedback_time_window_minutes` | integer | *Optional* | null | Time window for feedback in minutes (default: 20 minutes) |

### API Endpoints

#### Natural Language Query Endpoints

- `/query/unified` - Unified endpoint that routes to the selected model
- `/query/openai` - Convert NL to SQL using OpenAI
- `/query/claude` - Convert NL to SQL using Claude
- `/query/gemini` - Convert NL to SQL using Gemini
- `/query/compare` - Run the query through all models and compare results

#### Direct SQL Execution

- `/execute-query` - Execute a SQL query directly against Snowflake
  - Supports chart recommendations for query results
  - Handles edited SQL queries with original chart reuse when appropriate
  - Provides detailed execution metrics and error handling

#### Health Check and System Endpoints

- `/health` - System-wide health check
- `/health/client` - Health status for all clients
- `/health/client/{client_id}` - Client-specific health check
- `/models` - List available language models
- `/clients` - List all registered clients
- `/clients/{client_id}` - Get detailed info for a specific client

### Example Request with Only Required Parameters

```json
POST /query/openai
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024"
}
```

### Example Request with All Parameters

```json
POST /query/openai
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 50,
  "data_dictionary_path": "Data Dictionary/custom_schema.csv",
  "execute_query": true,
  "model": "gpt-4o"
}
```

### Example /execute-query Request

```json
POST /execute-query
Content-Type: application/json
X-Client-ID: mts

{
  "client_id": "mts",
  "query": "SELECT s.STORE_NUM_NAME, s.CITY, s.STATE, SUM(s.DAILY_TOTAL_SALE) as TOTAL_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES s WHERE EXTRACT(YEAR FROM s.BUSINESS_DATE) = 2024 GROUP BY s.STORE_NUM_NAME, s.CITY, s.STATE ORDER BY TOTAL_SALES DESC LIMIT 6",
  "limit_rows": 50,
  "original_prompt": "Show me the top 6 stores with highest sales in year 2024",
  "include_charts": true,
  "model": "gpt-4o"
}
```

### Example /execute-query Response

```json
{
  "query": "SELECT s.STORE_NUM_NAME, s.CITY, s.STATE, SUM(s.DAILY_TOTAL_SALE) as TOTAL_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES s WHERE EXTRACT(YEAR FROM s.BUSINESS_DATE) = 2024 GROUP BY s.STORE_NUM_NAME, s.CITY, s.STATE ORDER BY TOTAL_SALES DESC LIMIT 6",
  "query_output": [
    {
      "STORE_NUM_NAME": "123 - Main Street",
      "CITY": "New York",
      "STATE": "NY",
      "TOTAL_SALES": 1245678.90
    },
    // Additional rows...
  ],
  "success": true,
  "error_message": null,
  "execution_time_ms": 1831.69,
  "row_count": 6,
  "original_prompt": "Show me the top 6 stores with highest sales in year 2024",
  "edited": true,
  "chart_recommendations": [
    {
      "reasoning": "This data shows sales performance across different stores, which is well-suited for a bar chart to compare values.",
      "chart_config": {
        "chart_type": "bar",
        "title": "Top 6 Stores by Sales (2024)",
        "x_axis": "STORE_NUM_NAME",
        "y_axis": ["TOTAL_SALES"],
        "additional_config": {
          "orientation": "horizontal"
        }
      }
    }
  ],
  "chart_error": null
}
```

### Response Format

```json
{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "query": "SELECT s.STORE_NUM_NAME, s.CITY, s.STATE, SUM(s.DAILY_TOTAL_SALE) as TOTAL_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES s WHERE EXTRACT(YEAR FROM s.BUSINESS_DATE) = 2024 GROUP BY s.STORE_NUM_NAME, s.CITY, s.STATE ORDER BY TOTAL_SALES DESC LIMIT 6",
  "query_output": [
    {
      "STORE_NUM_NAME": "123 - Main Street",
      "CITY": "New York",
      "STATE": "NY",
      "TOTAL_SALES": 1245678.90
    },
    // Additional rows...
  ],
  "model": "gpt-4o",
  "token_usage": {
    "prompt_tokens": 1564,
    "completion_tokens": 65,
    "total_tokens": 1629
  },
  "success": true,
  "error_message": null,
  "execution_time_ms": 1831.69
}
```

## Testing

The codebase includes test scripts:
- `test_api.py`: Tests API endpoints
- `test_token_logging_comprehensive.py`: Tests token logging across all providers and scenarios

Run tests with:
```
python test_token_logging_comprehensive.py
```

## Project Structure

```
LLM Query Engine/
├── api_server.py                    # Main API server with unified query endpoint
├── llm_query_generator.py           # OpenAI query generator with optimized chart instructions
├── claude_query_generator.py        # Claude query generator with escape sequence handling
├── gemini_query_generator.py        # Gemini query generator with escape sequence handling
├── nlq_to_snowflake.py              # OpenAI end-to-end pipeline
├── nlq_to_snowflake_claude.py       # Claude end-to-end pipeline
├── nlq_to_snowflake_gemini.py       # Gemini end-to-end pipeline
├── snowflake_runner.py              # Snowflake connection with client-specific credentials
├── saved_queries.py                 # Saved queries functionality and API endpoints
├── cache_utils.py                   # Cache management with multiple backend support
├── token_logger.py                  # Token usage tracking and logging
├── health_check_utils.py            # System and client-specific health checks
├── error_hint_utils.py              # User-friendly error messages
├── prompt_query_history_api.py      # Query history tracking
├── prompt_query_history_route.py    # Query history API endpoints
├── rag_api.py                       # RAG system API endpoints
├── sql_structure_utils.py           # SQL validation and structure analysis
├── execute_query_route.py           # SQL execution with validation
├── generate_query_report.py         # Usage reporting
├── requirements.txt                 # Package dependencies
├── routes/                          # API route definitions
│   ├── feedback_route.py            # Feedback collection endpoints
│   └── dictionary_route.py          # Data dictionary management API
├── services/                        # Service layer implementations
│   ├── feedback_manager.py          # Feedback storage and retrieval
│   └── dictionary_service.py        # Dictionary processing service
├── config/
│   └── clients/                     # Client-specific configurations
│       ├── env/                      # Client environment files (.env)
│       ├── data_dictionaries/        # Contains schema information for prompts
│       │   ├── mts/                   # MTS client-specific dictionary
│       │   └── penguin/               # Penguin client-specific dictionary
│       ├── client_registry.csv       # Client registration information
│       └── feedback/                 # Client-specific feedback storage
├── feedback/                        # Feedback system
│   ├── executions.csv               # Execution tracking for feedback
│   └── feedback.csv                 # User feedback storage
├── milvus-setup/                    # RAG system implementation with Milvus
│   ├── docker-compose.yml            # Milvus container configuration
│   ├── rag_embedding.py              # Core RAG functionality
│   ├── multi_client_rag.py           # Client-specific RAG implementation
│   ├── rag_reranker.py               # Advanced reranking for improved relevance
│   ├── scripts/                      # Setup and maintenance scripts
│   │   ├── setup_milvus_containers.ps1 # Windows setup script
│   │   └── setup_milvus_containers.sh  # Linux/Mac setup script
│   ├── schema_processor.py           # Processes database schemas for embeddings
│   ├── generate_client_embeddings.py # Creates vector embeddings for clients
│   ├── rag_integration.py            # Integrates RAG with query generators
│   ├── rag_manager.py                # CLI tool for RAG collection management
│   ├── milvus_container_utils.py     # Docker container management utilities
│   ├── client_rag_manager.py         # Client-specific RAG management
│   ├── check_milvus_connection.py    # Milvus connection verification
│   ├── check_milvus_status.py        # Container status checking
│   ├── setup_milvus_windows.ps1      # Windows setup script for Milvus
│   ├── setup_milvus_containers.ps1   # PowerShell script for container setup
│   ├── setup.sh                      # Linux/Mac setup script
│   ├── scripts/                      # Additional setup scripts
│   │   └── setup_milvus_containers.ps1 # Container setup script (alternative)
│   └── SETUP_GUIDE.md                # Detailed RAG setup instructions
├── static/
│   ├── chart_viewer.html            # Enhanced interactive chart visualization
│   └── styles.css                    # Styling for web interface
└── token_usage.csv                  # Token usage logs
```

## Data Dictionary

The Data Dictionary is a crucial component that:
- Defines a data schema for NLQ to SQL conversion
- Supports client-specific data dictionaries in `Data Dictionary/{client_id}/`
- Ensures each client's queries are processed using their own data context
- Prevents cross-client data leakage and dictionary fallbacks

Each client's data dictionary path is strictly enforced through the client manager, avoiding silent fallbacks to other dictionaries. The system ensures that:

1. Client-specific dictionaries are used when available
2. No automatic fallbacks occur between clients' dictionaries
3. Proper error handling when dictionary paths are missing or invalid

## Chart Visualization

The system provides rich data visualization capabilities through an enhanced chart rendering system:

### Chart Types

Supports multiple chart types including:
- Line charts
- Bar charts
- Scatter plots
- Pie charts
- Area charts
- Mixed charts (combining different visualization types)

### Categorical X-Axis Handling

Implements special handling for categorical x-axis values:
- Ensures discrete categorical values are properly displayed using `ensureCategoricalXAxis()` function
- Prevents continuous scaling issues with categorical data
- Properly formats time-based discrete intervals (quarters, years, months)
- Applies Plotly.js settings (categoryorder, dtick, constrain) to prevent continuous scaling

### Smart Scale Detection

Optimized chart recommendations include:
- Automatic scale detection for numerical values
- Secondary Y-axis support for multi-scale data
- Proper handling for different value ranges in the same chart
- Token-optimized instructions for chart generation

## Multi-Client Dynamic Model Handling

- No hardcoded model names anywhere in the codebase
- Client-specific environment variables for model selection (e.g., `CLIENT_MTS_OPENAI_MODEL`)
- Strict client-specific API key validation for all models (OpenAI, Claude, Gemini)
- Consistent error messaging format across all API providers: "Client '{client_id}' {provider} API key not configured"
- Specific model name responses rather than generic provider names (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
- Support for "all" as a valid model parameter to compare results across providers by routing to the compare_models endpoint
- Proper client-specific data dictionary handling with no automatic fallbacks
- Client-specific health check endpoints to verify API keys and connections

## RAG System for Token Optimization

The system includes a Retrieval-Augmented Generation (RAG) implementation that significantly reduces token usage while improving query relevance:

### Benefits

- **Token Reduction**: 50-60% reduction in token consumption compared to sending full schema (from ~20,000+ to ~400-800 tokens)
- **Improved Accuracy**: More focused context leads to better SQL generation
- **Cost Efficiency**: Lower token usage translates to reduced API costs
- **Faster Response**: Smaller context means quicker processing by LLMs

### Technical Configuration

- **Vector Database**: Milvus v2.5.14
- **Embedding Model**: BAAI/bge-large-en-v1.5 via SentenceTransformer
- **Metric Type**: IP (Inner Product) - replaced COSINE which is not supported in Milvus v2.5.14
- **Index Type**: IVF_FLAT for optimal balance of build quality and query speed
- **Index Parameters**: 
  - nlist=64 (for ~369 rows of schema data)
  - nprobe=8 (for search operations)
- **Results Limit**: 8 most relevant schema items for context generation
- **Dual Optimization Approach**:
  - For index creation (rare operation): nlist=128 with IVF_FLAT and IP metric for fine-grained clustering
  - For daily searches (frequent operation): nprobe=8 with IP metric and limit=8 results for optimal SQL context generation

### Container Setup

The Milvus container setup uses docker-compose with three services:
- etcd: Configuration and metadata storage
- minio: Object storage for vector data
- milvus-standalone: Vector database service

### Integration

The RAG system is fully integrated with the multi-client architecture:
- Each client has its own isolated vector collection
- Client-specific schema data is embedded and stored separately
- Queries use the appropriate client collection based on context

## Feedback System

The system includes a comprehensive feedback collection and management system to continuously improve query generation:

### Feedback Collection

- **Types of Feedback**: Supports multiple feedback types:
  - Query correctness
  - SQL accuracy
  - Chart recommendations
  - Performance feedback
  - User experience
  - Thumbs up/down for quick sentiment capture
  - Suggestions for improvements
  - Corrections with SQL query fixes
  - Detailed text feedback

- **Feedback Endpoints**:
  ```
  POST /feedback/
  ```
  Allows users to submit feedback with:
  - Execution ID reference
  - Feedback type
  - Optional text explanation
  - Optional corrected SQL query
  - User identification

### Feedback Retrieval and Analysis

- **Feedback History**:
  ```
  GET /feedback/history
  ```
  Provides paginated access to feedback with:
  - Filtering by client ID
  - Optional inclusion of execution details
  - Sorting by timestamp (newest first)

- **Execution History**:
  ```
  GET /feedback/executions
  ```
  Retrieves execution history with:
  - Pagination and filtering options
  - Properly formatted SQL queries
  - Success/failure status

- **Execution Details**:
  ```
  GET /feedback/executions/{execution_id}
  ```
  Gets detailed information about specific executions including any associated feedback

### Feedback Integration

The system uses collected feedback to improve future query generation:

- **Context-Aware Suggestions**: Automatically retrieves relevant past feedback for similar prompts
- **Similarity Matching**: Uses fuzzy matching to find feedback for similar queries
- **Time-Window Filtering**: Prioritizes recent feedback within configurable time windows
- **Client-Specific Learning**: Optionally restricts feedback application to the same client

### Directory Structure

```
├── feedback/                        # Feedback system data storage
│   └── feedback.csv                  # Feedback data storage file
├── routes/
│   └── feedback_route.py             # API routes for feedback submission and retrieval
├── services/
│   └── feedback_manager.py           # Core feedback management functionality
├── utils/
│   └── file_csv_logger.py            # CSV logging utility for feedback and executions
```

## Maintenance and Best Practices

- **Token Logging**: Always ensure that token logging is implemented correctly across all providers for cost tracking.
- **Error Handling**: All LLM and database operations include robust error handling with consistent error messaging.
- **Testing**: Use the comprehensive test scripts to verify functionality after changes.
- **Dependencies**: Keep the dependencies updated for security and feature improvements.
- **Client Isolation**: Maintain strict client isolation for API keys, models, and data dictionaries.
- **RAG System**: Monitor token usage savings and regularly update embeddings when schemas change.