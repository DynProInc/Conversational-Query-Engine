# LLM Query Engine

A powerful conversational query engine that converts natural language questions into SQL queries and executes them against Snowflake databases. Supports multiple LLM providers (OpenAI, Anthropic Claude, and Google Gemini) and multiple clients with isolated configurations.

## Overview

This system allows users to query data in Snowflake databases using natural language. The engine:
- Converts natural language questions to SQL using state-of-the-art LLMs
- Executes the generated SQL against Snowflake
- Returns the query results in a structured format
- Tracks token usage and costs for all LLM interactions

## Architecture

The codebase follows a modular architecture to support multiple LLM providers and multiple clients:

### Core Components

1. **API Server** (`api_server.py`): FastAPI server that exposes endpoints for:
   - Converting natural language to SQL
   - Executing SQL against Snowflake
   - Client-specific and system-wide health checks
   - Model and client listings

2. **LLM Query Generators**:
   - OpenAI implementation (`llm_query_generator.py`)
   - Claude implementation (`claude_query_generator.py`) 
   - Gemini implementation (`gemini_query_generator.py`)

3. **End-to-End Pipelines**:
   - OpenAI: `nlq_to_snowflake.py`
   - Claude: `nlq_to_snowflake_claude.py`
   - Gemini: `nlq_to_snowflake_gemini.py`

4. **Database Connector** (`snowflake_runner.py`):
   - Handles Snowflake connectivity
   - Executes SQL queries
   - Returns results as pandas DataFrames

5. **Token Logger** (`token_logger.py`):
   - Tracks token usage across all LLM providers
   - Calculates costs based on model pricing
   - Logs to CSV with execution status flags

### Supporting Components

1. **Health Checks** (`health_check_utils.py`):
   - Verifies connectivity to LLM APIs with client-specific API keys
   - Checks Snowflake connection status with client-specific credentials
   - Supports both global and client-specific health monitoring

2. **Client Manager** (`config/client_manager.py`):
   - Manages multiple client configurations
   - Loads client-specific environment variables
   - Enforces strict client-specific API key validation

2. **Query History** (`prompt_query_history_api.py` & `prompt_query_history_route.py`):
   - Maintains history of user queries
   - Provides context for subsequent queries

3. **Error Handling** (`error_hint_utils.py`):
   - Provides user-friendly error messages
   - Suggests fixes for common issues

4. **Reporting** (`generate_query_report.py`):
   - Generates usage reports from token logs

## Data Flow

1. User submits a natural language question through the API
2. API server routes the request to the appropriate LLM provider
3. LLM converts the question to SQL using context from the data dictionary
4. SQL is optionally executed against Snowflake
5. Results are returned to the user
6. Token usage is logged for reporting and billing

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

# Configuration and environment
python-dotenv==1.1.0

# Additional dependencies
# See full requirements.txt for complete list
```

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

The system enforces strict client-specific API key validation with no automatic fallbacks between clients.

## Running the Application

Start the API server:
```
python api_server.py
```

The server will be available at `http://localhost:8000` with the following endpoints:

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

##### `/query/openai` - OpenAI SQL Generation

```json
POST /query/openai
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "data_dictionary_path": null,
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
| `limit_rows` | integer | *Optional* | 100 | Maximum number of rows to return |
| `data_dictionary_path` | string | *Optional* | null | Path to custom data dictionary (null uses default) |
| `execute_query` | boolean | *Optional* | true | Whether to execute the generated SQL |
| `model` | string | *Optional* | Depends on endpoint | Specific model to use |

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

The codebase includes comprehensive test scripts:
- `test_api.py`: Tests API endpoints
- `test_token_logging_comprehensive.py`: Tests token logging across all providers and scenarios

Run tests with:
```
python test_token_logging_comprehensive.py
```

## Project Structure

```
LLM Query Engine/
├── api_server.py                    # Main API server
├── llm_query_generator.py           # OpenAI query generator
├── claude_query_generator.py        # Claude query generator
├── gemini_query_generator.py        # Gemini query generator
├── nlq_to_snowflake.py              # OpenAI end-to-end pipeline
├── nlq_to_snowflake_claude.py       # Claude end-to-end pipeline
├── nlq_to_snowflake_gemini.py       # Gemini end-to-end pipeline
├── snowflake_runner.py              # Snowflake connection and query execution
├── token_logger.py                  # Token usage tracking and logging
├── health_check_utils.py            # API and database health checks
├── error_hint_utils.py              # User-friendly error messages
├── prompt_query_history_api.py      # Query history tracking
├── prompt_query_history_route.py    # Query history API endpoints
├── generate_query_report.py         # Usage reporting
├── requirements.txt                 # Package dependencies
├── test_token_logging_comprehensive.py  # Comprehensive token logging tests
├── test_api.py                      # API endpoint tests
├── Data Dictionary/                 # Contains schema information for prompts
└── token_usage.csv                  # Token usage logs
```

## Maintenance and Best Practices

- **Token Logging**: Always ensure that token logging is implemented correctly across all providers for cost tracking.
- **Error Handling**: All LLM and database operations include robust error handling.
- **Testing**: Use the comprehensive test scripts to verify functionality after changes.
- **Dependencies**: Keep the dependencies updated for security and feature improvements.
