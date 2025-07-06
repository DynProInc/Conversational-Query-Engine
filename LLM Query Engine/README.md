# LLM Query Engine

A powerful conversational query engine that converts natural language questions into SQL queries and executes them against Snowflake databases. Supports multiple LLM providers: OpenAI, Anthropic Claude, and Google Gemini.

## Overview

This system allows users to query data in Snowflake databases using natural language. The engine:
- Converts natural language questions to SQL using state-of-the-art LLMs
- Executes the generated SQL against Snowflake
- Returns the query results in a structured format
- Tracks token usage and costs for all LLM interactions

## Architecture

The codebase follows a modular architecture to support multiple LLM providers:

### Core Components

1. **API Server** (`api_server.py`): FastAPI server that exposes endpoints for:
   - Converting natural language to SQL
   - Executing SQL against Snowflake
   - Health checks and model listings

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
   - Verifies connectivity to LLM APIs
   - Checks Snowflake connection status

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

The application requires the following packages (see `requirements.txt`):
```
openai>=1.0.0
pandas>=1.5.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
snowflake-connector-python>=3.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
google-generativeai>=0.3.0
anthropic>=0.5.0
```

### Environment Variables

Create a `.env` file with the following variables:
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

## Running the Application

Start the API server:
```
python api_server.py
```

The server will be available at `http://localhost:8000` with the following endpoints:

### API Endpoints and Examples

#### `/query/openai` - OpenAI SQL Generation

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

#### `/query` - Unified Endpoint

```json
POST /query/unified
Content-Type: application/json

{
  "prompt": "Show me the top 6 stores with highest sales in year 2024",
  "limit_rows": 100,
  "data_dictionary_path": null,
  "execute_query": true,
  "model": "openai"  // or "claude", "gemini", "gpt-4o", etc.
}
```

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
