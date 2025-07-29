# Query History and Saved Queries API Documentation

This document describes the API routes for managing query history and saved queries in the Conversational Query Engine.

## Base URL
All routes are prefixed with `/api`

## Authentication
**Note**: These routes do not require authentication. User identification is handled through the `user_id` parameter.

## Query History Routes

### GET `/api/query_history`
Retrieve query history for a specific user.

**Parameters:**
- `user_id` (string, required): The user ID to get history for

**Response:**
```json
[
  {
    "id": 1,
    "user_id": "user123",
    "prompt": "Show me all customers",
    "sql_query": "SELECT * FROM customers",
    "database": "default",
    "timestamp": "2024-01-15T10:30:00",
    "model": "claude-3-5-sonnet",
    "query_executed": "1",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "input_cost": 0.001,
    "output_cost": 0.002,
    "total_cost": 0.003
  }
]
```

### DELETE `/api/query_history/{query_id}`
Delete a specific query from history.

**Parameters:**
- `query_id` (integer, required): The ID of the query to delete
- `user_id` (string, required): The user ID (for verification)

**Response:**
```json
{
  "success": true
}
```

## Saved Queries Routes

### GET `/api/saved_queries`
Retrieve saved queries for a specific user.

**Parameters:**
- `user_id` (string, required): The user ID to get saved queries for

**Response:**
```json
[
  {
    "id": 1,
    "user_id": "user123",
    "prompt": "Show me all customers",
    "name": "Customer Query",
    "description": "A query to get all customers",
    "sql_query": "SELECT * FROM customers",
    "database": "default",
    "timestamp": "2024-01-15T10:30:00",
    "tags": ["customers", "simple"],
    "model": "claude-3-5-sonnet",
    "query_executed": "1",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "input_cost": 0.001,
    "output_cost": 0.002,
    "total_cost": 0.003
  }
]
```

### POST `/api/saved_queries`
Save a new query.

**Request Body:**
```json
{
  "user_id": "user123",
  "prompt": "Show me all customers",
  "name": "Customer Query",
  "description": "A query to get all customers",
  "sql_query": "SELECT * FROM customers",
  "database": "default",
  "tags": ["customers", "simple"]
}
```

**Response:**
```json
{
  "id": 1,
  "user_id": "user123",
  "prompt": "Show me all customers",
  "name": "Customer Query",
  "description": "A query to get all customers",
  "sql_query": "SELECT * FROM customers",
  "database": "default",
  "timestamp": "2024-01-15T10:30:00",
  "tags": ["customers", "simple"],
  "model": "unknown",
  "query_executed": "0",
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "total_tokens": 0,
  "input_cost": 0.0,
  "output_cost": 0.0,
  "total_cost": 0.0
}
```

### PUT `/api/saved_queries/{query_id}`
Update a saved query.

**Parameters:**
- `query_id` (integer, required): The ID of the query to update
- `user_id` (string, required): The user ID (for verification)

**Request Body:**
```json
{
  "name": "Updated Customer Query",
  "description": "Updated description",
  "tags": ["customers", "updated"]
}
```

**Response:**
```json
{
  "success": true
}
```

### DELETE `/api/saved_queries/{query_id}`
Delete a saved query.

**Parameters:**
- `query_id` (integer, required): The ID of the query to delete
- `user_id` (string, required): The user ID (for verification)

**Response:**
```json
{
  "success": true
}
```

## Utility Functions

The following utility functions are available for programmatically saving queries:

### `save_query_to_history()`
Save a query to the query history table.

```python
from utils.query_storage import save_query_to_history

history_id = save_query_to_history(
    user_id="user123",
    prompt="Show me all customers",
    sql_query="SELECT * FROM customers",
    model="claude-3-5-sonnet",
    query_executed=True,
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
    input_cost=0.001,
    output_cost=0.002,
    total_cost=0.003
)
```

### `save_query_to_saved_queries()`
Save a query to the saved queries table.

```python
from utils.query_storage import save_query_to_saved_queries

saved_id = save_query_to_saved_queries(
    user_id="user123",
    prompt="Show me all customers",
    name="Customer Query",
    description="A query to get all customers",
    sql_query="SELECT * FROM customers",
    tags=["customers", "simple"],
    model="claude-3-5-sonnet",
    query_executed=True,
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
    input_cost=0.001,
    output_cost=0.002,
    total_cost=0.003
)
```

## Database Schema

### QueryHistory Table
- `id` (Integer, Primary Key)
- `user_id` (String, Required)
- `prompt` (Text, Required)
- `sql_query` (Text, Optional)
- `database` (String, Default: 'default')
- `timestamp` (DateTime, Default: UTC now)
- `model` (String, Default: 'unknown')
- `query_executed` (Boolean, Default: False)
- `prompt_tokens` (Integer, Default: 0)
- `completion_tokens` (Integer, Default: 0)
- `total_tokens` (Integer, Default: 0)
- `input_cost` (Numeric, Default: 0)
- `output_cost` (Numeric, Default: 0)
- `total_cost` (Numeric, Default: 0)

### SavedQuery Table
- `id` (Integer, Primary Key)
- `user_id` (String, Required)
- `prompt` (Text, Required)
- `name` (String, Required)
- `description` (Text, Optional)
- `sql_query` (Text, Optional)
- `database` (String, Default: 'default')
- `timestamp` (DateTime, Default: UTC now)
- `tags` (JSON, Optional)
- `model` (String, Default: 'unknown')
- `query_executed` (Boolean, Default: False)
- `prompt_tokens` (Integer, Default: 0)
- `completion_tokens` (Integer, Default: 0)
- `total_tokens` (Integer, Default: 0)
- `input_cost` (Numeric, Default: 0)
- `output_cost` (Numeric, Default: 0)
- `total_cost` (Numeric, Default: 0)

## Error Responses

All routes return appropriate HTTP status codes:

- `200`: Success
- `201`: Created (for POST requests)
- `404`: Not Found (when trying to access non-existent resources)
- `500`: Internal Server Error

Error responses include a JSON object with an `error` field:

```json
{
  "error": "Query not found"
}
```

## Testing

Use the provided test script to verify the routes are working:

```bash
python test_query_history_routes.py
```

Make sure the API server is running on `http://localhost:8000` before running the tests. 