# LLM Query Engine - Technical Documentation

## Project Overview

The LLM Query Engine is an advanced natural language interface for database querying. It transforms natural language questions into SQL queries using Large Language Models (LLMs) and executes them against Snowflake databases. The system supports three major LLM providers:

1. OpenAI (GPT models)
2. Anthropic Claude
3. Google Gemini

## System Architecture

### High-Level Architecture

```
[User Query] → [API Server] → [LLM Processing] → [SQL Generation] → [Snowflake Execution] → [Results] → [User]
                    ↓                                                      ↓
                [Token Logging] ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← [Query Status]
```

### Components Breakdown

#### 1. API Server (`api_server.py`)

The FastAPI server is the central orchestrator for the entire system. It:

- Defines request/response models for all endpoints
- Routes requests to appropriate LLM providers
- Handles authentication and rate limiting
- Manages error responses and health checks
- Exposes the API endpoints for frontend applications

**Key API Endpoints:**
- `/query/openai`: Generates SQL using OpenAI
- `/query/claude`: Generates SQL using Claude
- `/query/gemini/execute`: Generates SQL using Gemini
- `/query/compare`: Compares results from all providers
- `/query`: Unified endpoint with model selection
- `/health`: Health check endpoint
- `/models`: Lists available models

#### 2. LLM Query Generators

Each LLM provider has its own implementation:

##### OpenAI (`llm_query_generator.py`)
- Handles formatting data dictionary for context
- Constructs prompts with table/column information
- Makes API calls to OpenAI
- Parses response to extract SQL
- Tracks token usage

##### Claude (`claude_query_generator.py`)
- Adapts OpenAI's prompt engineering for Claude
- Handles Claude's API requirements
- Tracks token usage for Claude models
- Reuses some utility functions from `llm_query_generator.py`

##### Gemini (`gemini_query_generator.py`)
- Implements Google's Gemini API
- Handles token counting through Google's API
- Provides fallback token estimation when API doesn't support counting
- Matches the interface of other LLM generators

#### 3. End-to-End Pipelines

Each LLM provider has a dedicated pipeline file:

##### OpenAI Pipeline (`nlq_to_snowflake.py`)
- Orchestrates the complete process for OpenAI
- Takes natural language query input
- Passes to `llm_query_generator` for SQL generation
- Optionally executes SQL via `snowflake_runner`
- Logs token usage and execution status

##### Claude Pipeline (`nlq_to_snowflake_claude.py`)
- Implements the same workflow for Claude models
- Handles Claude-specific error cases
- Logs token usage for Claude

##### Gemini Pipeline (`nlq_to_snowflake_gemini.py`)
- Implements the same workflow for Gemini models
- Includes Gemini-specific error handling
- Logs token usage with custom counting logic

#### 4. Snowflake Runner (`snowflake_runner.py`)

- Manages connections to Snowflake database
- Executes SQL queries
- Returns results as pandas DataFrames
- Handles connection pooling and retries
- Manages credentials from environment variables

#### 5. Token Logger (`token_logger.py`)

The token logging system is critical for tracking usage, costs, and execution status:

- Records timestamp, model, tokens, costs, and query status
- Handles three execution states:
  - Query generated but not executed (`query_executed = None`)
  - Query generated and executed successfully (`query_executed = 1`)
  - Query generated but execution failed (`query_executed = 0`)
- Writes to CSV file for persistent storage
- Calculates costs based on token counts and model rates

#### 6. Supporting Modules

- `health_check_utils.py`: Verifies API and database connectivity
- `error_hint_utils.py`: Provides user-friendly error messages
- `prompt_query_history_api.py`: Maintains conversation context
- `prompt_query_history_route.py`: API routes for history

## Data Flow

### Query Generation Flow

1. User submits natural language query via API
2. API server routes to appropriate LLM generator
3. LLM generator:
   - Prepares prompt with data dictionary context
   - Calls LLM API with prompt
   - Extracts SQL from response
   - Counts tokens used
4. Generated SQL is returned, along with token information

### Query Execution Flow (when `execute_query=true`)

1. Generated SQL is passed to Snowflake runner
2. Snowflake runner:
   - Establishes connection to Snowflake
   - Executes the SQL
   - Captures results as a DataFrame
   - Handles any execution errors
3. Results are returned to the API server
4. API server formats and returns results to the user

### Token Logging Flow

1. Token usage is captured during LLM API calls
2. Execution status is determined:
   - If execution not requested: `query_executed = None`
   - If execution successful: `query_executed = 1`
   - If execution failed: `query_executed = 0`
3. Token logger:
   - Calculates costs based on token count and model
   - Logs details to CSV with timestamp
   - Records prompt, SQL, and execution status

## Code Deep Dive

### OpenAI Query Generator

The OpenAI implementation demonstrates the core pattern used across all providers:

```python
def generate_sql_from_nl(self, prompt, model="gpt-4-turbo"):
    """
    Generates SQL from natural language using OpenAI.
    
    Args:
        prompt: Natural language prompt/question
        model: OpenAI model to use
        
    Returns:
        dict with:
        - query: Generated SQL string
        - tokens: Token usage information
        - error: Any error encountered
    """
    data_dict_prompt = self._prepare_data_dictionary()
    full_prompt = self._construct_prompt(prompt, data_dict_prompt)
    
    try:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a SQL expert..."},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        # Extract SQL from response
        sql_query = self._extract_sql_from_response(response.choices[0].message.content)
        
        # Get token usage
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return {"query": sql_query, "tokens": tokens, "error": None}
    
    except Exception as e:
        return {"query": None, "tokens": None, "error": str(e)}
```

### Token Logger

The `TokenLogger` class demonstrates the token logging system:

```python
def log_token_usage(self, model_name, prompt, query, 
                   prompt_tokens, completion_tokens, total_tokens, 
                   query_executed=None, model_category="openai"):
    """
    Logs token usage to CSV file.
    
    Args:
        model_name: Name of the LLM model used
        prompt: User's natural language prompt
        query: Generated SQL query
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        total_tokens: Total tokens used
        query_executed: Execution status (None=not executed, 1=success, 0=failed)
        model_category: Model provider category (openai, claude, gemini)
    """
    # Calculate cost based on model and tokens
    cost = self._calculate_cost(model_name, prompt_tokens, completion_tokens, model_category)
    
    # Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = [
        timestamp, model_name, prompt, query, 
        prompt_tokens, completion_tokens, total_tokens,
        cost, query_executed
    ]
    
    # Append to CSV
    with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(log_entry)
```

### Gemini Token Counting

The Gemini implementation shows how token counting is handled:

```python
def count_tokens(self, text):
    """Count tokens for Gemini models"""
    try:
        # Try using the official Google GenerativeAI token counting
        if hasattr(self.genai, "count_tokens"):
            result = self.genai.count_tokens(text)
            return result.total_tokens
        
        # Fallback to estimation based on characters
        else:
            # Approximate token count (3.5 chars per token is typical)
            return len(text) // 3
    except Exception as e:
        # Fallback to character-based estimation if API fails
        logger.warning(f"Token counting failed: {e}. Using character estimation.")
        return len(text) // 3
```

## Configuration

### Environment Variables

The system requires the following environment variables:

```
# LLM API Keys
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key

# Snowflake Configuration
SNOWFLAKE_USER=your_snowflake_user
SNOWFLAKE_PASSWORD=your_snowflake_password
SNOWFLAKE_ACCOUNT=your_snowflake_account
SNOWFLAKE_WAREHOUSE=your_snowflake_warehouse
SNOWFLAKE_DATABASE=your_snowflake_database
SNOWFLAKE_SCHEMA=your_snowflake_schema
```

### Data Dictionary

The data dictionary is a critical component that provides context to the LLMs. It contains:

- Table names and descriptions
- Column names, data types, and descriptions
- Relationships between tables
- Sample queries for context

It can be provided in CSV format with a specific structure that the system parses and formats for LLM context.

## Testing

### Token Logging Tests

The comprehensive token logging test (`test_token_logging_comprehensive.py`) verifies that token usage is correctly tracked across all LLM providers and execution scenarios:

1. **Query Generated Only**: Tests that tokens are logged when SQL is generated but not executed
2. **Query Successfully Executed**: Tests that tokens are logged with successful execution status
3. **Query Execution Failed**: Tests that tokens are logged with failed execution status

For each scenario, the test:
- Runs the query through the appropriate LLM pipeline
- Checks that a log entry was created in `token_usage.csv`
- Verifies that token counts and execution status are correct

### API Tests

The API tests (`test_api.py`) verify that all API endpoints:
- Accept the correct parameters
- Return proper responses
- Handle errors gracefully
- Log tokens correctly

## Common Challenges and Solutions

### 1. Token Logging for Different LLM Providers

**Challenge**: Each LLM provider reports token usage differently.

**Solution**: 
- OpenAI: Direct from API response usage field
- Claude: Direct from API response usage field
- Gemini: Custom implementation using Google's count_tokens API with fallback

### 2. Error Handling

**Challenge**: Handling various failure points (LLM API, SQL parsing, Snowflake execution)

**Solution**: Multi-layered error handling:
- Try/except blocks at each critical point
- Error propagation with context
- User-friendly error messages
- Proper execution status logging

### 3. Context Management

**Challenge**: Providing sufficient context to LLMs without exceeding token limits

**Solution**: 
- Dynamic context selection based on query
- Schema summarization for large databases
- Query history for maintaining conversation context

## Deployment Considerations

### Scalability

- API server can be containerized for horizontal scaling
- Token logging can be adapted to use a database instead of CSV for high-volume use
- Connection pooling for Snowflake reduces connection overhead

### Security

- API keys stored in environment variables
- No hardcoded credentials
- Input validation on all API endpoints
- Limited SQL execution permissions

### Monitoring

- Token usage logs provide basis for usage monitoring
- Health check endpoints for service monitoring
- Error logging for debugging

## Extending the System

### Adding a New LLM Provider

To add support for a new LLM provider:

1. Create a new query generator file (e.g., `new_provider_query_generator.py`)
2. Create a new pipeline file (e.g., `nlq_to_snowflake_new_provider.py`)
3. Add provider to API server routes
4. Update token logger to support the new provider's pricing

### Supporting Additional Databases

The system can be extended to support databases beyond Snowflake:

1. Create a new runner module (e.g., `postgres_runner.py`)
2. Update prompts to include database-specific context
3. Modify execution logic in pipeline files

## Conclusion

The LLM Query Engine is a sophisticated system that leverages the power of multiple LLM providers to make database querying more accessible through natural language. Its modular architecture allows for easy extension and maintenance, while the comprehensive token logging system ensures usage can be tracked and monitored effectively.

The system demonstrates best practices in:
- API design and implementation
- Error handling and robustness
- Multi-provider LLM integration
- Database connectivity
- Usage tracking and reporting

By following this technical documentation, developers should be able to understand, use, extend, and maintain the LLM Query Engine system effectively.
