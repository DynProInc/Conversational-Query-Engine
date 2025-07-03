"""Claude LLM Query Generator - A module to convert natural language to SQL using Anthropic Claude
"""
import os
import pandas as pd
import datetime
import dotenv
from typing import Dict, List, Any, Optional
from token_logger import TokenLogger

# Load environment variables from .env file
dotenv.load_dotenv()

# Reuse these functions from the OpenAI implementation
from llm_query_generator import load_data_dictionary, format_data_dictionary

# Check if anthropic module is installed, install if not
try:
    import anthropic
except ImportError:
    print("Anthropic module not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "anthropic"])
    import anthropic


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str) -> str:
    """
    Generate system prompt for Claude with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        
    Returns:
        Formatted system prompt for Claude
    """
    # Format tables info into string
    tables_context = ""
    for table in tables:
        # Include schema in table name if available
        if table['schema']:
            tables_context += f"Table: {table['name']} (Schema: {table['schema']})\n"
        else:
            tables_context += f"Table: {table['name']}\n"
            
        if table['description']:
            tables_context += f"Description: {table['description']}\n"
            
        tables_context += "Columns:\n"
        
        for col in table['columns']:
            pk_indicator = " (PRIMARY KEY)" if col.get('is_primary_key') else ""
            fk_info = ""
            if col.get('is_foreign_key'):
                fk_info = f" (FOREIGN KEY to {col.get('foreign_key_table')}.{col.get('foreign_key_column')})"
            
            tables_context += f"  - {col['name']} ({col['type']}){pk_indicator}{fk_info}: {col['description']}"
            
            # Only add business name if it's different from column name
            if col['business_name'] != col['name']:
                tables_context += f" [Business Name: {col['business_name']}]"
                
            tables_context += "\n"
        
        tables_context += "\n"
    
    # Create prompt template for Claude
    prompt = f"""You are an expert SQL query generator for Snowflake database.

Your task is to convert natural language questions into valid SQL queries that can run on Snowflake.
Use the following data dictionary to understand the database schema:

{tables_context}

When generating SQL:
1. Use proper Snowflake SQL syntax with fully qualified table names including schema (e.g., SCHEMA.TABLE_NAME)
2. Include appropriate JOINs based on the data relationships or column name similarities
3. Format the SQL code clearly with proper indentation and aliases
4. Only use tables and columns that exist in the provided schema
5. Add helpful SQL comments to explain complex parts of the query
6. Return ONLY the SQL code without any other text or explanations
7. Limit results to 100 rows unless specified otherwise

Generate a SQL query for: {query}
"""
    return prompt


def generate_sql_query_claude(api_key: str, prompt: str, model: str = "claude-3-5-sonnet-20241022", 
                         query_text: str = "", log_tokens: bool = True) -> Dict[str, Any]:
    """
    Generate SQL query using Anthropic Claude API
    
    Args:
        api_key: Anthropic API key
        prompt: System prompt with schema and query
        model: Claude model to use (default: claude-3-5-sonnet)
        query_text: Original natural language query (for logging)
        log_tokens: Whether to log token usage
        
    Returns:
        Dictionary with SQL query and token usage info
    """
    client = anthropic.Anthropic(api_key=api_key)
    logger = TokenLogger() if log_tokens else None
    
    try:
        print(f"DEBUG: Calling Claude API with model: {model}")
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print("DEBUG: Claude API call successful")
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response content length: {len(response.content) if hasattr(response, 'content') else 'No content'}")
        if hasattr(response, 'content') and len(response.content) > 0:
            print(f"DEBUG: First content item type: {type(response.content[0])}")
            print(f"DEBUG: First content item text sample: {response.content[0].text[:100] if hasattr(response.content[0], 'text') else 'No text attribute'}")
        else:
            print("DEBUG: No content items in response")
        
        # Extract SQL query from response
        sql_query = None
        if response and hasattr(response, 'content') and response.content and len(response.content) > 0:
            # Make sure we're getting text content
            if hasattr(response.content[0], 'text'):
                if response.content[0].text is None:
                    print("DEBUG: response.content[0].text is None! Forcing fallback SQL.")
                    full_text = ""
                else:
                    full_text = response.content[0].text
                print(f"DEBUG: Full text type: {type(full_text)}")
                if not isinstance(full_text, str):
                    print(f"DEBUG: full_text is not a string ({type(full_text)}), converting to string.")
                    full_text = str(full_text)
                full_text = full_text.strip() if full_text else ""
                print(f"DEBUG: Full text length: {len(full_text)}")
                
                # Extract and clean code block if present
                if full_text and "```" in full_text:
                    # Handle SQL code blocks
                    import re
                    # Look for SQL code blocks with different markers: ```sql, ```SQL, ``` sql, etc.
                    sql_blocks = re.findall(r'```(?:sql|SQL)?([\s\S]*?)```', full_text)
                    
                    if sql_blocks:
                        # Use the first SQL block found
                        sql_candidate = sql_blocks[0]
                        if sql_candidate is None:
                            print("DEBUG: sql_candidate from sql_blocks is None! Forcing fallback SQL.")
                            sql_query = "SELECT 'Error: SQL block was None' AS error_message"
                        else:
                            sql_query = sql_candidate.strip() if isinstance(sql_candidate, str) else str(sql_candidate).strip()
                        print(f"DEBUG: Extracted SQL from code block: {sql_query[:50]}...")
                    else:
                        # If no SQL block found but we have code blocks, use the first code block
                        code_blocks = re.findall(r'```([\s\S]*?)```', full_text)
                        if code_blocks:
                            code_candidate = code_blocks[0]
                            if code_candidate is None:
                                print("DEBUG: code_candidate from code_blocks is None! Forcing fallback SQL.")
                                sql_query = "SELECT 'Error: Code block was None' AS error_message"
                            else:
                                sql_query = code_candidate.strip() if isinstance(code_candidate, str) else str(code_candidate).strip()
                            print(f"DEBUG: Extracted SQL from generic code block: {sql_query[:50]}...")
                
                # If no code blocks or couldn't extract SQL, use the full response
                if not sql_query:
                    # If response contains SELECT, use everything
                    if "SELECT" in full_text.upper():
                        sql_query = full_text
                        print(f"DEBUG: Using full text as SQL: {sql_query[:50]}...")
            else:
                print(f"WARNING: Response content doesn't have text attribute: {type(response.content[0])}")
        else:
            print(f"WARNING: Invalid response structure: {response}")
                    
        # Ensure we return a valid string for SQL even if extraction failed (OpenAI pattern)
        if not isinstance(sql_query, str) or not sql_query:
            print(f"WARNING: SQL query is None or not a string, setting fallback SQL string (OpenAI pattern)")
            sql_query = "SELECT 'Error: Could not extract valid SQL from Claude response' AS error_message"
        
        # Get token usage
        usage_data = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        # Log token usage if requested
        if logger and query_text:
            logger.log_usage(model, query_text, usage_data)
            
        return {
            "sql": sql_query,
            "model": model,
            "prompt_tokens": usage_data["prompt_tokens"],
            "completion_tokens": usage_data["completion_tokens"],
            "total_tokens": usage_data["total_tokens"]
        }
        
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"DEBUG: Exception in generate_sql_query_claude: {str(e)}\n{tb_str}")
        error_msg = f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}\nTRACEBACK: {tb_str.replace("'", "''").replace(chr(10), ' ')}' AS error_message"
        return {
            "sql": error_msg,
            "model": model,
            "error": str(e),
            "traceback": tb_str,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


def natural_language_to_sql_claude(query: str, data_dictionary_path: str, api_key: Optional[str] = None, 
                               model: str = "claude-3-5-sonnet-20241022", log_tokens: bool = True) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL using Claude
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to data dictionary Excel/CSV
        api_key: Anthropic API key (will use environment variable if not provided)
        model: Claude model to use
        log_tokens: Whether to log token usage
        
    Returns:
        Dictionary with SQL query, token usage and other metadata
    """
    print(f"DEBUG: Starting natural_language_to_sql_claude with query: {query}")
    
    # Set default API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    try:
        # Track execution time
        start_time = datetime.datetime.now()
        
        # Load data dictionary
        df = load_data_dictionary(data_dictionary_path)
        
        # Format data dictionary for prompt
        tables = format_data_dictionary(df)
        
        # Generate prompt
        prompt = generate_sql_prompt(tables, query)
        
        # Generate SQL
        result = generate_sql_query_claude(
            api_key=api_key,
            prompt=prompt,
            model=model,
            query_text=query,
            log_tokens=log_tokens
        )
        
        print(f"DEBUG: SQL from Claude query generator: '{result.get('sql', 'No SQL found')}', type: {type(result.get('sql'))}")
        
        # Ensure 'sql' key always exists and contains a valid string
        if 'sql' not in result or result['sql'] is None or not isinstance(result['sql'], str):
            print(f"DEBUG: SQL missing or invalid in result, setting default SQL message")
            result['sql'] = "SELECT 'No valid SQL query was generated' AS message"
        
        # Add additional metadata
        result.update({
            "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "data_dictionary": data_dictionary_path
        })
        
        print(f"DEBUG: Returning result with SQL: '{result['sql'][:50]}...' (truncated)")
        return result
    
    except Exception as e:
        print(f"DEBUG: Exception in natural_language_to_sql_claude: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "sql": f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}' AS error_message",
            "model": model,
            "error": str(e),
            "query": query,
            "data_dictionary": data_dictionary_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if we have a query from command line
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "Show me the top 5 stores with highest sales"
        
    # Check if we have a data dictionary path
    data_dict_path = None
    if len(sys.argv) > 2:
        data_dict_path = sys.argv[2]
    else:
        # Use default data dictionary path
        data_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "Data Dictionary", "mts.csv")
    
    # Set the model name
    model = "claude-3-5-sonnet-20241022"
    
    print(f"Testing Claude SQL generation with query: '{query}'")
    try:
        result = natural_language_to_sql_claude(query, data_dict_path, model=model)
        print("\nGenerated SQL:")
        print(result["sql"])
        print(f"\nModel: {result['model']}")
        print(f"Execution time: {result.get('execution_time_ms', 0):.2f} ms")
        print(f"Prompt tokens: {result.get('prompt_tokens', 0)}")
        print(f"Completion tokens: {result.get('completion_tokens', 0)}")
        print(f"Total tokens: {result.get('total_tokens', 0)}")
    except Exception as e:
        print(f"Error: {str(e)}")

