"""
LLM Query Generator - A simple module to convert natural language to SQL using OpenAI
"""
import os
import pandas as pd
import openai
import datetime
import dotenv
from typing import Dict, List, Any, Optional
from token_logger import TokenLogger

# Load environment variables from .env file
dotenv.load_dotenv()


def load_data_dictionary(file_path: str) -> pd.DataFrame:
    """
    Load data dictionary from Excel or CSV file
    
    Args:
        file_path: Path to Excel or CSV file with data dictionary
        
    Returns:
        DataFrame containing the data dictionary
    """
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel or CSV file.")


def format_data_dictionary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Format data dictionary DataFrame for LLM prompt
    
    Args:
        df: DataFrame with data dictionary
        
    Returns:
        List of formatted table dictionaries
    """
    tables = []
    required_columns = ['TABLE_NAME', 'COLUMN_NAME', 'DESCRIPTION']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data dictionary")
    
    # Add DB_SCHEMA if available
    schema_aware = 'DB_SCHEMA' in df.columns
    
    # Group by table name (and schema if available)
    if schema_aware:
        # Create a compound key for grouping that includes schema
        df['FULL_TABLE_NAME'] = df['DB_SCHEMA'] + '.' + df['TABLE_NAME']
        groupby_col = 'FULL_TABLE_NAME'
    else:
        groupby_col = 'TABLE_NAME'
    
    for full_table_name, group in df.groupby(groupby_col):
        columns = []
        for _, row in group.iterrows():
            column_info = {
                'name': row['COLUMN_NAME'],
                # Use a default type if not available
                'type': row.get('DATA_TYPE', 'VARCHAR'),
                'description': row['DESCRIPTION'],
                # Use the column name as business name if not provided
                'business_name': row.get('BUSINESS_NAME', row['COLUMN_NAME']),
                # Handle optional fields with safe defaults
                'is_primary_key': bool(row.get('IS_PRIMARY_KEY', False)),
                'is_foreign_key': bool(row.get('IS_FOREIGN_KEY', False)),
                'foreign_key_table': row.get('FOREIGN_KEY_TABLE', ''),
                'foreign_key_column': row.get('FOREIGN_KEY_COLUMN', '')
            }
            columns.append(column_info)
        
        # Use the first row's TABLE_NAME and DB_SCHEMA
        table_name = group['TABLE_NAME'].iloc[0]
        db_schema = group['DB_SCHEMA'].iloc[0] if schema_aware else ''
        
        table_info = {
            'name': full_table_name,
            'schema': db_schema if schema_aware else '',
            'simple_name': table_name,
            'description': group.get('TABLE_DESCRIPTION', '').iloc[0] if 'TABLE_DESCRIPTION' in group.columns else '',
            'columns': columns
        }
        tables.append(table_info)
    
    return tables


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str) -> str:
    """
    Generate system prompt for the LLM with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        
    Returns:
        Formatted system prompt for the LLM
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
    
    # Create prompt template
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


def generate_sql_query(api_key: str, prompt: str, model: str = "gpt-4", 
                    query_text: str = "", log_tokens: bool = True) -> Dict[str, Any]:
    """
    Generate SQL query using OpenAI's API
    
    Args:
        api_key: OpenAI API key
        prompt: System prompt with schema and query
        model: OpenAI model to use
        query_text: Original natural language query (for logging)
        log_tokens: Whether to log token usage
        
    Returns:
        Dictionary with SQL query and token usage info
    """
    client = openai.OpenAI(api_key=api_key)
    logger = TokenLogger() if log_tokens else None
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        # Extract SQL query from response
        sql_query = response.choices[0].message.content.strip()
        
        # Get token usage
        usage_data = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Log token usage if requested
        if logger and query_text:
            logger.log_usage(model, query_text, usage_data)
            
        return {
            "sql": sql_query,
            "model": model,
            **usage_data
        }
    except Exception as e:
        error_msg = f"Error generating SQL query: {str(e)}"
        return {
            "sql": error_msg,
            "model": model,
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


def natural_language_to_sql(query: str, data_dictionary_path: Optional[str] = None, 
                     api_key: Optional[str] = None, model: str = "gpt-4o", log_tokens: bool = True,
                     model_provider: str = "openai") -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to data dictionary Excel/CSV
        api_key: OpenAI API key (will use environment variable if not provided)
        model: OpenAI model to use
        log_tokens: Whether to log token usage
        
    Returns:
        Dictionary with SQL query, token usage and other metadata
    """
    # Choose appropriate API key and model provider
    if model_provider.lower() == "claude":
        # Import Claude functionality only when needed
        from claude_query_generator import natural_language_to_sql_claude
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Use Claude for SQL generation
        return natural_language_to_sql_claude(query, data_dictionary_path, api_key, model, log_tokens)
    else:  # Default to OpenAI
        # Default to environment variable if API key not provided
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Process data dictionary
    df = load_data_dictionary(data_dictionary_path)
    tables = format_data_dictionary(df)
    
    # Generate prompt and SQL query
    prompt = generate_sql_prompt(tables, query)
    result = generate_sql_query(api_key, prompt, model=model, query_text=query, log_tokens=log_tokens)
    
    # Add additional info to the result
    result["query"] = query
    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["data_dictionary"] = data_dictionary_path
    
    return result


# Example usage
if __name__ == "__main__":
    # Example usage
    from pprint import pprint
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test SQL generation with different models")
    parser.add_argument("--query", "-q", default="Show me the top 5 stores with highest sales", 
                        help="Natural language query")
    parser.add_argument("--model", "-m", default="openai", choices=["openai", "claude"],
                        help="Model provider to use (openai or claude)")
    parser.add_argument("--specific-model", "-s", help="Specific model version")
    args = parser.parse_args()
    
    # Get command line arguments or use defaults
    query = args.query
    model_provider = args.model
    model = args.specific_model or ("gpt-4o" if model_provider == "openai" else "claude-3-5-sonnet")
    
    # Get default data dictionary path
    data_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "Data Dictionary", "mts.csv")
    
    # Check if the required API key is set based on model provider
    if model_provider.lower() == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            api_key = input("Enter your Anthropic API key: ")
            os.environ["ANTHROPIC_API_KEY"] = api_key
    else:  # OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            api_key = input("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
    
    print(f"Using model provider: {model_provider}")
    print(f"Using specific model: {model}")
    
    try:
        # Generate SQL from natural language
        result = natural_language_to_sql(
            query=query, 
            data_dictionary_path=data_dict_path, 
            model=model,
            model_provider=model_provider
        )
        
        # Print SQL and token usage
        print(f"\nQuery: {query}\n")
        print(f"SQL:\n{result['sql']}\n")
        print("Token Usage:")
        print(f"  Prompt tokens: {result['prompt_tokens']}")
        print(f"  Completion tokens: {result['completion_tokens']}")
        print(f"  Total tokens: {result['total_tokens']}")
        print(f"\nExecution time: {result.get('execution_time_ms', 0):.2f} ms")
        
    except Exception as e:
        print(f"Error: {str(e)}")
