"""
Gemini LLM Query Generator - A module to convert natural language to SQL using Google Gemini (1.5 Flash)
"""
import os
import pandas as pd
import datetime
from typing import Dict, List, Any, Optional
from token_logger import TokenLogger

# Import Google GenerativeAI (Gemini) SDK
gemini_available = False
genai_with_token_counter = False
try:
    import google.generativeai as genai
    gemini_available = True
    try:
        # Check if token counter is available
        from google.genai.types import HttpOptions
        genai_with_token_counter = True
    except ImportError:
        print("Google GenerativeAI SDK installed but doesn't support token counting")
except ImportError:
    print("Google GenerativeAI SDK not found. Please install with: pip install google-generativeai")

# Reuse these functions from the OpenAI implementation
from llm_query_generator import load_data_dictionary, format_data_dictionary

def generate_sql_prompt(tables: List[Dict[str, Any]], query: str) -> str:
    """
    Generate system prompt for Gemini with table schema information
    """
    tables_context = ""
    for table in tables:
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
            if col['business_name'] != col['name']:
                tables_context += f" [Business Name: {col['business_name']}]"
            tables_context += "\n"
        tables_context += "\n"
    prompt = f"""You are an expert SQL query generator for Snowflake database.\n\nYour task is to convert natural language questions into valid SQL queries that can run on Snowflake.\nUse the following data dictionary to understand the database schema:\n\n{tables_context}\nWhen generating SQL:\n1. Use proper Snowflake SQL syntax with fully qualified table names including schema (e.g., SCHEMA.TABLE_NAME)\n2. Include appropriate JOINs based on the data relationships or column name similarities\n3. Format the SQL code clearly with proper indentation and aliases\n4. Only use tables and columns that exist in the provided schema\n5. Return ONLY the SQL code, without any comments, explanations, or markdown\n6. Limit results to 100 rows unless specified otherwise\n\nGenerate a SQL query for: {query}\n"""
    return prompt

def generate_sql_query_gemini(api_key: str, prompt: str, model: str = "models/gemini-1.5-flash-latest", query_text: str = "", log_tokens: bool = True) -> Dict[str, Any]:
    """
    Generate SQL query using Google Gemini API (OpenAI-compatible result dict)
    """
    if not gemini_available:
        return {
            "sql": "SELECT 'Gemini SDK not installed' AS error_message",
            "model": model,
            "error": "Gemini SDK not installed",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "success": False,
            "results": None,
            "row_count": 0,
            "error_execution": "Gemini SDK not installed",
            "execution_time_ms": 0
        }
    logger = TokenLogger() if log_tokens else None
    start_time = datetime.datetime.now()
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 1000
        }
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt, generation_config=generation_config)
        sql_query = "SELECT 'No SQL query was successfully extracted' AS error_message"
        if response and hasattr(response, 'text') and response.text:
            full_text = response.text.strip()
            if not full_text:
                return {
                    "sql": sql_query,
                    "model": model,
                    "error": "Empty response from Gemini",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "success": False,
                    "results": None,
                    "row_count": 0,
                    "error_execution": "Empty response from Gemini",
                    "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000
                }
            # Extract SQL from code blocks if present
            import re
            sql_candidate = full_text
            # Prefer to extract from code blocks, but fallback to raw text
            if "```" in full_text:
                sql_blocks = re.findall(r'```(?:sql|SQL)?([\s\S]*?)```', full_text)
                if sql_blocks and sql_blocks[0]:
                    sql_candidate = sql_blocks[0].strip()
                else:
                    code_blocks = re.findall(r'```([\s\S]*?)```', full_text)
                    if code_blocks and code_blocks[0]:
                        sql_candidate = code_blocks[0].strip()
            # Remove comments and markdown from SQL (lines starting with -- or #, and code block markers)
            sql_lines = [line for line in sql_candidate.splitlines() if not line.strip().startswith('--') and not line.strip().startswith('#') and not line.strip().startswith('```')]
            sql_query = '\n'.join(sql_lines).strip()
            # If nothing left, fallback to error
            if not sql_query or not sql_query.strip():
                sql_query = "SELECT 'Empty SQL query extracted' AS error_message"
        # Get token counts using the Gemini Count Tokens API if available
        execution_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if genai_with_token_counter:
            try:
                # Count tokens for the prompt
                client = genai.Client(http_options=HttpOptions(api_version="v1"))
                prompt_response = client.models.count_tokens(
                    model=model,
                    contents=prompt
                )
                prompt_tokens = getattr(prompt_response, 'total_tokens', 0)
                
                # Count tokens for the completion (the generated SQL)
                completion_response = client.models.count_tokens(
                    model=model,
                    contents=sql_query
                )
                completion_tokens = getattr(completion_response, 'total_tokens', 0)
                
                # Calculate total tokens
                total_tokens = prompt_tokens + completion_tokens
                print(f"Gemini token count - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            except Exception as token_err:
                print(f"Warning: Could not count tokens for Gemini: {str(token_err)}")
        else:
            # Estimate tokens if counter not available (approx 4 chars per token for Gemini)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(sql_query) // 4
            total_tokens = prompt_tokens + completion_tokens
            print(f"Gemini token ESTIMATE - Prompt: ~{prompt_tokens}, Completion: ~{completion_tokens}, Total: ~{total_tokens}")
            print("Note: This is an approximation. Install latest google-generativeai for accurate counts.")
            # Continue with estimated values if token counter isn't available
        
        return {
            "sql": sql_query,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "success": True,
            "results": None,
            "row_count": 0,
            "error_execution": None,
            "execution_time_ms": execution_time_ms
        }
    except Exception as e:
        execution_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
        return {
            "sql": f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}' AS error_message",
            "model": model,
            "error": str(e),
            "prompt_tokens": prompt_tokens if 'prompt_tokens' in locals() else 0,
            "completion_tokens": completion_tokens if 'completion_tokens' in locals() else 0,
            "total_tokens": total_tokens if 'total_tokens' in locals() else 0,
            "execution_time_ms": execution_time_ms
        }

def natural_language_to_sql_gemini(query: str, data_dictionary_path: Optional[str] = None, api_key: Optional[str] = None, model: str = "models/gemini-1.5-flash-latest", log_tokens: bool = True) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL using Gemini, matching OpenAI logic for data dictionary and prompt construction.
    If data_dictionary_path is not provided, use the default 'data_dictionary.csv' in the current directory.
    """
    import os
    import datetime
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "sql": "SELECT 'Gemini API key not provided' AS error_message",
            "model": model,
            "error": "Gemini API key not provided",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    # Set default data dictionary path if not provided
    if not data_dictionary_path:
        data_dictionary_path = "data_dictionary.csv"
    start_time = datetime.datetime.now()
    try:
        # Validate data dictionary path
        if not os.path.isfile(data_dictionary_path):
            raise FileNotFoundError(f"Data dictionary file not found: {data_dictionary_path}")
        df = load_data_dictionary(data_dictionary_path)
        if df.empty:
            raise ValueError("Data dictionary loaded but is empty. Please check your file.")
        tables = format_data_dictionary(df)
        prompt = generate_sql_prompt(tables, query)
        result = generate_sql_query_gemini(
            api_key=api_key,
            prompt=prompt,
            model=model,
            query_text=query,
            log_tokens=log_tokens
        )
        # Ensure 'sql' key always exists and is a string
        if 'sql' not in result or result['sql'] is None or not isinstance(result['sql'], str):
            result['sql'] = "SELECT 'No valid SQL query was generated' AS message"
        # Add additional metadata
        result.update({
            "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "data_dictionary": data_dictionary_path
        })
        return result
    except Exception as e:
        execution_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
        return {
            "sql": f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}' AS error_message",
            "model": model,
            "error": str(e),
            "query": query,
            "data_dictionary": data_dictionary_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "execution_time_ms": execution_time_ms
        }

if __name__ == "__main__":
    import argparse
    import os
    from pprint import pprint
    parser = argparse.ArgumentParser(description="Gemini LLM Query Generator - Convert natural language to SQL using Gemini")
    parser.add_argument("--query", "-q", type=str, default="Show me the top 5 stores with highest sales", help="Natural language query")
    parser.add_argument("--model", "-m", type=str, default="models/gemini-1.5-flash-latest", help="Gemini model name")
    parser.add_argument("--data-dictionary", "-d", type=str, default=None, help="Path to data dictionary CSV/Excel file")
    parser.add_argument("--no-log", action="store_true", help="Disable token logging")
    args = parser.parse_args()

    # Match OpenAI CLI: use default data dictionary path if not provided
    data_dict_path = args.data_dictionary
    if data_dict_path is None:
        data_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data Dictionary", "mts.csv")

    print(f"Using Gemini model: {args.model}")

    try:
        result = natural_language_to_sql_gemini(
            query=args.query,
            data_dictionary_path=data_dict_path,
            model=args.model,
            log_tokens=not args.no_log
        )
        print(f"\nQuery: {args.query}\n")
        sql = result.get('sql', '')
        # Print SQL in triple backticks if it looks like code
        if sql.strip().startswith("SELECT") or sql.strip().startswith("--") or "\n" in sql:
            print(f"SQL:\n```sql\n{sql}\n````)")
        else:
            print(f"SQL: {sql}")
        print("Token Usage:")
        print(f"  Prompt tokens: {result.get('prompt_tokens', 0)}")
        print(f"  Completion tokens: {result.get('completion_tokens', 0)}")
        print(f"  Total tokens: {result.get('total_tokens', 0)}")
        print(f"\nExecution time: {result.get('execution_time_ms', 0):.2f} ms")
        if result.get('error'):
            print(f"Error: {result.get('error')}")
    except Exception as e:
        print(f"Error: {str(e)}")
