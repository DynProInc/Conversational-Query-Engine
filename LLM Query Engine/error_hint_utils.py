"""
error_hint_utils.py
Provides user-friendly hints and error message cleanup for common SQL/database errors.
"""
import os
import re
import time
import random
from typing import Optional, List, Dict, Any

# Check if OpenAI is installed
try:
    import openai
except ImportError:
    openai = None
    print("OpenAI module not found. Install with: pip install openai")
    
# Check if Anthropic is installed
try:
    import anthropic
except ImportError:
    anthropic = None
    print("Anthropic module not found. Install with: pip install anthropic")
    
# Check if Google GenerativeAI is installed
try:
    import google.generativeai as genai
    genai_available = True
except ImportError:
    genai = None
    genai_available = False
    print("Google GenerativeAI not found. Install with: pip install google-generativeai")


def identify_error_type(error_message: str) -> str:
    """
    Identify the type of error based on the error message.
    
    Returns:
        String indicating error type: "api_connection", "api_authentication", "sql", "unknown"
    """
    if not error_message:
        return "unknown"
    
    # API Connection errors
    if any(term in error_message.lower() for term in [
        "api key", "invalid_api_key", "authentication", "unauthorized", "401", 
        "403", "connection error", "timeout", "rate limit", "capacity",
        "overloaded", "busy", "unavailable", "down", "openai", "anthropic", 
        "gemini", "api", "http"
    ]):
        return "api_connection"
    
    # SQL errors (Snowflake specific patterns)
    if any(term in error_message.lower() for term in [
        "sql", "syntax error", "invalid identifier", "object does not exist",
        "schema", "table", "column", "database", "warehouse", "snowflake"
    ]):
        return "sql"
    
    return "unknown"


def clean_error_message(error_message: str, error_type: str = None) -> str:
    """
    Clean up raw error messages by removing error codes, line numbers, and positions.
    Returns a simplified, user-friendly error message.
    
    Args:
        error_message: The raw error message
        error_type: The type of error ("api_connection", "sql", "unknown")
    """
    if not error_message:
        return ""
    
    # Determine error type if not provided
    if not error_type:
        error_type = identify_error_type(error_message)
    
    # For API connection errors, return a generic message
    if error_type == "api_connection":
        return "Unable to connect to AI service. Please check your connection and try again."
    
    # For SQL errors, clean up the message
    if error_type == "sql":
        # Remove error codes like 000904 (42000)
        error_message = re.sub(r'\d{6}\s*\([^)]*\):\s*', '', error_message)
        
        # Remove line and position information
        error_message = re.sub(r'error\s+line\s+\d+\s+at\s+position\s+\d+\s*\n*', '', error_message)
        
        # Clean up extra whitespace and newlines
        error_message = re.sub(r'\s+', ' ', error_message).strip()
    
    return error_message


def generate_llm_error_hint(error_message: str, sql_query: str = None, preferred_model: str = None, error_type: str = None) -> Optional[str]:
    """
    Use an LLM to generate a concise, user-friendly error hint for SQL errors.
    Will try available LLM providers in sequence: OpenAI, Anthropic Claude, Google Gemini.
    
    Args:
        error_message: The error message
        sql_query: The original SQL query that caused the error (optional)
        preferred_model: The preferred model to use first
        error_type: The type of error ("api_connection", "sql", "unknown")
        
    Returns:
        A one-line user-friendly error hint, or None if all LLM calls fail
    """
    # Don't use LLM for API connection errors or empty error messages
    if error_type == "api_connection" or not error_message:
        return None
        
    # Only use LLM for SQL errors
    if error_type != "sql":
        return None
    
    # Clean the error message first
    clean_error = clean_error_message(error_message, error_type)
    
    # Build the prompt - same for all providers
    prompt = f"""I need a concise one-line hint to help a user fix this SQL error. 
    
    Original Error: {clean_error}
    """
    
    if sql_query:
        prompt += f"\n\nSQL Query that caused the error: {sql_query}"
        
    prompt += """\n\nIMPORTANT GUIDELINES:
1. Provide ONLY a one-line actionable hint focusing on what the user should do to fix this.
2. Make your hint GENERIC - do NOT mention specific table names, schema names, or database objects from the error.
3. Do NOT include the exact column names or table paths from the error.
4. Focus on the general concept of what's wrong rather than specifics.
5. Suggest general approaches instead of specific corrections.

For example:
- BAD: "Check if 'BRAND_NAME' exists in table 'REPORTING_UAT.GOLD_SALES.V_SMM_ITEM_DAILY_SALE'"
- GOOD: "Try using a different column name as this one doesn't appear to exist in the table"

Provide only your generic hint:"""

    # Determine which model/provider to use based on preferred_model
    provider = None
    model = None
    
    # If a specific model is requested, use only that model
    if preferred_model:
        # Determine the provider based on the model name
        if 'gpt' in preferred_model.lower() or 'openai' in preferred_model.lower():
            provider = 'openai'
            model = preferred_model
        elif 'claude' in preferred_model.lower() or 'anthropic' in preferred_model.lower():
            provider = 'claude'
            model = preferred_model
        elif 'gemini' in preferred_model.lower():
            provider = 'gemini'
            model = preferred_model
    
    # If no specific model is identified, use default OpenAI
    if not provider:
        if openai is not None:
            provider = 'openai'
            model = os.environ.get('OPENAI_MODEL', 'gpt-4o')
    
    # Use only the selected model/provider (no fallbacks)
    try:
        if provider == 'openai' and openai is not None:
            return generate_openai_hint(prompt, model)
        elif provider == 'claude' and anthropic is not None:
            return generate_claude_hint(prompt, model)
        elif provider == 'gemini' and genai is not None:
            return generate_gemini_hint(prompt, model)
        else:
            print(f"Provider {provider} with model {model} is not available")
            return None
    except Exception as e:
        print(f"Error with {provider} ({model}): {str(e)}")
        return None
    return None


def generate_openai_hint(prompt: str, model: str = "gpt-4o") -> Optional[str]:
    """Generate error hint using OpenAI"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or openai is None:
        return None
        
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        # Extract hint
        hint = response.choices[0].message.content.strip()
        
        # Remove any extra formatting like quotation marks, numbers, etc.
        hint = re.sub(r'^["\'\d\.\-\*]\s*', '', hint)
        hint = re.sub(r'["\'.]$', '', hint)
        
        return hint
        
    except Exception as e:
        print(f"OpenAI error: {str(e)}")
        return None


def generate_claude_hint(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> Optional[str]:
    """Generate error hint using Anthropic Claude"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or anthropic is None:
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract hint
        hint = response.content[0].text.strip()
        
        # Remove any extra formatting like quotation marks, numbers, etc.
        hint = re.sub(r'^["\'\d\.\-\*]\s*', '', hint)
        hint = re.sub(r'["\'.]$', '', hint)
        
        return hint
        
    except Exception as e:
        print(f"Claude error: {str(e)}")
        return None


def generate_gemini_hint(prompt: str, model: str = "models/gemini-1.5-flash-latest") -> Optional[str]:
    """Generate error hint using Google Gemini"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 100
        }
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt, generation_config=generation_config)
        
        if response and hasattr(response, 'text') and response.text:
            hint = response.text.strip()
            
            # Remove any extra formatting like quotation marks, numbers, etc.
            hint = re.sub(r'^["\'\d\.\-\*]\s*', '', hint)
            hint = re.sub(r'["\'.]$', '', hint)
            
            return hint
        return None
        
    except Exception as e:
        print(f"Gemini error: {str(e)}")
        return None


def get_success_hint(sql_query: str = None, model: str = None) -> str:
    """
    Generate a helpful hint for successful queries based on the executed SQL.
    Uses the same LLM provider as the one that generated the original SQL query.
    
    Args:
        sql_query: The executed SQL query
        model: The model used to generate the SQL
    """
    if not sql_query:
        return None
        
    # Create prompt for success hint
    prompt = f"""You are an expert SQL assistant. I've successfully executed the following SQL query:

```sql
{sql_query}
```

Provide a concise, one-line hint about the query. Do not explain what the query does, but instead offer a useful insight about the query structure, performance, or a tip that might help the user understand SQL better. Keep it generic without mentioning specific table or column names.

Your response should be plain text, without quotes, and no more than 15 words."""

    # Determine provider based on model name
    provider = None
    if model:
        if 'gpt' in model.lower() or 'openai' in model.lower():
            provider = 'openai'
        elif 'claude' in model.lower() or 'anthropic' in model.lower():
            provider = 'claude'
        elif 'gemini' in model.lower():
            provider = 'gemini'
    
    # Use the same provider that generated the SQL
    try:
        if provider == 'openai' and openai is not None:
            hint = generate_openai_hint(prompt, model)
            if hint:
                return hint
        elif provider == 'claude' and anthropic is not None:
            hint = generate_claude_hint(prompt, model)
            if hint:
                return hint
        elif provider == 'gemini' and genai is not None:
            hint = generate_gemini_hint(prompt, model)
            if hint:
                return hint
    except Exception as e:
        print(f"Error generating success hint: {str(e)}")
    
    # Fallback static hints
    fallback_hints = [
        "Using DISTINCT can help eliminate duplicate rows in results.",
        "Consider adding indexes to improve query performance.",
        "LIMIT clauses are useful to restrict result size for testing.",
        "JOINs connect related data across multiple tables."
    ]
    return random.choice(fallback_hints)


def get_user_friendly_hint(error_message: str, sql_query: str = None, model: str = None) -> str:
    """
    Given a raw error message, return a user-friendly hint focused on how to fix the issue.
    Uses LLM if available, falls back to rule-based hints.
    
    Args:
        error_message: The error message
        sql_query: The original SQL query that caused the error (optional)
        model: The preferred LLM model to use (optional)
    """
    if not error_message:
        return ""
        
    # Identify the error type
    error_type = identify_error_type(error_message)
    
    # Special handling for API connection errors
    if error_type == "api_connection":
        return "API connection error. Please verify your connection settings."
        
    # Try LLM-based hint first for SQL errors
    llm_hint = generate_llm_error_hint(error_message, sql_query, preferred_model=model, error_type=error_type)
    if llm_hint:
        return llm_hint
        
    # Fall back to rule-based hints if LLM fails
    
    # Invalid identifier (column or table)
    match = re.search(r"invalid identifier '([^']+)'", error_message, re.IGNORECASE)
    if match:
        return "Verify the column name for accuracy or check if it exists in the table schema."
    
    # Table does not exist
    match = re.search(r"Object does not exist, or operation cannot be performed: (.+)", error_message, re.IGNORECASE)
    if match:
        return "The specified table could not be found. Check available tables in the data dictionary."
    
    # Syntax error
    if "syntax error" in error_message.lower():
        return "Check the SQL syntax and try again with correct formatting."
    
    # Permission denied
    if "permission denied" in error_message.lower() or "not authorized" in error_message.lower():
        return "Access denied. Contact your administrator for required permissions."
    
    # Connection error
    if "could not connect" in error_message.lower() or "connection failed" in error_message.lower():
        return "Database connection issue. Please try again later."
    
    # Default fallback
    return "Please check your query and try again with different parameters."
