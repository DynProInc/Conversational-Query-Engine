"""
Gemini LLM Query Generator - A module to convert natural language to SQL using Google Gemini (1.5 Flash)
"""
import os
import pandas as pd
import datetime
import json
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

def generate_sql_prompt(tables: List[Dict[str, Any]], query: str, limit_rows: int = 100, include_charts: bool = False) -> str:
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
    # Create chart instructions if needed
    chart_instructions = """

After generating the SQL query, you must also recommend appropriate chart types for visualizing the results. Follow these rules for chart recommendations:

1. Analyze the query structure to understand what data will be returned
2.For single numeric values KPI_CARD: 
   Display as minimal cards with bold label at top, large formatted number below, no icons, clean white background, centered text only.
3.For purely categorical data with no numeric measures, recommend a table visualization.
4.Recommend 1-3 appropriate chart types (bar, line, pie, scatter,KPI_CARD,MIX etc.) based on the query's structure
5. For each recommendation, provide:
   - chart_type: The type of chart (bar, line, pie, scatter,mix, etc.)
   - reasoning: Brief explanation of why this chart type is appropriate
   - priority: Importance ranking (1 = highest)
   - chart_config: Detailed configuration including:
     * title: Descriptive chart title
     * x_axis: Column to use for x-axis
     * y_axis: Column to use for y-axis
     * color_by: Column to use for segmentation/colors (if applicable)
     * aggregate_function: Any aggregation needed (SUM, AVG, etc.)
     * chart_library: Recommended visualization library (plotly)
     * additional_config: Other relevant settings like orientation, legend, etc.

6. Also provide 2-3 data insights that would be valuable to highlight

Your response must be a valid JSON object with the following structure:
{
  "sql": "YOUR SQL QUERY HERE",
  "chart_recommendations": [
    {
      "chart_type": "bar|pie|line|scatter|mix|kpi_card|table|etc",
      "reasoning": "Why this chart is appropriate",
      "priority": 1,
      "chart_config": {
        "title": "Chart title",
        "x_axis": "column_name",
        "y_axis": "column_name",
        "color_by": "column_name",
        "aggregate_function": "NONE|SUM|AVG|etc",
        "chart_library": "plotly",
        "additional_config": {
          "show_legend": true,
          "orientation": "vertical|horizontal"
        }
      }
    }
  ],
  }
""" if include_charts else """
Return ONLY the SQL code without any other text or explanations.
"""

    prompt = f"""You are an expert SQL query generator for Snowflake database.

Your task is to convert natural language questions into valid SQL queries that can run on Snowflake.
Use the following data dictionary to understand the database schema:

{tables_context}

When generating SQL:
1. Only generate SELECT queries.
2. Use proper Snowflake SQL syntax with fully qualified table names including schema (e.g., SCHEMA.TABLE_NAME)
3. For column selections:
   - Always list columns individually (never concatenate or combine columns till user not asking in prompt)
   - Use consistent column casing - uppercase for all column names
   - Format each column on a separate line with proper indentation
4. Include appropriate JOINs based only on the relationships defined in the schema metadata.
5. Only use tables and columns that exist in the provided schema.
6. For numeric values:
   - Use standard CAST() function for type conversions (e.g., CAST(field AS DECIMAL) or CAST(field AS NUMERIC))
   - When using GROUP BY, always apply aggregate functions (SUM, AVG, etc.) to non-grouped numeric fields
   - Example: SUM(CAST(SALES_AMOUNT AS NUMERIC)) AS TOTAL_SALES
   - ALWAYS use NULLIF() for divisions to prevent division by zero errors:
     * For percentage calculations: (new_value - old_value) / NULLIF(old_value, 0) * 100
     * For ratios: numerator / NULLIF(denominator, 0)
   - For sensitive calculations that must return specific values on zero division:
     * Use CASE: CASE WHEN denominator = 0 THEN NULL ELSE numerator/denominator END
7. Format results with consistent column naming:
   - For aggregations, use uppercase names (e.g., SUM(sales) AS TOTAL_SALES)
   - For regular columns, maintain original casing
8. CRITICAL: Follow these row limit rules EXACTLY:
   a. If the user explicitly specifies a number in their query (e.g., "top 5", "first 10"), use EXACTLY that number in the LIMIT clause
   b. Otherwise, limit results to {limit_rows} rows
   c. NEVER override a user-specified limit with a different number
9. SUPERLATIVE QUERY HANDLING:
   a. CRITICAL: For PLURAL nouns in queries like "which sales representatives sold most" - NEVER add LIMIT 1, return MULTIPLE results
   b. For SINGULAR nouns in queries like "which sales rep sold most" - ALWAYS add `ORDER BY [relevant_metric] DESC LIMIT 1`
   c. Explicitly check if words like "representatives", "products", "customers" (plural) are used
   d. Examples: "which sales rep sold most" → ONE result (LIMIT 1), "which sales representatives sold most" → MULTIPLE results (NO LIMIT 1)

Generate a SQL query for: {query}{chart_instructions}
"""
    return prompt

def generate_sql_query_gemini(api_key: str, prompt: str, model: str = "models/gemini-1.5-flash-latest", query_text: str = "", log_tokens: bool = True, include_charts: bool = False) -> Dict[str, Any]:
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
            "max_output_tokens": 4000
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
                    "chart_recommendations": None,
                    "chart_error": None,
                    "error_execution": "Empty response from Gemini",
                    "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Extract SQL based on whether charts are requested
            import re
            import json
            
            # Initialize chart_recommendations and chart_error
            chart_recommendations = None
            chart_error = None
            
            # Print debug info about the raw response
            print("\n==== GEMINI RAW RESPONSE ====")
            print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
            print("=============================")
            
            if include_charts:
                # Try to parse the JSON response with SQL and chart recommendations
                try:
                    # Remove code blocks if present
                    cleaned_text = full_text
                    if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[7:-3].strip()
                    elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[3:-3].strip()
                    
                    # Look for JSON object markers
                    json_start = cleaned_text.find('{')
                    json_end = cleaned_text.rfind('}')
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = cleaned_text[json_start:json_end+1]
                        print("Found JSON content between { and }")
                    else:
                        json_content = cleaned_text
                        print("No JSON markers found, using cleaned text")
                    
                    # Parse the JSON response
                    json_data = json.loads(json_content)
                    print(f"Successfully parsed JSON with keys: {list(json_data.keys())}")
                    
                    # Extract SQL query and chart recommendations
                    sql_query = json_data.get("sql", "")
                    chart_recommendations = json_data.get("chart_recommendations", [])
                    
                    # If chart_recommendations exists but is empty, try looking for other formats
                    if not chart_recommendations and "charts" in json_data:
                        chart_recommendations = json_data.get("charts", [])
                    
                    print(f"Found chart recommendations: {len(chart_recommendations) if chart_recommendations else 0} items")
                    
                except json.JSONDecodeError as e:
                    # Fallback if JSON parsing fails
                    print(f"Error parsing JSON response from Gemini: {str(e)}")
                    
                    # Try to find chart recommendations in a non-JSON format
                    # Many LLMs mention chart recommendations in prose even if JSON parsing fails
                    chart_section_match = re.search(r'(?:chart recommendations|recommended charts|chart suggestions)(?:[\s\S]*?)(?:1\.|\*|-)([\s\S]*?)(?:$|\n\n)', full_text, re.IGNORECASE)
                    
                    if chart_section_match:
                        chart_text = chart_section_match.group(1).strip()
                        print(f"Found potential chart text section: {chart_text[:100]}...")
                        
                        # Try to identify chart types from the text
                        chart_types = []
                        for chart_type in ['bar', 'line', 'pie', 'scatter', 'area', 'mixed']:
                            if chart_type in chart_text.lower():
                                chart_types.append(chart_type)
                        
                        if chart_types:
                            print(f"Identified chart types from text: {chart_types}")
                            # Create basic chart recommendations from the identified types
                            chart_recommendations = []
                            for i, chart_type in enumerate(chart_types[:3]):  # Limit to 3
                                chart_recommendations.append({
                                    "chart_type": chart_type,
                                    "reasoning": f"Auto-detected {chart_type} chart from Gemini response",
                                    "priority": i+1,
                                    "chart_config": {
                                        "title": f"Auto-generated {chart_type.capitalize()} Chart",
                                        "chart_library": "plotly"
                                    }
                                })
                    
                    # First, try to extract just the SQL query from JSON if possible
                    # This handles cases where the response is malformed JSON but contains a proper SQL field
                    # Try to match SQL inside JSON object with multiline support
                    sql_json_match = re.search(r'"sql"\s*:\s*"(.+?)(?=",|"})', cleaned_text, re.DOTALL)
                    if sql_json_match:
                        sql_candidate = sql_json_match.group(1)
                        print(f"Extracted SQL from partial JSON: {sql_candidate[:50]}...")
                    else:
                        # Extract SQL using the regular approach
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
                    
                    # Remove comments and markdown from SQL
                    sql_lines = [line for line in sql_candidate.splitlines() if not line.strip().startswith('--') and not line.strip().startswith('#') and not line.strip().startswith('```')]
                    sql_query = '\n'.join(sql_lines).strip()
                    
                    # Set chart error if we couldn't extract any charts
                    if not chart_recommendations:
                        chart_error = "Cannot generate charts: error in JSON parsing"
            else:
                # Just extract SQL without chart parsing
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
                
                # Remove json prefix, comments and markdown from SQL
                sql_candidate = sql_candidate.strip()
                # Handle 'json' prefix if present
                if sql_candidate.startswith('json'):
                    sql_candidate = sql_candidate[4:].strip()
                    
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
                # Fallback to estimate
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(sql_query) // 4
                total_tokens = prompt_tokens + completion_tokens
                print(f"Gemini token ESTIMATE - Prompt: ~{prompt_tokens}, Completion: ~{completion_tokens}, Total: ~{total_tokens}")
        else:
            # Estimate tokens if counter not available (approx 4 chars per token for Gemini)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(sql_query) // 4
            total_tokens = prompt_tokens + completion_tokens
            print(f"Gemini token ESTIMATE - Prompt: ~{prompt_tokens}, Completion: ~{completion_tokens}, Total: ~{total_tokens}")
            print("Note: This is an approximation. Install latest google-generativeai for accurate counts.")
            # Continue with estimated values if token counter isn't available
        
        # Print chart recommendations debug info before returning
        if chart_recommendations:
            print(f"\nReturning {len(chart_recommendations)} chart recommendations:")
            for i, chart in enumerate(chart_recommendations):
                print(f"Chart {i+1}: {chart.get('chart_type')}")
        else:
            print("\nNo chart recommendations to return")
        
        # Return the result dictionary with chart recommendations
        return {
            "sql": sql_query,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "chart_recommendations": chart_recommendations,
            "chart_error": chart_error,
            "execution_time_ms": execution_time_ms
        }
    except Exception as e:
        execution_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
        return {
            "sql": f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}'  AS error_message",
            "model": model,
            "error": str(e),
            "chart_recommendations": None,
            "chart_error": f"Chart generation failed: {str(e)}",
            "prompt_tokens": prompt_tokens if 'prompt_tokens' in locals() else 0,
            "completion_tokens": completion_tokens if 'completion_tokens' in locals() else 0,
            "total_tokens": total_tokens if 'total_tokens' in locals() else 0,
            "execution_time_ms": execution_time_ms
        }

def natural_language_to_sql_gemini(query: str, data_dictionary_path: Optional[str] = None, api_key: Optional[str] = None, model: str = None, log_tokens: bool = True, limit_rows: int = 100, include_charts: bool = False) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL using Gemini, matching OpenAI logic for data dictionary and prompt construction.
    If data_dictionary_path is not provided, use the default 'data_dictionary.csv' in the current directory.
    """
    import os
    import datetime
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
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
        prompt = generate_sql_prompt(tables, query, limit_rows=limit_rows, include_charts=include_charts)
        result = generate_sql_query_gemini(
            api_key=api_key,
            prompt=prompt,
            model=model,
            query_text=query,
            log_tokens=log_tokens,
            include_charts=include_charts
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
            "chart_recommendations": None,
            "chart_error": str(e),
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
