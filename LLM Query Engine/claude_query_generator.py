"""Claude LLM Query Generator - A module to convert natural language to SQL using Anthropic Claude
"""
import os
import pandas as pd
import datetime
import json
import re  # Ensure re is imported at the top level for use throughout the module
import dotenv
import traceback
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


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str, limit_rows: int = 100, include_charts: bool = False) -> str:
    """
    Generate system prompt for Claude with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        limit_rows: Default row limit if not specified in query
        
    Returns:
        Formatted system prompt for Claude
    """
    # Extract limit from query if present
    extracted_limit = extract_limit_from_query(query)
    
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
    
    # Create prompt template for Claude - matching OpenAI's detailed chart instructions
    chart_instructions = """

After generating the SQL query, recommend appropriate chart types for visualizing the results. Follow these rules:

1. Analyze the query to determine if it returns numeric columns or only categorical data
2. IMPORTANT: Provide EXACTLY the top 2 chart recommendations that best visualize the data, ordered by priority.
3. Recommend charts that match the data structure based on these rules:

**CRITICAL: Time-Based Data Handling for Charts:**
- For time-based data spanning multiple years, ALWAYS use the column with full date information (YYYY-MM format) as the x-axis
- Multi-year data: Use YYYY-MM column for x-axis, not month names

1. Bar charts: For comparing categorical data or time periods. Suitable for up to ~15 categories.
   - Required: One categorical column and at least one numeric measure
   - Best for: Comparisons across categories

2. Line charts: For showing trends over a continuous range (usually time).
   - Required: One continuous axis (usually time) and at least one numeric measure
   - Best for: Trends over time, continuous data
   - IMPORTANT: For time series across multiple years, ALWAYS use the column containing year+month format

3. Pie charts: For showing composition or parts of a whole (limited to 7 or fewer categories).
   - Required: One categorical column with 7 or fewer categories and one numeric measure
   - Best for: Part-to-whole relationships, proportions

4. Scatter plots: For showing relationships between two numeric variables.
   - Required: Two numeric measures
   - Best for: Correlation analysis, distribution patterns

5. Area charts: For showing cumulative totals over time or categories.
   - Required: One continuous axis (usually time) and at least one numeric measure
   - Best for: Cumulative trends, stacked compositions over time
   - IMPORTANT: For time series across multiple years, ALWAYS use the column containing year+month format

6. Mixed/Combo charts: For comparing different but related metrics with appropriate visualizations.
   - Required: One shared axis and multiple measures that may have different scales
   - Best for: Multi-measure comparisons with different scales

For purely categorical data with no numeric measures, recommend a table visualization.

Scale Detection and Secondary Axis:
- When two measures have significantly different scales (100x+ difference), use a secondary Y-axis
- For mixed charts, specify which series should use the secondary axis
- Include scale_detection: true and scale_threshold: 2 in the additional_config

Your response MUST be a valid JSON object with the following structure:
{
  "sql": "YOUR SQL QUERY HERE",
  "chart_recommendations": [
    {
      "chart_type": "bar|pie|line|scatter|histogram|area|combo|mixed|table|suggestion",
      "reasoning": "Why this chart is appropriate",
      "priority": 1,
      "chart_config": {
        "title": "Chart title",
        "x_axis": "column_name",
        "y_axis": ["column_name1", "column_name2"],
        "color_by": "column_name",
        "aggregate_function": "NONE|SUM|AVG|etc",
        "chart_library": "plotly",
        "additional_config": {
          "show_legend": true,
          "orientation": "vertical|horizontal",
          "use_secondary_axis": true,
          "secondary_axis_columns": ["column_name2"],
          "scale_detection": true,
          "scale_threshold": 2
        }
      }
    }
  ]
}

Example with scale detection:
{
  "sql": "SELECT MONTH_NAME, SUM(SALES) AS TOTAL_SALES, COUNT(ORDER_ID) AS ORDER_COUNT FROM ORDERS GROUP BY MONTH_NAME ORDER BY MONTH_NUMBER;",
  "chart_recommendations": [{
    "chart_type": "mixed",
    "reasoning": "TOTAL_SALES and ORDER_COUNT have different scales",
    "priority": 1,
    "chart_config": {
      "title": "Monthly Sales and Orders",
      "x_axis": "MONTH_NAME",
      "series": [
        { "column": "TOTAL_SALES", "type": "bar", "axis": "primary" },
        { "column": "ORDER_COUNT", "type": "line", "axis": "secondary" }
      ],
      "additional_config": {
        "use_secondary_axis": true,
        "secondary_axis_columns": ["ORDER_COUNT"],
        "scale_detection": true,
        "scale_threshold": 2
      }
    }
  }]
}

IMPORTANT: DO NOT include any explanations, code blocks, or markdown formatting outside the JSON. Your entire response must be valid JSON that can be parsed directly.
""" if include_charts else """
Return ONLY the SQL code without any other text or explanations.
"""

    prompt = f"""You are an expert SQL query generator for Snowflake database.

Your task is to convert natural language questions into valid SQL queries that can run on Snowflake.
Use the following data dictionary to understand the database schema:

{tables_context}{limit_instruction}

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
8. Add helpful SQL comments to explain complex parts of the query
9. CRITICAL: Follow these row limit rules EXACTLY in this order of priority:
     a. If the user explicitly specifies a number in their query (e.g., "top 5", "first 10"), use EXACTLY that number in the LIMIT clause
     b. If no specific number is given but the query mentions "top" or "first" without a number, use exactly {limit_rows} as the LIMIT
     c. For all other queries, limit results to {limit_rows} rows
     d. NEVER use default values like 5 or 10 when the limit_rows parameter of {limit_rows} is provided
10. If the query is unclear, include this comment: -- Please clarify: [specific aspect]
11. EXTREMELY IMPORTANT: SQL formatting rules:
    - Always include spaces between SQL keywords and identifiers (e.g., "SELECT column FROM table" not "SELECTcolumn FROMtable")
    - Use proper spacing around operators (e.g., "column = value" not "column=value")
    - Always separate keywords with spaces (e.g., "GROUP BY" not "GROUPBY")
    - For column aliases, always put a space after the AS keyword (e.g., "SUM(value) AS TOTAL" not "SUM(value) ASTOTAL")
    - Always add a space after each comma in lists (e.g., "col1, col2, col3" not "col1,col2,col3")
12. CRITICAL: Time-based data handling:
    - For time-based queries, ALWAYS include both YEAR and MONTH components for proper chronological ordering
    - When returning monthly data that spans multiple years, use YYYY-MM format (e.g., "2024-05") as the primary x-axis value
    - For chart recommendations with time data that spans multiple years, ALWAYS use the full date format column (like TRANSACTION_YEAR_MONTH) as the x-axis, NOT just the month name
    - If month names are included (Jan, Feb, etc.), they should be supplementary and not the primary x-axis for charts

Generate a SQL query for: {query}{chart_instructions}
"""
    return prompt


def generate_sql_query_claude(api_key: str, prompt: str, model: str = "claude-3-5-sonnet-20241022", 
                         query_text: str = "", log_tokens: bool = True, include_charts: bool = False) -> Dict[str, Any]:
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
        
        # Configure Claude to generate valid JSON when charts are requested
        if include_charts:
            response = client.messages.create(
                model=model,
                max_tokens=3000,  # Increased token limit for chart recommendations
                temperature=0.1,
                system="You are an expert SQL generator that returns valid JSON responses with SQL and chart recommendations. Always ensure your responses are valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        else:
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        
        
        
        # (DEBUG prints removed)
        # The following blocks are intentionally left empty after debug removal.
        pass

        # Extract SQL query from response - new simplified approach
        # Start with a valid default SQL that will work as a fallback
        sql_query = "SELECT 'No SQL query was successfully extracted' AS error_message"
        
        # Only proceed if we have a valid response with content
        if response and hasattr(response, 'content') and response.content:
            try:
                # Get the text content directly from Claude's response (similar to OpenAI pattern)
                full_text = ""
                
                # Safety check for text attribute
                if hasattr(response.content[0], 'text'):
                    if response.content[0].text is not None:
                        full_text = response.content[0].text
                        if not isinstance(full_text, str):
                            full_text = str(full_text)
                        full_text = full_text.strip()
                
                # Exit early if we have no text to parse
                if not full_text:
                    print("DEBUG: Empty text content from Claude response")
                    return {
                        "sql": sql_query,
                        "model": model,
                        "error": "Empty response from Claude",
                        "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                        "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                        "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0
                    }
                
                # Extract SQL from code blocks if present
                if "```" in full_text:
                    # First try to find SQL-specific code blocks
                    sql_blocks = re.findall(r'```(?:sql|SQL)?([\s\S]*?)```', full_text)
                    
                    if sql_blocks and sql_blocks[0]:
                        # Successfully found an SQL code block
                        extracted_sql = sql_blocks[0].strip()
                        if extracted_sql:  # Only update if we got something meaningful
                            sql_query = extracted_sql
                    else:
                        # If no SQL-specific blocks, try any code blocks
                        code_blocks = re.findall(r'```([\s\S]*?)```', full_text)
                        if code_blocks and code_blocks[0]:
                            extracted_code = code_blocks[0].strip()
                            if extracted_code:  # Only update if we got something meaningful
                                sql_query = extracted_code
                
                # For chart recommendations, try to parse JSON - exactly matching OpenAI implementation
                chart_recommendations = []
                chart_error = None
                
                # Flag to track if we found valid JSON to prevent the entire JSON response being used as SQL
                json_parsing_success = False
                
                if include_charts:
                    try:
                        # Remove code blocks if present
                        original_text = full_text
                        if full_text.startswith("```json") and full_text.endswith("```"):
                            full_text = full_text[7:-3].strip()
                        elif full_text.startswith("```") and full_text.endswith("```"):
                            full_text = full_text[3:-3].strip()
                        
                        # First attempt: Try to clean invalid control characters and parse
                        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', full_text)
                        try:
                            # Parse the cleaned JSON response
                            json_data = json.loads(cleaned_text)
                            json_parsing_success = True
                            print("Successfully parsed JSON after cleaning control characters")
                        except json.JSONDecodeError:
                            # If that still fails, try with the original but stricter cleanup
                            # Replace all non-printable characters
                            ultra_clean = ''.join(c for c in full_text if c.isprintable())
                            json_data = json.loads(ultra_clean)
                            json_parsing_success = True
                            print("Successfully parsed JSON after strict character cleaning")
                        
                        # Extract SQL query and chart recommendations
                        if "sql" in json_data:
                            sql_query = json_data.get("sql", "")
                            # Ensure we're not sending JSON as SQL
                            if sql_query.startswith("{") and sql_query.endswith("}"):
                                try:
                                    # Check if SQL is actually JSON
                                    test_json = json.loads(sql_query)
                                    if "sql" in test_json:
                                        # Extract the actual SQL from nested JSON
                                        sql_query = test_json.get("sql", "")
                                except:
                                    # If it's not valid JSON, keep it as is
                                    pass
                        if "chart_recommendations" in json_data:
                            chart_recommendations = json_data.get("chart_recommendations", [])
                        if "chart_error" in json_data:
                            chart_error = json_data.get("chart_error")
                        
                    except json.JSONDecodeError as e:
                        # Fallback if JSON parsing fails
                        print(f"Error parsing JSON response: {str(e)}")
                        
                        # Never use JSON-looking text as SQL
                        if full_text.strip().startswith('{') and full_text.strip().endswith('}'):
                            # This looks like JSON but failed to parse - try to extract chart recommendations
                            # Look for SQL statement pattern within the text
                            sql_pattern = re.search(r'"sql"\s*:\s*"([^"]+)"', full_text)
                            if sql_pattern:
                                sql_query = sql_pattern.group(1)
                                # Unescape any escaped quotes
                                sql_query = sql_query.replace("\\\"", '"')
                            else:
                                # Try to find a SQL statement by looking for SELECT or WITH
                                sql_match = re.search(r'(?:WITH|SELECT)\s+[\s\S]*?(?:;|$)', full_text, re.IGNORECASE)
                                if sql_match:
                                    sql_query = sql_match.group(0)
                            
                            # Try to extract chart recommendations from the JSON-like text
                            try:
                                # First attempt - try to extract the entire chart_recommendations array
                                chart_match = re.search(r'"chart_recommendations"\s*:\s*(\[.*?\]\s*}?)(?:,|\s*}|$)', full_text, re.DOTALL)
                                if chart_match:
                                    chart_json = chart_match.group(1)
                                    
                                    # Handle case where we captured too much
                                    if chart_json.count('[') != chart_json.count(']'):
                                        # Find a balanced closing bracket
                                        depth = 0
                                        end_pos = 0
                                        for i, c in enumerate(chart_json):
                                            if c == '[':
                                                depth += 1
                                            elif c == ']':
                                                depth -= 1
                                                if depth == 0:
                                                    end_pos = i
                                                    break
                                        if end_pos > 0:
                                            chart_json = chart_json[:end_pos+1]
                                    
                                    # Clean the extracted JSON before parsing
                                    chart_json = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', chart_json)
                                    chart_json = ''.join(c for c in chart_json if c.isprintable())
                                    
                                    # Fix common JSON formatting issues
                                    chart_json = chart_json.replace('\"', '"')  # Fix escaped quotes
                                    chart_json = chart_json.replace('\\"', '"')  # Fix double-escaped quotes
                                    chart_json = re.sub(r',\s*}', '}', chart_json)  # Fix trailing commas
                                    chart_json = re.sub(r',\s*]', ']', chart_json)  # Fix trailing commas in arrays
                                    
                                    # Parse the chart recommendations
                                    try:
                                        chart_recommendations = json.loads(chart_json)
                                        print("Successfully extracted chart recommendations from malformed JSON")
                                    except json.JSONDecodeError:
                                        # If we can't parse the array, try extracting individual chart objects
                                        chart_objects = re.findall(r'\{[^\{\}]*"chart_type"[^\{\}]*\}', full_text)
                                        if chart_objects:
                                            chart_recommendations = []
                                            for obj in chart_objects:
                                                try:
                                                    clean_obj = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', obj)
                                                    clean_obj = ''.join(c for c in clean_obj if c.isprintable())
                                                    chart_obj = json.loads(clean_obj)
                                                    if 'chart_type' in chart_obj:
                                                        chart_recommendations.append(chart_obj)
                                                except:
                                                    continue
                                            print(f"Extracted {len(chart_recommendations)} individual chart recommendations")
                                else:
                                    # Last resort - try to find complete chart objects
                                    chart_objects = re.findall(r'\{[^\{\}]*"chart_type"[^\{\}]*\}', full_text)
                                    if chart_objects:
                                        chart_recommendations = []
                                        for obj in chart_objects:
                                            try:
                                                clean_obj = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', obj)
                                                clean_obj = ''.join(c for c in clean_obj if c.isprintable())
                                                chart_obj = json.loads(clean_obj)
                                                if 'chart_type' in chart_obj:
                                                    chart_recommendations.append(chart_obj)
                                            except:
                                                continue
                                        print(f"Extracted {len(chart_recommendations)} individual chart recommendations")
                            except Exception as chart_err:
                                print(f"Failed to extract chart recommendations: {str(chart_err)}")
                                # Try one last approach - look for chart_type patterns
                                try:
                                    chart_types = re.findall(r'"chart_type"\s*:\s*"([^"]+)"', full_text)
                                    if chart_types:
                                        chart_recommendations = []
                                        for chart_type in chart_types:
                                            chart_recommendations.append({
                                                "chart_type": chart_type,
                                                "reasoning": "Recovered from malformed JSON",
                                                "priority": len(chart_recommendations) + 1
                                            })
                                        print(f"Recovered {len(chart_recommendations)} basic chart recommendations")
                                    else:
                                        chart_recommendations = []
                                        chart_error = "Cannot generate charts: error in JSON parsing"
                                except:
                                    chart_recommendations = []
                                    chart_error = "Cannot generate charts: error in JSON parsing"
                        elif "SELECT" in full_text.upper() or "WITH" in full_text.upper():
                            # Extract only the SQL part if we can identify it
                            sql_match = re.search(r'(?:WITH|SELECT)\s+[\s\S]*?(?:;|$)', full_text, re.IGNORECASE)
                            if sql_match:
                                sql_query = sql_match.group(0)
                            else:
                                sql_query = full_text
                        else:
                            # Try to extract SQL code block
                            sql_blocks = re.findall(r'```(?:sql)?([\s\S]*?)```', full_text)
                            if sql_blocks and sql_blocks[0].strip():
                                sql_query = sql_blocks[0].strip()
                            else:
                                # Default fallback - but ensure it doesn't look like JSON
                                if not (full_text.strip().startswith('{') and full_text.strip().endswith('}')):
                                    # Only use as SQL if it doesn't look like JSON
                                    sql_query = full_text
                                else:
                                    # Use a default basic SQL query since we can't extract valid SQL
                                    sql_query = "SELECT 'Failed to extract valid SQL' AS error_message"
                        
                        # If we haven't set chart recommendations yet, initialize as empty
                        if 'chart_recommendations' not in locals():
                            chart_recommendations = []
                        if 'chart_error' not in locals():
                            chart_error = "Cannot generate charts: error in JSON parsing"
                    except Exception as e:
                        print(f"Error processing Claude response for charts: {str(e)}")
                        chart_error = f"Error extracting chart recommendations: {str(e)}"
                    
                    # No additional fallback logic for chart recommendations
                    # This matches the OpenAI implementation which only uses chart recommendations
                    # directly from the LLM response
                else:
                    # When charts are not included, focus on extracting SQL - similar to OpenAI implementation
                    # Just extract SQL without chart parsing
                    if full_text.startswith("```") and "```" in full_text[3:]:
                        # Remove markdown code blocks if present
                        lines = full_text.split("\n")
                        # Remove opening ```sql or ``` line
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        # Remove closing ``` line if present
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        # Join back into a cleaned SQL string
                        sql_query = "\n".join(lines)
                    elif "SELECT" in full_text.upper() or "WITH" in full_text.upper():
                        # Extract SQL statements directly from the text
                        sql_match = re.search(r'(?:WITH|SELECT)\s+[\s\S]*?(?:;|$)', full_text, re.IGNORECASE)
                        if sql_match:
                            sql_query = sql_match.group(0)
                        else:
                            sql_query = full_text
                    
                    # Set chart fields to null when charts aren't requested
                    chart_recommendations = None
                    chart_error = None
                
                # If we still don't have SQL but the response contains SELECT keywords
                if "SELECT" in full_text.upper() and not (sql_query and "SELECT" in sql_query.upper()):
                    sql_query = full_text
                
                # Final check to ensure sql_query is a string
                if sql_query is None:
                    sql_query = "SELECT 'SQL extraction resulted in None' AS error_message"
                elif not isinstance(sql_query, str):
                    sql_query = str(sql_query)
                
                # Ensure SQL is not empty
                if not sql_query.strip():
                    sql_query = "SELECT 'Empty SQL query extracted' AS error_message"
                    
            except Exception as e:
                print(f"DEBUG: Error extracting SQL from Claude response: {str(e)}")
                sql_query = f"SELECT 'Error extracting SQL: {str(e).replace("'", "''")}' AS error_message"
        else:
            print("DEBUG: Invalid response structure from Claude")
            sql_query = "SELECT 'Invalid response structure from Claude' AS error_message"
    
        # Ensure we return a valid string for SQL even if extraction failed (OpenAI pattern)
        if sql_query is None:
            sql_query = "SELECT 'Error: SQL query extraction resulted in None' AS error_message"
        elif not isinstance(sql_query, str):
            # Try to convert to string if possible
            try:
                sql_query = str(sql_query)
            except:
                sql_query = "SELECT 'Error: Could not convert SQL query to string' AS error_message"
        elif not sql_query.strip():  # Empty string or whitespace
            sql_query = "SELECT 'Error: Empty SQL query extracted' AS error_message"
        
        # Get token usage
        usage_data = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        # NOTE: Token usage is now logged only in nlq_to_snowflake_claude.py after execution
        # to avoid duplicate entries in the token_usage.csv file
            
        result = {
            "sql": sql_query,
            "model": model,
            "prompt_tokens": usage_data["prompt_tokens"],
            "completion_tokens": usage_data["completion_tokens"],
            "total_tokens": usage_data["total_tokens"]
        }
        
        # Always include chart-related fields in the response for consistent structure
        if include_charts:
            # Make sure chart recommendations are properly formatted
            if chart_recommendations and isinstance(chart_recommendations, list):
                result["chart_recommendations"] = chart_recommendations
            else:
                result["chart_recommendations"] = []
                
            # Make sure chart error is properly formatted
            if chart_error:
                result["chart_error"] = chart_error
            elif not chart_recommendations:
                result["chart_error"] = "No chart recommendations available"
            else:
                result["chart_error"] = None
        else:
            # Set chart fields to null when charts aren't requested
            result["chart_recommendations"] = None
            result["chart_error"] = None
            
        return result
        
    except Exception as e:
        tb_str = traceback.format_exc()
        
        error_msg = f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}\nTRACEBACK: {tb_str.replace("'", "''").replace(chr(10), ' ')}' AS error_message"
        result = {
            "sql": error_msg,
            "model": model,
            "error": str(e),
            "traceback": tb_str,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        # Always include chart-related fields in the response for consistent structure
        # If charts are requested, provide empty list and error message
        # If charts are not requested, provide null values
        if include_charts:
            result["chart_recommendations"] = []
            result["chart_error"] = "Cannot generate charts: error in SQL generation"
        else:
            result["chart_recommendations"] = None
            result["chart_error"] = None
            
        return result


def natural_language_to_sql_claude(query: str, data_dictionary_path: str = None, api_key: Optional[str] = None, 
                               model: str = None, log_tokens: bool = True,
                               limit_rows: int = 100, include_charts: bool = False) -> Dict[str, Any]:
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
    
    
    # Set default API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
    # Set default model from environment if not provided
    if model is None:
        # Try both uppercase and lowercase environment variable names
        model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("anthropic_model", "claude-3-5-sonnet-20241022")
    
    try:
        # Track execution time
        start_time = datetime.datetime.now()
        
        # Load data dictionary
        df = load_data_dictionary(data_dictionary_path)
        
        # Format data dictionary for prompt
        tables = format_data_dictionary(df)
        
        # Generate prompt and SQL query
        prompt = generate_sql_prompt(tables, query, limit_rows=limit_rows, include_charts=include_charts)
        result = generate_sql_query_claude(api_key, prompt, model=model, query_text=query, log_tokens=log_tokens, include_charts=include_charts)
        
        
        
        # Ensure 'sql' key always exists and contains a valid string
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
    
    try:
        result = natural_language_to_sql_claude(query, data_dict_path, model=model)
        
        # Match output format with llm_query_generator.py
        print(f"Using model provider: claude")
        print(f"Using specific model: {result['model']}")
        
        print(f"\nQuery: {query}\n")
        print(f"SQL:\n```sql\n{result['sql']}\n```\n")
        
        print("Token Usage:")
        print(f"  Prompt tokens: {result.get('prompt_tokens', 0)}")
        print(f"  Completion tokens: {result.get('completion_tokens', 0)}")
        print(f"  Total tokens: {result.get('total_tokens', 0)}")
        print(f"\nExecution time: {result.get('execution_time_ms', 0):.2f} ms" if result.get('execution_time_ms') is not None else "\nExecution time: Not available")
    except Exception as e:
        print(f"Error: {str(e)}")

