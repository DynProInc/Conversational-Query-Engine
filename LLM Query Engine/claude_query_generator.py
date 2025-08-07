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


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str, limit_rows: int = 100, include_charts: bool = False, sql_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate system prompt for Claude with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        limit_rows: Default row limit if not specified in query
        include_charts: Whether to include chart recommendations
        sql_context: Optional SQL context from RAG with query intent and recommendations
        
    Returns:
        Formatted system prompt for Claude
    """
    
    # Format tables info into string
    tables_context = ""
    
    # Add SQL context if available
    sql_context_str = ""
    if sql_context and isinstance(sql_context, dict):
        sql_context_str = "\n## QUERY ANALYSIS AND RECOMMENDATIONS\n\n"
        
        # Add query intent if available
        if 'query_intent' in sql_context:
            intent = sql_context['query_intent']
            sql_context_str += "### Query Intent Analysis\n"
            if 'operation' in intent:
                sql_context_str += f"- Operation: {intent['operation']}\n"
            if 'aggregation' in intent and intent['aggregation']:
                sql_context_str += f"- Aggregation: {intent['aggregation']}\n"
            if 'filtering' in intent and intent['filtering']:
                sql_context_str += f"- Filtering: {', '.join(intent['filtering'])}\n"
            if 'sorting' in intent and intent['sorting']:
                sql_context_str += f"- Sorting: {intent['sorting']}\n"
            if 'grouping' in intent and intent['grouping']:
                sql_context_str += f"- Grouping: {intent['grouping']}\n"
            if 'limit' in intent and intent['limit']:
                sql_context_str += f"- Limit: {intent['limit']}\n"
            sql_context_str += "\n"
        
        # Add recommended tables if available
        if 'recommended_tables' in sql_context and sql_context['recommended_tables']:
            sql_context_str += "### Recommended Tables\n"
            for i, table in enumerate(sql_context['recommended_tables'], 1):
                sql_context_str += f"{i}. {table['full_name']} (relevance: {table['relevance_score']:.2f})\n"
            sql_context_str += "\n"
        
        # Add column suggestions if available
        if 'column_suggestions' in sql_context and sql_context['column_suggestions']:
            sql_context_str += "### Recommended Columns\n"
            for table_name, suggestions in sql_context['column_suggestions'].items():
                if 'select' in suggestions and suggestions['select']:
                    sql_context_str += f"- For SELECT: {', '.join(suggestions['select'])}\n"
                if 'where' in suggestions and suggestions['where']:
                    sql_context_str += f"- For WHERE: {', '.join(suggestions['where'])}\n"
                if 'group_by' in suggestions and suggestions['group_by']:
                    sql_context_str += f"- For GROUP BY: {', '.join(suggestions['group_by'])}\n"
            sql_context_str += "\n"
    
    # Format tables info
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
    prompt = f"""
        Human: I need you to generate a Snowflake SQL query based on user intent,optimized SQL query for the following natural language request. Use only the tables and columns provided below.

## DATABASE SCHEMA
{tables_context}
{sql_context_str}
## NATURAL LANGUAGE QUERY
{query}

## REQUIREMENTS (Apply to Both SQL and Charts)

        ### Column-Table Validation Framework
        1. Single Table Queries: Verify ALL columns exist in selected table before query generation
        2. Multi-Table Queries: Identify table for each column, ensure proper JOIN relationships exist
        3. Error Handling: Format as `-- ERROR: Column '[name]' not found in table '[table_name]'`
        4. Resolution Process: 
        - Select primary table based on query intent
        - Find alternative columns in same table if missing
        - Add JOINs for cross-table column access
        - Use business context for disambiguation

        ### Time-Based Data Handling (Universal)
        - Multi-year or year data: ALWAYS use YYYY-MM format (e.g., "2024-05") for proper chronological ordering
        - SQL: Create/use year-month columns from date fields for time-based queries
        - Charts: Use full date format column as x-axis, NOT month names alone
        - Examples: TRANSACTION_YEAR_MONTH, DATE_TRUNC('month', date_column)

        ### Row Limits and Partitioning
        Priority Order:
        1. Explicit number (e.g., "top 5") → LIMIT that exact number
        2. Partition keywords ("each", "every", "per", "by") → ROW_NUMBER() OVER (PARTITION BY [group] ORDER BY [metric]) WHERE rn <= X
        3. "Top/first" without number → LIMIT {limit_rows}
        4. Default → LIMIT {limit_rows}

        ### JOIN Logic
        - Default: INNER JOIN unless context suggests otherwise
        - Preservation: LEFT JOIN for "all X even without Y" scenarios
        - Auto-detection: Use schema foreign keys for relationships
        - Multi-table: Chain JOINs using common keys with table aliases always
         Make sure that you never return two columns with the same name, especially
         after joining two tables. You can differentiate the same column name by
         applying column_name + table_name or used alis tables always.
        
        Before generating SQL, identify:
        - Primary Objective: What user wants to achieve
        - Query Type: aggregation|filtering|joining|ranking|time_series|comparison|distribution
        - Key Metrics: What to measure/analyze
        - Grouping Dimensions: How data should be segmented
        - Expected Output: single_value|list|trend|comparison|distribution etc.
        
        ## PHASE 1: SQL GENERATION

        ### Basic Requirements
        1. Only SELECT queries with proper Snowflake syntax
        2. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the
        database    
        3. Fully qualified table names (SCHEMA.TABLE_NAME)
        4. be careful do not query for columns that do not exist in the table
        5. Use table alias for table names in the query
        6. Uppercase column names, individual column listing
        7. Proper indentation and formatting
        8. end query with ;

        ### Numeric Value Handling
        - Type conversion: Monetary=NUMERIC(15,2), Quantities=INTEGER, Percentages=NUMERIC(5,2),
         Rates=NUMERIC(8,4), use COALESCE(field,0) for NULL safety, always specify precision to prevent truncation across all databases.
        - Aggregations: Always use SUM, AVG, etc. with GROUP BY
        - Division safety: Use NULLIF() to prevent division by zero
        - Percentages: `(new_value - old_value) / NULLIF(old_value, 0) * 100`
        - Ratios: `numerator / NULLIF(denominator, 0)`
        - Naming: Uppercase for aggregations (TOTAL_SALES), original casing for regular columns

        ### SQL Formatting Standards
        - Spaces between keywords and identifiers: `SELECT column FROM table`
        - Spaces around operators: `column = value`
        - Spaces after commas: `col1, col2, col3`
        - Spaces after AS keyword: `SUM(value) AS TOTAL`
        - Separate keywords with spaces: `GROUP BY`

        ### Query Optimization
        - Avoid SELECT * in production
        - Use QUALIFY for window functions instead of subqueries
        - Apply appropriate indexes when available

        ### SUPERLATIVE QUERY HANDLING:
        1. CRITICAL: For PLURAL nouns in queries like "which sales representatives sold most" - NEVER add LIMIT 1, return MULTIPLE results
        2. For SINGULAR nouns in queries like "which sales rep sold most" - ALWAYS add `ORDER BY [relevant_metric] DESC LIMIT 1`
        3. Explicitly check if words like "representatives", "products", "customers" (plural) are used
        4. Examples: "which sales rep sold most" → ONE result (LIMIT 1), "which sales representatives sold most" → MULTIPLE results (NO LIMIT 1)

        ## PHASE 2: CHART RECOMMENDATIONS

        ### Chart Selection Rules
        1. Analyze SQL output structure (columns, data types, aggregations)
        2. Provide atmost 2 chart recommendations ordered by priority
        3. Match chart type to data structure using rules below

        ### Chart Type Framework
        Each chart type follows this structure:
        - Required Data: Specific column types and count needed
        - Best Use Cases: When to recommend this chart type
        - Scale Considerations: How to handle multiple measures

        Generate a SQL query and chart recommendations for: {query}
        """

    chart_instructions = """
        ### Chart Type Definitions

        Bar Charts: Categorical comparisons (≤15 categories)
        - Required: 1 categorical column + 1+ numeric measures
        - Best for: Category comparisons, time periods
        - Multi-scale: Convert to mixed chart if needed

        Line Charts: Trends over continuous ranges
        - Required: 1 continuous axis + 1+ numeric measures  
        - Best for: Time trends, continuous data
        - Multi-scale: Use 2-4 Y-axis configuration

        Pie Charts: Composition/parts of whole (≤7 categories)
        - Required: 1 categorical column (≤7 values) + 1 numeric measure
        - Best for: Proportions, part-to-whole relationships

        Scatter Plots: Relationships between variables
        - Required: 2+ numeric measures
        - Best for: Correlation analysis, distribution patterns
        - Multi-scale: Use different axis scales

        Area Charts: Cumulative totals over time/categories
        - Required: 1 continuous axis + 1+ numeric measures
        - Best for: Cumulative trends, stacked compositions
        - Multi-scale: Convert to mixed chart if needed

        Mixed/Combo Charts: Multi-measure comparisons
        - Required: 1 shared axis + multiple measures (different scales)
        - Best for: Different but related metrics
        - Multi-scale: Primary purpose of this chart type

        KPI Cards: Single numeric values
        - Required: 1 numeric measure
        - Best for: Key metrics, minimal display
        - Multi-scale: Not applicable

        Tables: Categorical data without numeric measures
        - Required: Multiple categorical columns
        - Best for: Detailed data display
        - Multi-scale: Not applicable

        Histogram Charts: Distribution analysis of continuous variables
        - Required: Numerical continuous data only (no categorical)
        - Best for: Frequency/spread/skewness/outliers in large datasets
        - Use auto-binning (Sturges/Freedman-Diaconis) for proper bin sizing
        - X-axis: value range,
        - "y_axis": [],   "y_axis": null,  // No column needed - frequency is calculated
        For a histogram:
        - Use 1 numeric column.
        - X-axis = value bins from that column.
        - Y-axis = count (frequency), computed from how many values fall in each bin.
        - Y-axis is not from any column.

        
        - Ensure contiguous bins (no gaps)
        - Avoid overlapping distributions (use separate plots/density plots)
        - Skip for small datasets (use box/dot plots instead)

        Box Plot Charts: Distribution comparison between groups
        - Required: Numerical data (can group by categorical)
          if one columns then use null for x_axis,
          other wise use categorical column for x_axis
           e.g:  "x_axis": "PRODUCT_CATEGORY",
            "y_axis": ["SALES_AMOUNT"],
            "color_by": "PRODUCT_CATEGORY",
            - Best for: Comparing distributions, showing central tendency/spread/outliers
            - Box = IQR (Q1-Q3), line = median
            - Whiskers = Q1-1.5×IQR to Q3+1.5×IQR
            - Points beyond whiskers = outliers
            - Best for side-by-side comparisons
            - Consider combining with histograms/violin plots for distribution shape details

        ### Multi-Scale Detection (All Chart Types)
        Scale Detection Rules:
        - Trigger: Scale ratio ≥ 100x (larger_value / smaller_value ≥ 100)
        - Examples: Sales $50K vs Orders 25 (2000x ratio → secondary axis)
        - Action: Use secondary Y-axis or convert to mixed chart
        - if required Use normalization to bring scales together

        Chart-Specific Scale Handling:
        - Bar/Line/Area: Convert to mixed chart or use dual Y-axis
        - Mixed: Assign measures to appropriate axes
        - Configuration: Always include scale detection parameters

        ### Enhanced Configuration
        All charts with multiple measures must include:
        ```json
        "additional_config": {
        "show_legend": true,
        "orientation": "vertical|horizontal",
        "scale_detection": true,
        "scale_threshold": 2,
        "use_secondary_axis": true,
        "secondary_axis_columns": ["column_name"],
        "primary_axis_label": "Primary Metric Unit",
        "secondary_axis_label": "Secondary Metric Unit",
        "axis_assignment_reasoning": "Brief explanation"
        }
        ```

        ## OUTPUT FORMAT

        Your response MUST be a valid JSON object with the following structure:
             {
        "sql": "YOUR SQL QUERY HERE",
        "chart_recommendations": [
            {
            "chart_type": "bar|pie|line|scatter|histogram|area|combo|mixed|table|KPI Card",
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
                "scale_detection": true,
                "scale_threshold": 2,
                "use_secondary_axis": true,
                "secondary_axis_columns": ["column_name2"],
                "primary_axis_label": "Primary Metric Unit",
                "secondary_axis_label": "Secondary Metric Unit",
                "axis_assignment_reasoning": "Brief explanation"
                }
            }
            }
        ]
        }

        Example with Multi-Scale Detection:
        {
        "sql": "SELECT MONTH_NAME, SUM(SALES) AS TOTAL_SALES, COUNT(ORDER_ID) AS ORDER_COUNT FROM ORDERS GROUP BY MONTH_NAME ORDER BY MONTH_NUMBER;",
        "chart_recommendations": [{
            "chart_type": "mixed",
            "reasoning": "TOTAL_SALES and ORDER_COUNT have different scales requiring secondary axis",
            "priority": 1,
            "chart_config": {
            "title": "Monthly Sales and Orders",
            "x_axis": "MONTH_NAME",
            "series": [
                { "column": "TOTAL_SALES", "type": "bar", "axis": "primary" },
                { "column": "ORDER_COUNT", "type": "line", "axis": "secondary" },
                            { "column": "PROFIT_MARGIN", "type": "scatter", "axis": "primary", "scale": "normalized" }
            ],
            "additional_config": {
                "scale_detection": true,
                "scale_threshold": 2,
                "use_secondary_axis": true,
                "secondary_axis_columns": ["ORDER_COUNT"],
                "primary_axis_label": "Sales Amount ($)",
                "secondary_axis_label": "Order Count",
                "axis_assignment_reasoning": "Sales values are 1000x larger than order counts"
            }
            }
        }]
        }

        

        IMPORTANT: DO NOT include any explanations, code blocks, or markdown formatting outside the JSON. Your entire response must be valid JSON that can be parsed directly.IMPORTANT: DO NOT include any explanations, code blocks, or markdown formatting outside the JSON. Your entire response must be valid JSON that can be parsed directly.
        
        """
    if include_charts:
        prompt += chart_instructions
    else:
        prompt += """
        Return ONLY the SQL code wrapped in triple backticks with the sql tag like this:
        ```sql
        SELECT column FROM table WHERE condition;
        ```
        Do not include any explanations, text, or other content before or after the SQL code block.
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
                    # Removed debug print statement
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
                else:
                    # Fallback: If no code blocks found, try to identify SQL by common SQL keywords
                    # This helps when Claude returns raw SQL without code blocks
                    sql_pattern = re.search(r'(?:SELECT|WITH)\s+[\s\S]*?(?:;|$)', full_text, re.IGNORECASE)
                    if sql_pattern:
                        extracted_sql = sql_pattern.group(0).strip()
                        if extracted_sql:  # Only update if we got something meaningful
                            sql_query = extracted_sql

                # Clean up SQL by removing comment lines, blank lines, and json prefix
                if sql_query:
                    # Remove 'json' prefix if present (happens in some responses)
                    if sql_query.strip().startswith('json'):
                        sql_query = sql_query[4:].strip()
                        
                    # Remove any comment lines (both -- and #) and blank lines
                    sql_query = '\n'.join([line for line in sql_query.splitlines() 
                                          if line.strip() and not line.strip().startswith('--') and not line.strip().startswith('#')])
                    
                    # Final cleanup
                    sql_query = sql_query.strip()

                # Debug print
                print("[Claude SQL Extraction] Final SQL to execute:\n", sql_query)
               

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
                        except json.JSONDecodeError:
                            # If that still fails, try with the original but stricter cleanup
                            # Replace all non-printable characters
                            ultra_clean = ''.join(c for c in full_text if c.isprintable())
                            json_data = json.loads(ultra_clean)
                            json_parsing_success = True
                        
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
                        # Error parsing JSON response - debug print removed
                        
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
                # FINAL CLEANUP: Remove json prefix, comments and blank lines from extracted SQL
                # First, check if SQL starts with 'json' prefix (happens with some Claude responses)
                if sql_query.strip().startswith('json'):
                    sql_query = sql_query[4:].strip()
                
                # Remove comment lines (both -- and #) and blank lines
                lines = [line for line in sql_query.splitlines() 
                         if line.strip() and not line.strip().startswith("--") and not line.strip().startswith("#")]
                sql_query_cleaned = "\n".join(lines)
                
                if not sql_query_cleaned:
                    sql_query_cleaned = "SELECT 'No valid SQL extracted' AS error_message"
                    
                sql_query = sql_query_cleaned.strip()  # Final strip to remove any leading/trailing whitespace
                
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
                               model: str = None, log_tokens: bool = True, client_id: str = None,
                               use_rag: bool = False, limit_rows: int = 100, include_charts: bool = False, 
    """
    End-to-end function to convert natural language to SQL using Claude
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to data dictionary Excel/CSV
        api_key: Anthropic API key (will use environment variable if not provided)
        model: Claude model to use
        log_tokens: Whether to log token usage
        client_id: Client ID for RAG context retrieval
        use_rag: Whether to use RAG for context retrieval
        limit_rows: Maximum rows to return
        include_charts: Whether to include chart recommendations
        
    Returns:
        Dictionary with SQL query, token usage and other metadata
    """
    import logging
    import os  # Ensure os is imported at function scope
    import sys  # Import sys at function scope
    logger = logging.getLogger("claude_query")
    
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
        
        # Initialize function-level SQL context variable
        function_sql_context = None
        
        # Determine if we should use RAG
        context_type = "RAG" if use_rag else "Full Dictionary"
        logger.info(f"Using {context_type} for query: '{query}'")
        
        # Get context either from RAG or full dictionary
        if use_rag and client_id:
            try:
                logger.info(f"Retrieving RAG context for client {client_id}")
                
                # Access RAG functionality directly
                try:
                    # Add milvus-setup to path if needed
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    milvus_setup_dir = os.path.join(current_dir, "milvus-setup")
                    if milvus_setup_dir not in sys.path:
                        sys.path.append(milvus_setup_dir)
                    
                    # Import the RAG manager directly
                    logger.info(f"Importing RAG embedding module from {milvus_setup_dir}")
                    from rag_embedding import RAGManager
                    
                    
                    # Execute the enhanced query
                    logger.info(f"Executing RAG enhanced query for client {client_id} with top_k={top_k}")
                    success, message, results, sql_context = rag_manager.enhanced_query(
                        client_id=client_id,
                        query_text=query,
                        top_k=top_k
                    )
                    
                    # Format the response like the API would
                    rag_data = {
                        "success": success,
                        "message": message,
                        "results": results if success and results else [],
                        "sql_context": sql_context if success and sql_context else None
                    }
                    
                    # Store the SQL context at the function level to ensure it's available for the prompt
                    function_sql_context = sql_context if success and sql_context else None
                    
                    # Log the sql_context specifically
                    logger.info(f"SQL Context type: {type(sql_context)}")
                    logger.info(f"SQL Context is None: {sql_context is None}")
                    if sql_context:
                        logger.info(f"SQL Context keys: {sql_context.keys() if isinstance(sql_context, dict) else 'Not a dict'}")
                    else:
                        logger.info("SQL Context is None or empty")
                    
                    if success and results:
                        logger.info(f"Successfully retrieved RAG results: {len(results)} items")
                        
                        # Print the raw RAG results structure
                        logger.info("=== RAW RAG RESULTS STRUCTURE ===")
                        import json
                        logger.info(json.dumps(results, indent=2))
                        logger.info("=== END RAW RAG RESULTS STRUCTURE ===")
                        
                        # Print the SQL context if available
                        if sql_context:
                            logger.info("=== RAG SQL CONTEXT ===")
                            logger.info(json.dumps(sql_context, indent=2))
                            logger.info("=== END RAG SQL CONTEXT ===")
                        
                        # Group results by table
                        table_dict = {}
                        for item in results:
                            table_name = item.get("table_name")
                            if not table_name:
                                logger.warning("Missing table_name in RAG result item, skipping")
                                continue
                                
                            if table_name not in table_dict:
                                table_dict[table_name] = {
                                    "name": table_name,
                                    "schema": item.get("schema_name", ""),
                                    "description": "Table from RAG context",
                                    "columns": []
                                }
                                
                            # Add column information
                            table_dict[table_name]["columns"].append({
                                "name": item.get("column_name", ""),
                                "type": item.get("data_type", ""),
                                "description": item.get("description", ""),
                                "business_name": item.get("column_name", "")
                            })
                            
                        # Convert dictionary to list format expected by generate_sql_prompt
                        tables = list(table_dict.values())
                        logger.info(f"Retrieved {len(tables)} tables from RAG")
                        
                        # Print the RAG embeddings that will go into the prompt
                        logger.info("=== RAG EMBEDDINGS START ===")
                        for table in tables:
                            logger.info(f"Table: {table['name']}")
                            for column in table['columns']:
                                logger.info(f"  - {column['name']} ({column['type']}): {column['description']}")
                        logger.info("=== RAG EMBEDDINGS END ===")
                    else:
                        logger.warning(f"RAG query failed: {message}")
                        raise Exception(f"RAG query failed: {message}")
                    
                    # Check if we need to fall back to the full dictionary
                    if not tables or len(tables) == 0:
                        logger.warning("RAG results empty or invalid, falling back to full dictionary")
                        df = load_data_dictionary(data_dictionary_path)
                        tables = format_data_dictionary(df)
                except Exception as e:
                    logger.error(f"Error accessing RAG functionality directly: {str(e)}")
                    logger.warning("Falling back to full dictionary due to RAG error")
                    df = load_data_dictionary(data_dictionary_path)
                    tables = format_data_dictionary(df)
            except Exception as e:
                # Error handling
                logger.error(f"Error using RAG: {str(e)}, falling back to full dictionary")
                df = load_data_dictionary(data_dictionary_path)
                tables = format_data_dictionary(df)
        else:
            # Use traditional full dictionary approach
            logger.info(f"Using full dictionary from {data_dictionary_path}")
            df = load_data_dictionary(data_dictionary_path)
            tables = format_data_dictionary(df)
        
        # Generate prompt and SQL query with SQL context if available
        logger.info(f"SQL context before prompt generation: {type(function_sql_context)}")
        if function_sql_context:
            logger.info(f"SQL context has keys: {function_sql_context.keys() if isinstance(function_sql_context, dict) else 'Not a dict'}")
            if 'recommended_tables' in function_sql_context:
                logger.info(f"Number of recommended tables: {len(function_sql_context['recommended_tables'])}")
                for i, table in enumerate(function_sql_context['recommended_tables'][:3]):
                    logger.info(f"Recommended table {i+1}: {table['full_name']} (score: {table['relevance_score']:.2f})")
        
        prompt = generate_sql_prompt(tables, query, limit_rows=limit_rows, include_charts=include_charts, 
                               sql_context=function_sql_context)
        
        # Log only the prompt length to avoid excessive logging
        logger.info(f"Generated prompt for Claude with length: {len(prompt)} characters")
        
        result = generate_sql_query_claude(api_key, prompt, model=model, query_text=query, log_tokens=log_tokens, include_charts=include_charts)
        
        
        
        # Ensure 'sql' key always exists and contains a valid string
        if 'sql' not in result or result['sql'] is None or not isinstance(result['sql'], str):
            result['sql'] = "SELECT 'No valid SQL query was generated' AS message"
        
        # Process escape sequences in the SQL query
        # This handles cases where the query contains literal \n instead of actual newlines
        if '\\n' in result['sql']:
            result['sql'] = result['sql'].encode().decode('unicode_escape')
            logger.info("Processed SQL query to handle escape sequences")
            print("[Claude Query Generator] Processed SQL with escape sequences")
        
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

