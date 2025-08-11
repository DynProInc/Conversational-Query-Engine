"""
Gemini LLM Query Generator - A module to convert natural language to SQL using Google Gemini (1.5 Flash)
"""
import os
import pandas as pd
import datetime
import json
import re
from typing import Dict, List, Any, Optional
from token_logger import TokenLogger

# Import cache utilities
from cache_utils import cache_manager

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
from typing import List, Dict, Any

def generate_sql_prompt(tables: List[Dict[str, Any]], query: str, limit_rows: int = 100, include_charts: bool = False, sql_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate system prompt for Gemini with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        limit_rows: Default row limit if not specified in query
        include_charts: Whether to include chart recommendations
        sql_context: Optional SQL context from RAG with query intent and recommendations
        
    Returns:
        Formatted system prompt for Gemini
    """
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
            
            # Only add business name if it's different from column name
            if col['business_name'] != col['name']:
                tables_context += f" [Business Name: {col['business_name']}]"
                
            tables_context += "\n"
        
        tables_context += "\n"
    
    # Create prompt template for Gemini - EXACTLY matching Claude's detailed instructions
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
          Rates=NUMERIC(8,4), use COALESCE(field,0) for NULL safety, always specify precision to prevent truncation across all databases
         dont used TRY_CAST
        - Aggregations: Always use SUM, AVG, etc. with GROUP BY
        - Division safety: Use NULLIF() to prevent division by zero
        - Percentages: [(new_value - old_value) / NULLIF(old_value, 0) * 100](cci:1://file:///c:/Users/git_manoj/Conversational-Query-Engine/LLM%20Query%20Engine/gemini_query_generator.py:28:0-174:17)
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
        # Start with a valid default SQL that will work as a fallback
        sql_query = "SELECT 'No SQL query was successfully extracted' AS error_message"
        chart_error = None
        # Initialize chart_recommendations to empty list to avoid variable scope issues
        chart_recommendations = []
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
                    "chart_recommendations": [],
                    "chart_error": None,
                    "error_execution": "Empty response from Gemini",
                    "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Extract SQL based on whether charts are requested
            import re
            import json
            
            # Initialize chart_recommendations and chart_error
            chart_recommendations = []
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
                    
                    # First attempt: Try to clean invalid control characters and parse
                    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)
                    
                    # Look for JSON object markers
                    json_start = cleaned_text.find('{')
                    json_end = cleaned_text.rfind('}')
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = cleaned_text[json_start:json_end+1]
                        print("Found JSON content between { and }")
                    else:
                        json_content = cleaned_text
                        print("No JSON markers found, using cleaned text")
                    
                    try:
                        # Parse the JSON response with initial cleaning
                        json_data = json.loads(json_content)
                        print(f"Successfully parsed JSON with keys: {list(json_data.keys())}")
                    except json.JSONDecodeError:
                        # If that fails, try with stricter cleanup
                        ultra_clean = ''.join(c for c in json_content if c.isprintable())
                        json_data = json.loads(ultra_clean)
                        print(f"Successfully parsed JSON after ultra cleaning with keys: {list(json_data.keys())}")
                    
                    # Extract SQL query and chart recommendations
                    if "sql" in json_data:
                        sql_query = json_data.get("sql", "")
                        # Ensure we're not sending JSON as SQL
                        if isinstance(sql_query, str):
                            # If the SQL itself is a JSON string, extract the actual SQL
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
                            
                            # Additional safety check - if the SQL still looks like a JSON object, extract just the query
                            if sql_query.strip().startswith("{") and "sql" in sql_query:
                                try:
                                    # One more attempt to extract SQL from JSON
                                    sql_match = re.search(r'"sql"\s*:\s*"(.+?)(?=",|"\s*})', sql_query, re.DOTALL)
                                    if sql_match:
                                        extracted_sql = sql_match.group(1)
                                        # Unescape any escaped quotes
                                        extracted_sql = extracted_sql.replace("\\\"", '"')
                                        sql_query = extracted_sql
                                except Exception as e:
                                    print(f"Additional SQL extraction attempt failed: {str(e)}")
                    
                    # Update chart_recommendations if found in JSON
                    if "chart_recommendations" in json_data:
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
                    
                    # Set chart error if we couldn't extract any charts and charts were requested
                    if include_charts and not chart_recommendations:
                        chart_error = "Cannot generate charts: error in JSON parsing"
                        
                # Always ensure chart_recommendations is defined before returning
                if 'chart_recommendations' not in locals() or chart_recommendations is None:
                    chart_recommendations = []
                    
                # Always ensure chart_error is defined when charts were requested
                if include_charts and 'chart_error' not in locals():
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
                
                # Set chart fields to null when charts aren't requested
                chart_recommendations = []
                chart_error = None
                
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
        
        # Prepare the result dictionary
        result = {
            "sql": sql_query,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Only include chart-related fields if charts were requested
        if include_charts:
            result["chart_recommendations"] = chart_recommendations
            if chart_error:
                result["chart_error"] = chart_error
        
        # Return the result dictionary
        result["execution_time_ms"] = execution_time_ms
        return result
    except Exception as e:
        execution_time_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
        return {
            "sql": f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}'  AS error_message",
            "model": model,
            "error": str(e),
            "chart_recommendations": [],
            "chart_error": f"Chart generation failed: {str(e)}",
            "prompt_tokens": prompt_tokens if 'prompt_tokens' in locals() else 0,
            "completion_tokens": completion_tokens if 'completion_tokens' in locals() else 0,
            "total_tokens": total_tokens if 'total_tokens' in locals() else 0,
            "execution_time_ms": execution_time_ms
        }

def natural_language_to_sql_gemini(query: str, data_dictionary_path: Optional[str] = None, api_key: Optional[str] = None, model: str = None, log_tokens: bool = True, client_id: str = None, use_rag: bool = False, limit_rows: int = 100, include_charts: bool = False, top_k: int = 10, enable_reranking: bool = True) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL using Gemini, matching OpenAI logic for data dictionary and prompt construction.
    If data_dictionary_path is not provided, use the default 'data_dictionary.csv' in the current directory.
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to data dictionary Excel/CSV
        api_key: Gemini API key (will use environment variable if not provided)
        model: Gemini model to use
        log_tokens: Whether to log token usage
        client_id: Client ID for RAG context retrieval
        use_rag: Whether to use RAG for context retrieval
        limit_rows: Maximum rows to return
        include_charts: Whether to include chart recommendations
        top_k: Number of results to return from RAG (default: 10)
        enable_reranking: Whether to enable reranking of RAG results (default: True)
        
    Returns:
        Dictionary with SQL query, token usage and other metadata
    """
    # Check cache first
    cache_context = {
        'function': 'natural_language_to_sql_gemini',
        'model': model,
        'limit_rows': limit_rows,
        'include_charts': include_charts,
        'use_rag': use_rag,
        'top_k': top_k,
        'enable_reranking': enable_reranking
    }
    
    cached_result = cache_manager.get(query, client_id, cache_context)
    if cached_result is not None:
        print(f"Cache HIT for Gemini query: '{query[:30]}...' - client: {client_id}")
        return cached_result
    
    print(f"Cache MISS for Gemini query: '{query[:30]}...' - client: {client_id}")
    import os
    import datetime
    import logging
    import sys  # Import sys for path manipulation
    
    logger = logging.getLogger("gemini_query")
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
                    
                    # Create a RAG manager instance with enable_reranking parameter
                    logger.info(f"Creating RAG manager instance for client {client_id} with enable_reranking={enable_reranking}")
                    rag_manager = RAGManager(enable_reranking=enable_reranking)
                    
                    # Execute the enhanced query
                    # Note: RAG manager's enhanced_query has a default top_k=10
                    top_k_value = top_k  # Use the provided top_k value
                    logger.info(f"Executing RAG enhanced query for client {client_id} with top_k={top_k_value}")
                    success, message, results, sql_context = rag_manager.enhanced_query(
                        client_id=client_id,
                        query_text=query,
                        top_k=top_k_value
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
            # Validate data dictionary path
            if not os.path.isfile(data_dictionary_path):
                raise FileNotFoundError(f"Data dictionary file not found: {data_dictionary_path}")
            df = load_data_dictionary(data_dictionary_path)
            if df.empty:
                raise ValueError("Data dictionary loaded but is empty. Please check your file.")
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
        
        # Multi-layer JSON parsing similar to Claude implementation
        # Layer 1: Check if the SQL is actually a JSON object and extract the SQL string
        if isinstance(result['sql'], str) and result['sql'].strip().startswith('{') and result['sql'].strip().endswith('}'): 
            try:
                # Try to parse as JSON
                sql_json = json.loads(result['sql'])
                if isinstance(sql_json, dict) and 'sql' in sql_json:
                    # Extract the actual SQL from the JSON
                    result['sql'] = sql_json['sql']
                    logger.info("Extracted SQL from JSON response - Layer 1")
                    print("[Gemini Query Generator] Extracted SQL from JSON response - Layer 1")
            except json.JSONDecodeError as e:
                # If it's not valid JSON, try regex extraction as fallback
                logger.info(f"SQL string looks like JSON but couldn't be parsed: {str(e)}")
                print(f"[Gemini Query Generator] SQL string looks like JSON but couldn't be parsed: {str(e)}")
                
                # Fallback regex extraction similar to Claude implementation
                sql_pattern = re.search(r'"sql"\s*:\s*"([^"]+)"', result['sql'])
                if sql_pattern:
                    extracted_sql = sql_pattern.group(1)
                    # Unescape any escaped quotes
                    extracted_sql = extracted_sql.replace("\\\"", '"')
                    result['sql'] = extracted_sql
                    logger.info("Extracted SQL using regex fallback")
                    print("[Gemini Query Generator] Extracted SQL using regex fallback")
        
        # Layer 2: Check if the extracted SQL is still a JSON object (nested JSON)
        if isinstance(result['sql'], str) and result['sql'].strip().startswith('{') and result['sql'].strip().endswith('}'): 
            try:
                # Try to parse as JSON again (handles double-nested JSON)
                nested_sql_json = json.loads(result['sql'])
                if isinstance(nested_sql_json, dict) and 'sql' in nested_sql_json:
                    # Extract the actual SQL from the nested JSON
                    result['sql'] = nested_sql_json['sql']
                    logger.info("Extracted SQL from nested JSON response - Layer 2")
                    print("[Gemini Query Generator] Extracted SQL from nested JSON response - Layer 2")
            except json.JSONDecodeError:
                # If it's not valid JSON at this point, keep it as is
                pass
        
        # Layer 3: If SQL still doesn't look like SQL, try to extract SQL pattern
        if isinstance(result['sql'], str) and not re.search(r'(?:SELECT|WITH)\s+', result['sql'], re.IGNORECASE):
            # Try to find a SQL statement by looking for SELECT or WITH
            sql_match = re.search(r'(?:WITH|SELECT)\s+[\s\S]*?(?:;|$)', result['sql'], re.IGNORECASE)
            if sql_match:
                result['sql'] = sql_match.group(0).strip()
                logger.info("Extracted SQL using pattern matching - Layer 3")
                print("[Gemini Query Generator] Extracted SQL using pattern matching - Layer 3")
        
        # Process escape sequences in the SQL query
        # This handles cases where the query contains literal \n instead of actual newlines
        if '\\n' in result['sql']:
            result['sql'] = result['sql'].encode().decode('unicode_escape')
            logger.info("Processed SQL query to handle escape sequences")
            print("[Gemini Query Generator] Processed SQL with escape sequences")
        # Add additional metadata
        result.update({
            "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "data_dictionary": data_dictionary_path
        })
        
        # Cache the result
        ttl = cache_manager.config.QUERY_GENERATION_TTL
        cache_manager.set(query, result, client_id, cache_context, ttl=ttl)
        
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
            "chart_recommendations": [],
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
