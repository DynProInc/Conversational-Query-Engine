"""
LLM Query Generator - A simple module to convert natural language to SQL using OpenAI
"""
import os
import pandas as pd
import openai
import datetime
import json
import re
import dotenv
from typing import Dict, List, Any, Optional, Union
from token_logger import TokenLogger

# Load environment variables from .env file
dotenv.load_dotenv()


def load_data_dictionary(file_path: str) -> pd.DataFrame:
    """
    Load data dictionary from Excel or CSV
    
    Args:
        file_path: Path to Excel or CSV file with data dictionary
        
    Returns:
        DataFrame containing the data dictionary
    """
    # Handle None case to prevent 'NoneType' object has no attribute 'endswith' error
    if file_path is None:
        print("WARNING: No data dictionary path provided")
        # Return an empty DataFrame with some basic columns - using uppercase column names
        return pd.DataFrame(columns=["TABLE_NAME", "COLUMN_NAME", "DESCRIPTION", "SAMPLE_VALUES", "DATA_TYPE"])
    
    print(f"\nLoading data dictionary from: {file_path}")
        
    # Now we can safely check the file extension
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel or CSV file.")
    
    # Keep this function lean without debug prints
    return df


def format_data_dictionary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Format data dictionary DataFrame for LLM prompt
    
    Args:
        df: DataFrame with data dictionary
        
    Returns:
        List of formatted table dictionaries
    """
    # Format the data dictionary
    tables = []
    
    # Map columns dynamically based on what's available
    column_mappings = {
        # Map for table name: try multiple possible column names
        'TABLE_NAME': next((col for col in ['TABLE_NAME', 'VIEW_NAME', 'ENTITY_NAME'] if col in df.columns), None),
        # Map for column name
        'COLUMN_NAME': next((col for col in ['COLUMN_NAME', 'FIELD_NAME'] if col in df.columns), None),
        # Map for description
        'DESCRIPTION': next((col for col in ['DESCRIPTION', 'COLUMN_DESCRIPTION'] if col in df.columns), None),
        # Map for data type
        'DATA_TYPE': next((col for col in ['DATA_TYPE', 'TYPE', 'COLUMN_TYPE'] if col in df.columns), None),
        # Map for schema
        'DB_SCHEMA': next((col for col in ['DB_SCHEMA', 'SCHEMA_NAME', 'DATABASE_SCHEMA'] if col in df.columns), None),
    }
    
    # Validate required columns are found
    required_mappings = ['TABLE_NAME', 'COLUMN_NAME', 'DESCRIPTION']
    missing_mappings = [m for m in required_mappings if column_mappings[m] is None]
    if missing_mappings:
        missing_str = ', '.join(missing_mappings)
        available_cols = ', '.join(df.columns)
        raise ValueError(f"Required mapping(s) '{missing_str}' not found in data dictionary. Available columns: {available_cols}")
    
    # Add schema awareness check
    schema_aware = column_mappings['DB_SCHEMA'] is not None
    
    # Create a working copy with standardized column names for internal processing
    working_df = df.copy()
    for internal_name, df_column in column_mappings.items():
        if df_column and df_column in df.columns:
            working_df[internal_name] = df[df_column]
    
    # Group by table name (and schema if available)
    if schema_aware:
        # Make sure DB_SCHEMA is properly formatted with database and schema
        working_df['FULL_TABLE_NAME'] = working_df.apply(
            lambda row: f"{row['DB_SCHEMA']}.{row['TABLE_NAME']}" 
            if '.' in row['DB_SCHEMA'] 
            else f"{row['DB_SCHEMA']}.{row['TABLE_NAME']}", 
            axis=1
        )
        # Using schema-aware mode with full table names
        groupby_col = 'FULL_TABLE_NAME'
    else:
        # No schema column found - grouping by table name only
        groupby_col = 'TABLE_NAME'
    
    for full_table_name, group in working_df.groupby(groupby_col):
        # Process each table
        columns = []
        for _, row in group.iterrows():
            column_info = {
                'name': row['COLUMN_NAME'],
                # Use a default type if DATA_TYPE is available, otherwise VARCHAR
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
            
            # Add sample values if available (from DISTINCT_VALUES or SAMPLE_VALUES)
            if 'DISTINCT_VALUES' in row:
                column_info['sample_values'] = str(row['DISTINCT_VALUES'])
            elif 'SAMPLE_VALUES' in row:
                column_info['sample_values'] = str(row['SAMPLE_VALUES'])
                
            columns.append(column_info)
        
        # Use the first row's TABLE_NAME and DB_SCHEMA
        table_name = group['TABLE_NAME'].iloc[0]
        db_schema = group['DB_SCHEMA'].iloc[0] if schema_aware else ''
        
        # Try to get table description from various possible columns
        table_description = ''
        for desc_col in ['TABLE_DESCRIPTION', 'VIEW_DESCRIPTION', 'ENTITY_DESCRIPTION']:
            if desc_col in group.columns and not group[desc_col].iloc[0] is None:
                table_description = group[desc_col].iloc[0]
                break
        
        table_info = {
            'name': full_table_name,
            'schema': db_schema if schema_aware else '',
            'simple_name': table_name,
            'description': table_description,
            'columns': columns
        }
        
        tables.append(table_info)
    
    # Print a summary of the tables that will be used in the prompt
    print(f"\nFormatted {len(tables)} tables for LLM prompt:")
    for i, table in enumerate(tables):
        print(f"  {i+1}. {table['name']} - {len(table['columns'])} columns")
    
    return tables


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str, limit_rows: int = 100, include_charts: bool = False, sql_context: Dict[str, Any] = None) -> str:
    """
    Generate system prompt for the LLM with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        limit_rows: Maximum number of rows to return
        include_charts: Whether to include chart recommendations in the output
        
    Returns:
        Formatted system prompt for the LLM
    """
    # Tables information is used for the prompt context
    
    # Format tables info into string
    tables_context = ""
    
    # Add RAG-derived SQL context if available
    sql_context_str = ""
    if sql_context and isinstance(sql_context, dict):
        sql_context_str = "\n### RELEVANT SQL CONTEXT FROM SIMILAR QUERIES\n"
        
        # Add query intent if available
        if "query_intent" in sql_context and isinstance(sql_context['query_intent'], str):
            sql_context_str += f"\nQuery Intent: {sql_context['query_intent']}\n"
        
        # Add recommended tables if available
        if "recommended_tables" in sql_context and isinstance(sql_context['recommended_tables'], list):
            sql_context_str += "\nRecommended Tables:\n"
            for table in sql_context['recommended_tables']:
                if isinstance(table, dict) and 'full_name' in table and 'relevance_score' in table:
                    sql_context_str += f"- {table['full_name']} (Relevance: {table['relevance_score']:.2f})\n"
                elif isinstance(table, dict) and 'name' in table:
                    # Alternative format sometimes returned by RAG
                    sql_context_str += f"- {table['name']} (Relevance: {table.get('score', 0):.2f})\n"
        
        # Add column suggestions if available
        if "column_suggestions" in sql_context and isinstance(sql_context['column_suggestions'], list):
            sql_context_str += "\nRecommended Columns:\n"
            for col in sql_context['column_suggestions']:
                if isinstance(col, dict) and 'name' in col and 'table' in col and 'score' in col:
                    sql_context_str += f"- {col['name']} in {col['table']} (Relevance: {col['score']:.2f})\n"
                elif isinstance(col, dict) and 'name' in col:
                    # Alternative format
                    sql_context_str += f"- {col['name']} (Relevance: {col.get('relevance', 0):.2f})\n"
        
        # Add example queries if available
        if "example_queries" in sql_context and isinstance(sql_context["example_queries"], list) and sql_context["example_queries"]:
            sql_context_str += "\nExample Queries:\n"
            for i, example in enumerate(sql_context["example_queries"][:2]):  # Limit to 2 examples
                if isinstance(example, str):
                    sql_context_str += f"- {example}\n"
                elif isinstance(example, dict) and 'query' in example:
                    # Alternative format
                    sql_context_str += f"- {example['query']}\n"
                    if 'sql' in example and example['sql']:
                        sql_context_str += f"  SQL: {example['sql']}\n"
        
        # Add the SQL context string to tables_context
        tables_context += sql_context_str
        
        tables_context += "\n### DATABASE SCHEMA\n"
    
    # Add table schema information
    for table in tables:
        tables_context += f"Table: {table['name']}"
        if 'schema' in table and table['schema']:
            tables_context += f" (Schema: {table['schema']})"
        if 'description' in table and table['description']:
            tables_context += f"\nDescription: {table['description']}"
        tables_context += "\nColumns:\n"
        
        for column in table['columns']:
            tables_context += f"- {column['name']}"
            if 'type' in column and column['type']:
                tables_context += f" (Type: {column['type']})"
            if 'description' in column and column['description']:
                tables_context += f": {column['description']}"
            tables_context += "\n"
        
        tables_context += "\n"
    
    # Create prompt template
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


def generate_sql_query(api_key: str, prompt: str, model: str = "gpt-4", 
                    query_text: str = "", log_tokens: bool = True, include_charts: bool = False) -> Dict[str, Any]:
    """
    Generate SQL query using OpenAI's API
    
    Args:
        api_key: OpenAI API key
        prompt: System prompt with schema and query
        model: OpenAI model to use
        query_text: Original natural language query (for logging)
        log_tokens: Whether to log token usage
        include_charts: Whether chart recommendations are requested
        
    Returns:
        Dictionary with SQL query and token usage info
    """
    client = openai.OpenAI(api_key=api_key)
    logger = TokenLogger() if log_tokens else None
    
    import datetime
    start_time = datetime.datetime.now()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Extract SQL query from response
        content = response.choices[0].message.content.strip()
        
        # Log tokens if requested
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        result = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Extract SQL from content regardless of include_charts parameter
        if include_charts:
            # Try to parse the JSON response with SQL and chart recommendations
            try:
                # Remove code blocks if present
                if content.startswith("```json") and content.endswith("```"):
                    content = content[7:-3].strip()
                elif content.startswith("```") and content.endswith("```"):
                    content = content[3:-3].strip()
                
                # Parse the JSON response
                json_data = json.loads(content)
                
                # Extract SQL query and chart recommendations
                result["sql"] = json_data.get("sql", "")
                result["chart_recommendations"] = json_data.get("chart_recommendations", [])
                
            except json.JSONDecodeError as e:
                # Fallback if JSON parsing fails
                print(f"Error parsing JSON response: {str(e)}")
                # Extract SQL from the content if it contains a SELECT statement
                if "SELECT" in content.upper():
                    result["sql"] = content
                else:
                    # Try to extract SQL code block
                    import re
                    sql_blocks = re.findall(r'```(?:sql)?([\s\S]*?)```', content)
                    if sql_blocks and sql_blocks[0].strip():
                        result["sql"] = sql_blocks[0].strip()
                    else:
                        # Default fallback
                        result["sql"] = content
                        
                result["chart_recommendations"] = []
                result["chart_error"] = "Cannot generate charts: error in JSON parsing"
        else:
            # Just extract SQL without chart parsing
            if content.startswith("```") and "```" in content[3:]:
                # Remove markdown code blocks if present
                lines = content.split("\n")
                # Remove opening ```sql or ``` line
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove closing ``` line if present
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                # Join back into a cleaned SQL string
                result["sql"] = "\n".join(lines)
            else:
                # Use content as-is
                result["sql"] = content
                
            # Set chart fields to null when charts aren't requested
            result["chart_recommendations"] = None
            result["chart_error"] = None
            
        # FINAL CLEANUP: Remove json prefix, comments and blank lines from extracted SQL
        if "sql" in result and result["sql"]:
            sql_query = result["sql"]
            # Remove 'json' prefix if present 
            if sql_query.strip().startswith('json'):
                sql_query = sql_query[4:].strip()
            
            # Remove comment lines (both -- and #) and blank lines
            lines = [line for line in sql_query.splitlines() 
                    if line.strip() and not line.strip().startswith("--") and not line.strip().startswith("#")]
            sql_query_cleaned = "\n".join(lines)
            
            if not sql_query_cleaned:
                sql_query_cleaned = "SELECT 'No valid SQL extracted' AS error_message"
                
            # Final strip to remove any leading/trailing whitespace
            result["sql"] = sql_query_cleaned.strip()
        
        # Only log token usage for non-OpenAI models here
        # For OpenAI, token logging is handled in nlq_to_snowflake.py to avoid duplicate logs
        if logger and log_tokens and not model.startswith("gpt-"):
            logger.log_usage(
                model=model,
                query=query_text,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                prompt=query_text,  # Only log the actual user query, not the system prompt
                sql_query=result["sql"]
            )

        return result
    except Exception as e:
        # Handle any exceptions during API call
        print(f"Error generating SQL query: {str(e)}")
        result = {
            "sql": f"SELECT 'Error generating SQL query: {str(e).replace("'", "''")}' AS error_message",
            "model": model,
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "timestamp": datetime.datetime.now().isoformat()
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


def natural_language_to_sql(query: str, data_dictionary_path: Optional[str] = None, 
                       api_key: Optional[str] = None, model: str = None, log_tokens: bool = True,
                       model_provider: str = "openai", limit_rows: int = 100, include_charts: bool = False,
                       client_id: str = None, use_rag: bool = False) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to data dictionary Excel/CSV
        api_key: OpenAI API key (will use environment variable if not provided)
        model: OpenAI model to use
        log_tokens: Whether to log token usage
        model_provider: Model provider to use (openai or claude)
        limit_rows: Maximum number of rows to return
        include_charts: Whether chart recommendations are requested
        client_id: Client ID for RAG context retrieval
        use_rag: Whether to use RAG for context retrieval
        
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
        return natural_language_to_sql_claude(query, data_dictionary_path, api_key, model, log_tokens, limit_rows, include_charts)
    else:  # Default to OpenAI
        # Default to environment variable if API key not provided
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Default to environment variable if model not provided
        if model is None:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    
    # Track execution time
    start_time = datetime.datetime.now()
    
    # Initialize function-level SQL context variable
    function_sql_context = None
    
    # Determine if we should use RAG
    context_type = "RAG" if use_rag else "Full Dictionary"
    print(f"Using {context_type} for query: '{query}'")
    
    # Get context either from RAG or full dictionary
    if use_rag and client_id:
        try:
            print(f"Retrieving RAG context for client {client_id}")
            
            # Access RAG functionality directly
            try:
                # Add milvus-setup to path if needed
                import sys
                current_dir = os.path.dirname(os.path.abspath(__file__))
                milvus_setup_dir = os.path.join(current_dir, "milvus-setup")
                if milvus_setup_dir not in sys.path:
                    sys.path.append(milvus_setup_dir)
                
                # Import the RAG manager directly
                print(f"Importing RAG embedding module from {milvus_setup_dir}")
                from rag_embedding import RAGManager
                
                # Create a RAG manager instance
                print(f"Creating RAG manager instance for client {client_id}")
                rag_manager = RAGManager()
                
                # Execute the enhanced query
                # Note: RAG manager's enhanced_query has a default top_k=10
                top_k_value = 10  # Match the default in RAG manager
                print(f"Executing RAG enhanced query for client {client_id} with top_k={top_k_value}")
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
                print(f"SQL Context type: {type(sql_context)}")
                print(f"SQL Context is None: {sql_context is None}")
                if sql_context:
                    print(f"SQL Context keys: {sql_context.keys() if isinstance(sql_context, dict) else 'Not a dict'}")
                else:
                    print("SQL Context is None or empty")
                
                if success and results:
                    print(f"Successfully retrieved RAG results: {len(results)} items")
                    
                    # Print the raw RAG results structure
                    print("=== RAW RAG RESULTS STRUCTURE ===")
                    import json
                    try:
                        # Print full results like Claude does
                        print(json.dumps(results, indent=2))
                    except Exception as e:
                        print(f"Could not serialize RAG results to JSON: {str(e)}")
                    print("=== END RAW RAG RESULTS STRUCTURE ===")
                    
                    # Print the SQL context if available
                    if sql_context:
                        print("=== RAG SQL CONTEXT ===")
                        try:
                            print(json.dumps(sql_context, indent=2))
                        except Exception as e:
                            print(f"Could not serialize SQL context to JSON: {str(e)}")
                        print("=== END RAG SQL CONTEXT ===")
                    
                    # Group results by table
                    table_dict = {}
                    for item in results:
                        # Check if item is a dictionary
                        if not isinstance(item, dict):
                            print(f"Skipping non-dict item in RAG results: {type(item)}")
                            continue
                            
                        table_name = item.get("table_name")
                        if not table_name:
                            print("Missing table_name in RAG result item, skipping")
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
                    print(f"Retrieved {len(tables)} tables from RAG")
                    
                    # Print the RAG embeddings that will go into the prompt (like Claude does)
                    print("=== RAG EMBEDDINGS START ===")
                    for table in tables:
                        print(f"Table: {table['name']}")
                        for column in table['columns']:
                            print(f"  - {column['name']} ({column['type']}): {column['description']}")
                    print("=== RAG EMBEDDINGS END ===")
                else:
                    print(f"RAG query failed: {message}")
                    raise Exception(f"RAG query failed: {message}")
                
                # Check if we need to fall back to the full dictionary
                if not tables or len(tables) == 0:
                    print("RAG results empty or invalid, falling back to full dictionary")
                    df = load_data_dictionary(data_dictionary_path)
                    tables = format_data_dictionary(df)
            except Exception as e:
                print(f"Error accessing RAG functionality directly: {str(e)}")
                print("Falling back to full dictionary due to RAG error")
                df = load_data_dictionary(data_dictionary_path)
                tables = format_data_dictionary(df)
        except Exception as e:
            # Error handling
            print(f"Error using RAG: {str(e)}, falling back to full dictionary")
            df = load_data_dictionary(data_dictionary_path)
            tables = format_data_dictionary(df)
    else:
        # Use traditional full dictionary approach
        print(f"Using full dictionary from {data_dictionary_path}")
        df = load_data_dictionary(data_dictionary_path)
        tables = format_data_dictionary(df)
    
    # Generate prompt and SQL query with SQL context if available
    print(f"SQL context before prompt generation: {type(function_sql_context)}")
    if function_sql_context and isinstance(function_sql_context, dict):
        print(f"SQL context has keys: {function_sql_context.keys()}")
        if 'recommended_tables' in function_sql_context and isinstance(function_sql_context['recommended_tables'], list):
            print(f"Number of recommended tables: {len(function_sql_context['recommended_tables'])}")
            for i, table in enumerate(function_sql_context['recommended_tables'][:3]):
                if isinstance(table, dict) and 'full_name' in table and 'relevance_score' in table:
                    print(f"Recommended table {i+1}: {table['full_name']} (score: {table['relevance_score']:.2f})")
                else:
                    print(f"Recommended table {i+1}: {table} (invalid format)")
        elif 'recommended_tables' in function_sql_context:
            print(f"Recommended tables has invalid format: {type(function_sql_context['recommended_tables'])}")
        else:
            print("No recommended_tables key in SQL context")
    
    # Generate prompt with tables and query with SQL context if available
    prompt = generate_sql_prompt(tables, query, limit_rows=limit_rows, include_charts=include_charts, sql_context=function_sql_context)
    
    # Log only the prompt length to avoid excessive logging
    print(f"Generated prompt for OpenAI with length: {len(prompt)} characters")
    
    result = generate_sql_query(api_key, prompt, model=model, query_text=query, log_tokens=log_tokens, include_charts=include_charts)
    
    # Ensure 'sql' key always exists and is a string
    if 'sql' not in result or result['sql'] is None or not isinstance(result['sql'], str):
        result['sql'] = "SELECT 'No valid SQL query was generated' AS message"
    
    # Multi-layer JSON parsing similar to Claude and Gemini implementations
    # Layer 1: Check if the SQL is actually a JSON object and extract the SQL string
    if isinstance(result['sql'], str) and result['sql'].strip().startswith('{') and result['sql'].strip().endswith('}'): 
        try:
            # Try to parse as JSON
            sql_json = json.loads(result['sql'])
            if isinstance(sql_json, dict) and 'sql' in sql_json:
                # Extract the actual SQL from the JSON
                result['sql'] = sql_json['sql']
                print("[OpenAI Query Generator] Extracted SQL from JSON response - Layer 1")
        except json.JSONDecodeError as e:
            # If it's not valid JSON, try regex extraction as fallback
            print(f"[OpenAI Query Generator] SQL string looks like JSON but couldn't be parsed: {str(e)}")
            
            # Fallback regex extraction similar to Claude implementation
            sql_pattern = re.search(r'"sql"\s*:\s*"([^"]+)"', result['sql'])
            if sql_pattern:
                extracted_sql = sql_pattern.group(1)
                # Unescape any escaped quotes
                extracted_sql = extracted_sql.replace("\\\"", '"')
                result['sql'] = extracted_sql
                print("[OpenAI Query Generator] Extracted SQL using regex fallback")
    
    # Layer 2: Check if the extracted SQL is still a JSON object (nested JSON)
    if isinstance(result['sql'], str) and result['sql'].strip().startswith('{') and result['sql'].strip().endswith('}'): 
        try:
            # Try to parse as JSON again (handles double-nested JSON)
            nested_sql_json = json.loads(result['sql'])
            if isinstance(nested_sql_json, dict) and 'sql' in nested_sql_json:
                # Extract the actual SQL from the nested JSON
                result['sql'] = nested_sql_json['sql']
                print("[OpenAI Query Generator] Extracted SQL from nested JSON response - Layer 2")
        except json.JSONDecodeError:
            # If it's not valid JSON at this point, keep it as is
            pass
    
    # Layer 3: If SQL still doesn't look like SQL, try to extract SQL pattern
    if isinstance(result['sql'], str) and not re.search(r'(?:SELECT|WITH)\s+', result['sql'], re.IGNORECASE):
        # Try to find a SQL statement by looking for SELECT or WITH
        sql_match = re.search(r'(?:WITH|SELECT)\s+[\s\S]*?(?:;|$)', result['sql'], re.IGNORECASE)
        if sql_match:
            result['sql'] = sql_match.group(0).strip()
            print("[OpenAI Query Generator] Extracted SQL using pattern matching - Layer 3")
    
    # Process escape sequences in the SQL query
    # This handles cases where the query contains literal \n instead of actual newlines
    if '\\n' in result['sql']:
        result['sql'] = result['sql'].encode().decode('unicode_escape')
        print("[OpenAI Query Generator] Processed SQL with escape sequences")
    
    # Add additional info to the result
    result.update({
        "query": query,
        "timestamp": datetime.datetime.now().isoformat(),
        "data_dictionary": data_dictionary_path,
        "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000
    })
    
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
