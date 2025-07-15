"""
LLM Query Generator - A simple module to convert natural language to SQL using OpenAI
"""
import os
import pandas as pd
import openai
import datetime
import json
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


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str, limit_rows: int = 100, include_charts: bool = False) -> str:
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
    chart_instructions = """
After generating the SQL query, recommend appropriate chart types for visualizing the results. Follow these rules:

1. Analyze the query to determine if it returns numeric columns or only categorical data
2. Recommend charts that match the data structure based on these rules:

   Pie Chart:
   - Need: Category column + positive numeric Value column
   - Best for: < 8 categories, values representing parts of a whole (100%)

   Bar Chart:
   - Need: Category column + numeric Value column(s)
   - Best for: Comparing different items
   - For multi-series with different scales (e.g., millions vs hundreds), use secondary axis and side-by-side bars
   - IMPORTANT: When showing multiple numeric columns, display them as side-by-side bars, not stacked

   Line Chart:
   - Need: Time/ordered column + numeric Value column(s)
   - Best for: Trends over time
   - For multi-series with different scales, use secondary axis

   Scatter Plot:
   - Need: 2 numeric columns (X and Y)
   - Best for: Finding patterns/relationships
   - For multi-series with different scales, consider secondary axis

   Area Chart:
   - Need: Time/ordered column + positive numeric values
   - Best for: Showing totals over time
   - For multi-series with different scales, use secondary axis

   Mixed/Combo Chart:
   - Need: Multiple numeric columns with potentially different scales
   - Best for: Combining different chart types (bar, line) for optimal visualization
   - For numeric columns of similar types, prefer side-by-side bars with dual y-axes
   - For count/quantity measures with revenue/sales measures, use bars + line with secondary axis
   - Assign columns to primary/secondary axis based on magnitude and semantics
   - Similar scales (e.g., SALES and PROFIT in dollars): same axis
   - Different scales (e.g., SALES in millions vs COUNT in hundreds): different axes

Histogram: 1 numeric column, shows value distribution
Box Plot: 1 numeric column + optional grouping, shows statistical spread
Table View: For purely categorical data or complex data

For single numeric values KPI Card: 
   Display as minimal cards with bold label at top, large formatted number below, no icons, clean white background, centered text only.

For purely categorical data (no numeric columns): Recommend a table view and suggest query modifications

For each of the 2 chart recommendations, provide:
   - chart_type: The type of chart (pie, bar, line, scatter, area, histogram, boxplot, table, suggestion,KPI Card)
   - reasoning: Brief explanation of why this chart type is appropriate based on the rules
   - priority: Importance ranking (1 = highest)
   - chart_config: Detailed configuration including:
     * title: Descriptive chart title
     * x_axis: Column to use for x-axis (can be a single column name)
     * y_axis: Column(s) to use for y-axis - this can be a single column name OR an array of column names for multi-series charts
     * color_by: Column to use for segmentation/colors (if applicable)
     * aggregate_function: Any aggregation needed (SUM, AVG, etc.)
     * chart_library: Recommended visualization library (plotly)
     * additional_config: Other relevant settings like orientation, legend, etc.

CRITICAL: HANDLING PURELY CATEGORICAL DATA
If the query will return ONLY categorical columns (no numeric measures):
1. DO NOT recommend bar, pie, line, or other charts that require numeric data
2. Instead, recommend a table view as the primary visualization
3. Add a "suggestion" recommendation with query modification ideas
4. Include a chart_error field in your response explaining the issue

SCALE DETECTION AND AXIS ASSIGNMENT:
- Analyze magnitude ranges of each numeric column
- Use secondary axis for vastly different scales (e.g., millions vs hundreds)
- Group by semantics: monetary values together, counts/quantities separate
- Set use_secondary_axis: true and specify secondary_axis_columns
- Enable scale_detection: true for frontend optimization

DISCRETE CATEGORICAL X-AXIS VALUES:
- For time-based data with discrete intervals (quarters, years, months), ensure values are formatted as strings (e.g., 'Q1', 'Q2', '2023', 'Jan')
- For numeric IDs or codes that should be discrete categories, cast them as strings in SQL (e.g., CAST(quarter AS VARCHAR) AS QUARTER)
- Never use numeric representations for categorical data that could be interpreted as continuous (e.g., use 'Q1' not 1)



Example with scale detection:
{
  "sql": "SELECT MONTH_NAME, SUM(SALES) AS TOTAL_SALES, COUNT(ORDER_ID) AS ORDER_COUNT FROM ORDERS GROUP BY MONTH_NAME ORDER BY MONTH_NUMBER;",
  "chart_recommendations": [{
    "chart_type": "mixed",
    "reasoning": "TOTAL_SALES (millions) and ORDER_COUNT (hundreds) have different scales",
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

Your response must be a valid JSON object with the following structure:
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
        "y_axis": ["column_name1", "column_name2"],  // Can be a string or array of strings for multi-series
        "color_by": "column_name",
        "aggregate_function": "NONE|SUM|AVG|etc",
        "chart_library": "plotly",
        "additional_config": {
          "show_legend": true,
          "orientation": "vertical|horizontal",
          "use_secondary_axis": true,  // For dual-axis charts
          "secondary_axis_columns": ["column_name2"],  // Columns to plot on secondary axis
          "scale_detection": true,  // Enable automatic scale detection between series
          "scale_threshold": 2  // Log10 scale difference threshold to trigger secondary axis (default 2 = 100x difference)
        }
      }
    }
  
  ]
    "chart_error": "No numeric measures available for chart visualization. The data contains only categorical columns. Consider adding COUNT, SUM, or other aggregations to create meaningful charts."

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
8. Add helpful SQL comments to explain complex parts of the query
9. CRITICAL: Follow these row limit rules EXACTLY:
    a. If the user explicitly specifies a number in their query (e.g., "top 5", "first 10"), use EXACTLY that number in the LIMIT clause
    b. Otherwise, limit results to {limit_rows} rows
    c. NEVER override a user-specified limit with a different number
10. If the query is unclear, include this comment: -- Please clarify: [specific aspect]

Generate a SQL query for: {query}{chart_instructions}
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
                      model_provider: str = "openai", limit_rows: int = 100, include_charts: bool = False) -> Dict[str, Any]:
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
    
    # Process data dictionary
    df = load_data_dictionary(data_dictionary_path)
    tables = format_data_dictionary(df)
    
    # Generate prompt with tables and query
    prompt = generate_sql_prompt(tables, query, limit_rows=limit_rows, include_charts=include_charts)
    result = generate_sql_query(api_key, prompt, model=model, query_text=query, log_tokens=log_tokens, include_charts=include_charts)
    
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
