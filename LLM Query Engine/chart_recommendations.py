"""
Chart Recommendations for SQL Query Results

This module analyzes SQL query results and provides appropriate chart recommendations
based on the data structure and content.
"""
import pandas as pd
import re
from typing import Dict, Any, List, Optional, Union


def smart_chart_selector(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Select appropriate chart types based on the data characteristics using a rules-based approach.
    
    This function analyzes the structure and content of a DataFrame and recommends
    appropriate chart types based on the following strict chart rules:
    
    Pie Chart Rules:
    - At least 2 columns: Category and Value
    - Values must be numbers and positive
    - Use less than 8 categories
    - Values must represent parts of a whole (100%)

    Bar Chart Rules:
    - 2 columns: Category and Value
    - Category = text, Value = number
    - Use to compare different items

    Line Chart Rules:
    - 2 columns: Time (or ordered values) and Numeric
    - Time must be in order
    - Use to show trends over time

    Scatter Plot Rules:
    - 2 numeric columns: X and Y
    - Each row is a point
    - Use to find patterns or relationships

    Area Chart Rules:
    - Same as line chart
    - No negative values
    - Shows totals over time

    Histogram Rules:
    - Only 1 numeric column
    - Shows how values are distributed into bins

    Box Plot Rules:
    - 1 numeric column, optional grouping column
    - Use to show spread (min, Q1, median, Q3, max)

    Table View Rules (for purely categorical data):
    - When data contains only categorical columns (no numeric values)
    - When data is too complex for visualization
    
    Args:
        df: DataFrame containing the query results
        
    Returns:
        List of chart recommendations with reasons for each selection
    """
    recommendations = []
    
    # Get column types for analysis
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_columns = [col for col in df.columns if is_date_column(df[col])]
    
    # Extract unique values and row counts for rule evaluation
    row_count = len(df)
    column_count = len(df.columns)
    
    # Check if we have purely categorical data with no numeric columns
    if not numeric_columns:
        # For purely categorical data, recommend a table view
        recommendations.append({
            "chart_type": "table",
            "reasoning": "Data contains only categorical values without numeric measures. A table is the most appropriate way to display this information.",
            "priority": 1,
            "chart_config": {
                "title": "Data Table View",
                "x_axis": None,  # No axes for table view
                "y_axis": None,
                "chart_library": "table",
                "data_config": {
                    "columns": df.columns.tolist(),
                    "sortable": True,
                    "searchable": True,
                    "paginated": True
                },
                "display_config": {
                    "show_row_numbers": True,
                    "highlight_duplicates": False,
                    "alternating_rows": True
                }
            }
        })
        
        # Add a suggestion for query modification
        recommendations.append({
            "chart_type": "suggestion",
            "reasoning": "To create meaningful charts, consider adding numeric measures like COUNT, SUM, or aggregations.",
            "priority": 2,
            "chart_config": {
                "title": "Suggested Query Modifications",
                "x_axis": None,
                "y_axis": None,
                "chart_library": "suggestion",
                "suggestions": [
                    f"Add COUNT(*) to show frequency of each {', '.join(categorical_columns[:2])} combination",
                    "Include sales amounts, quantities, or other numeric measures",
                    f"Group by {categorical_columns[0] if categorical_columns else 'category'} to show distributions",
                    "Add time dimensions for trend analysis"
                ]
            }
        })
        
        # Return early since we can't create meaningful charts without numeric data
        return recommendations
    
    # Extract unique values and row counts for rule evaluation
    row_count = len(df)
    
    # Track priority order for recommendations
    priority = 1
    
    # Check for pie chart conditions
    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        # Get unique values in the first categorical column
        first_cat_col = categorical_columns[0]
        unique_categories = df[first_cat_col].nunique()
        first_num_col = numeric_columns[0]
        
        # Pie Chart Rules:
        # - At least 2 columns: Category and Value
        # - Values must be numbers and positive
        # - Use less than 8 categories
        # - Values must represent parts of a whole (100%)
        if 2 <= unique_categories <= 7 and df[first_num_col].min() >= 0:
            # Check if values can represent parts of a whole
            total_sum = df[first_num_col].sum()
            if total_sum > 0:  # Ensure we have positive values to create a meaningful pie chart
                recommendations.append({
                    "chart_type": "pie",
                    "reasoning": f"Pie chart shows distribution of {first_num_col} across {first_cat_col} categories ({unique_categories} distinct values)",
                    "priority": 1 if unique_categories <= 5 else 2,  # Lower priority for more categories
                    "chart_config": {
                        "title": f"{first_num_col} Distribution by {first_cat_col}",
                        "x_axis": first_cat_col,
                        "y_axis": first_num_col,
                        "color_by": first_cat_col,
                        "aggregate_function": "SUM",
                        "chart_library": "plotly",
                        "additional_config": {
                            "show_legend": True,
                            "show_labels": True,
                            "label_position": "outside"
                        }
                    }
                })
    
    # Check for bar chart conditions
    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        # Bar Chart Rules:
        # - 2 columns: Category and Value
        # - Category = text, Value = number
        # - Use to compare different items
        first_cat_col = categorical_columns[0]
        first_num_col = numeric_columns[0]
        
        recommendations.append({
            "chart_type": "bar",
            "reasoning": f"Bar chart compares {first_num_col} values across {first_cat_col} categories",
            "priority": 1,
            "chart_config": {
                "title": f"{first_num_col} by {first_cat_col}",
                "x_axis": first_cat_col,
                "y_axis": first_num_col,
                "color_by": first_cat_col,
                "aggregate_function": "SUM",
                "chart_library": "plotly",
                "additional_config": {
                    "orientation": "vertical",
                    "show_legend": True
                }
            }
        })
    
    # Check for line chart conditions
    if date_columns and numeric_columns:
        # Line Chart Rules:
        # - 2 columns: Time (or ordered values) and Numeric
        # - Time must be in order
        # - Use to show trends over time
        first_date_col = date_columns[0]
        first_num_col = numeric_columns[0]
        
        recommendations.append({
            "chart_type": "line",
            "reasoning": f"Line chart shows {first_num_col} trend over {first_date_col}",
            "priority": 1,
            "chart_config": {
                "title": f"{first_num_col} Over Time",
                "x_axis": first_date_col,
                "y_axis": first_num_col,
                "color_by": None,
                "aggregate_function": "NONE",
                "chart_library": "plotly",
                "additional_config": {
                    "show_markers": row_count < 30,  # Only show markers for smaller datasets
                    "connect_gaps": True
                }
            }
        })
    
    # Check for scatter plot conditions
    if len(numeric_columns) >= 2:
        # Scatter Plot Rules:
        # - 2 numeric columns: X and Y
        # - Each row is a point
        # - Use to find patterns or relationships
        first_num_col = numeric_columns[0]
        second_num_col = numeric_columns[1]
        
        recommendations.append({
            "chart_type": "scatter",
            "reasoning": f"Scatter plot shows relationship between {first_num_col} and {second_num_col}",
            "priority": 2,
            "chart_config": {
                "title": f"{first_num_col} vs {second_num_col}",
                "x_axis": first_num_col,
                "y_axis": second_num_col,
                "color_by": categorical_columns[0] if categorical_columns else None,
                "aggregate_function": "NONE",
                "chart_library": "plotly",
                "additional_config": {
                    "show_trendline": True,
                    "trendline_type": "ols"  # Ordinary Least Squares regression line
                }
            }
        })
    
    # Check for area chart conditions
    if date_columns and numeric_columns:
        # Area Chart Rules:
        # - Same as line chart
        # - No negative values
        # - Shows totals over time
        first_date_col = date_columns[0]
        first_num_col = numeric_columns[0]
        
        if df[first_num_col].min() >= 0:  # No negative values
            recommendations.append({
                "chart_type": "area",
                "reasoning": f"Area chart shows {first_num_col} trend over {first_date_col} with emphasis on volume",
                "priority": 2,
                "chart_config": {
                    "title": f"{first_num_col} Volume Over Time",
                    "x_axis": first_date_col,
                    "y_axis": first_num_col,
                    "color_by": None,
                    "aggregate_function": "NONE",
                    "chart_library": "plotly",
                    "additional_config": {
                        "fill": "tozeroy",
                        "connect_gaps": True
                    }
                }
            })
    
    # Check for histogram conditions
    if numeric_columns:
        # Histogram Rules:
        # - Only 1 numeric column
        # - Shows how values are distributed into bins
        first_num_col = numeric_columns[0]
        
        recommendations.append({
            "chart_type": "histogram",
            "reasoning": f"Histogram shows the distribution of {first_num_col} values",
            "priority": 3,
            "chart_config": {
                "title": f"Distribution of {first_num_col}",
                "x_axis": first_num_col,
                "y_axis": "count",
                "color_by": None,
                "aggregate_function": "COUNT",
                "chart_library": "plotly",
                "additional_config": {
                    "bin_count": min(20, max(5, int(row_count / 10)))  # Dynamic bin count based on data size
                }
            }
        })
    
    # Check for boxplot conditions
    if numeric_columns:
        # Box Plot Rules:
        # - 1 numeric column, optional grouping column
        # - Use to show spread (min, Q1, median, Q3, max)
        first_num_col = numeric_columns[0]
        
        box_config = {
            "chart_type": "boxplot",
            "reasoning": f"Box plot shows the statistical distribution of {first_num_col}",
            "priority": 3,
            "chart_config": {
                "title": f"Statistical Distribution of {first_num_col}",
                "x_axis": None,
                "y_axis": first_num_col,
                "color_by": None,
                "aggregate_function": "NONE",
                "chart_library": "plotly",
                "additional_config": {
                    "show_points": row_count < 100,  # Only show individual points for smaller datasets
                    "box_mean": True  # Show mean as a dashed line
                }
            }
        }
        
        # If we have a categorical column, use it for grouping
        if categorical_columns:
            first_cat_col = categorical_columns[0]
            unique_categories = df[first_cat_col].nunique()
            
            # Only use categorical grouping if there aren't too many categories
            if unique_categories <= 10:
                box_config["reasoning"] = f"Box plot shows the statistical distribution of {first_num_col} grouped by {first_cat_col}"
                box_config["chart_config"]["title"] = f"Statistical Distribution of {first_num_col} by {first_cat_col}"
                box_config["chart_config"]["x_axis"] = first_cat_col
                box_config["chart_config"]["color_by"] = first_cat_col
        
        recommendations.append(box_config)
    
    # Check for mixed/combo chart conditions
    if len(categorical_columns) >= 1 and len(numeric_columns) >= 2:
        # Mixed/Combo Chart Rules:
        # - Multiple numeric columns with potentially different scales
        # - Combines different chart types (bar, line) for optimal visualization
        # - Variables with similar scales should use the same chart type
        first_cat_col = categorical_columns[0]
        
        # Group numeric columns by their scale type
        monetary_columns = []
        count_columns = []
        other_columns = []
        
        # Simple heuristic to identify column types based on name
        for col in numeric_columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['sales', 'revenue', 'profit', 'price', 'cost', 'amount', '$']):
                monetary_columns.append(col)
            elif any(term in col_lower for term in ['count', 'quantity', 'number', 'total', 'sum']):
                count_columns.append(col)
            else:
                other_columns.append(col)
        
        # If we couldn't identify types clearly, use the first two columns
        if not monetary_columns and not count_columns:
            monetary_columns = [numeric_columns[0]]
            if len(numeric_columns) > 1:
                count_columns = [numeric_columns[1]]
        
        # Create series configuration
        series = []
        
        # Add monetary columns as bars on primary axis
        for col in monetary_columns:
            series.append({
                "column": col,
                "type": "bar",
                "axis": "primary"
            })
        
        # Add count columns as lines on secondary axis
        for col in count_columns:
            series.append({
                "column": col,
                "type": "line",
                "axis": "secondary"
            })
        
        # Add any remaining columns
        for col in other_columns:
            series.append({
                "column": col,
                "type": "line" if len(monetary_columns) > 0 else "bar",
                "axis": "secondary" if len(monetary_columns) > 0 else "primary"
            })
        
        # Only add recommendation if we have at least two series
        if len(series) >= 2:
            recommendations.append({
                "chart_type": "mixed",
                "reasoning": f"Mixed chart shows multiple metrics with appropriate visualizations. Similar scale variables use the same chart type.",
                "priority": 2,
                "chart_config": {
                    "title": f"Multi-metric Analysis by {first_cat_col}",
                    "x_axis": first_cat_col,
                    "series": series,
                    "chart_library": "plotly",
                    "additional_config": {
                        "show_legend": True,
                        "bar_mode": "group"
                    }
                }
            })
    
    # Sort recommendations by priority
    recommendations.sort(key=lambda x: x['priority'])
    
    return recommendations


def analyze_query_results_for_charts(
    query: str, 
    query_output: Union[pd.DataFrame, List[Dict[str, Any]], None]
) -> Dict[str, Any]:
    """
    Analyze SQL query results and recommend appropriate charts.
    
    Args:
        query: The SQL query that was executed
        query_output: The results of the query execution (DataFrame or list of dicts)
        
    Returns:
        Dictionary with chart recommendations and any error messages
    """
    # Handle None or empty list results
    if query_output is None:
        return {
            "chart_recommendations": [],
            "chart_error": "No chart recommendations available - results are empty or null"
        }
    
    # Handle string inputs (error messages or raw text)
    if isinstance(query_output, str):
        return {
            "chart_recommendations": [],
            "chart_error": f"No chart recommendations available - received string instead of data: {query_output[:100]}..."
        }
        
    # Convert to DataFrame if it's a list of dictionaries
    if isinstance(query_output, list):
        try:
            df = pd.DataFrame(query_output)
        except Exception as e:
            return {
                "chart_recommendations": [],
                "chart_error": f"No chart recommendations available - unable to process results: {str(e)}"
            }
    else:
        # At this point, query_output should be a DataFrame
        if not isinstance(query_output, pd.DataFrame):
            return {
                "chart_recommendations": [],
                "chart_error": f"No chart recommendations available - expected DataFrame but got {type(query_output)}"
            }
        df = query_output
    
    # Handle empty DataFrame case
    if df.empty:
        return {
            "chart_recommendations": [],
            "chart_error": "No chart recommendations available - results are empty"
        }
    
    # Handle single value result case
    if len(df) == 1 and len(df.columns) == 1:
        # Single scalar value - not suitable for charting
        return {
            "chart_recommendations": [],
            "chart_error": "No chart recommendations available - results contain only a single value"
        }
    
    # Check if we have any numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        # Get recommendations (will include table view and suggestions)
        recommendations = smart_chart_selector(df)
        
        # Create a more descriptive error message based on the data structure
        column_names = df.columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        error_message = (
            f"No numeric measures available for chart visualization. "
            f"The data contains only categorical columns: {', '.join(categorical_columns)}. "
            f"Consider adding COUNT, SUM, or other aggregations to create meaningful charts."
        )
        
        return {
            "chart_recommendations": recommendations,
            "chart_error": error_message
        }
        
    # Use the smart chart selector to get recommendations
    recommendations = smart_chart_selector(df)
    
    # If no chart types were recommended after analysis
    if not recommendations:
        return {
            "chart_recommendations": [],
            "chart_error": "No suitable chart recommendations found for this data structure"
        }
    
    # Return chart recommendations only
    return {
        "chart_recommendations": recommendations
    }


def extract_data_insights(df: pd.DataFrame, numeric_columns: List[str]) -> List[str]:
    """Extract key insights from the data."""
    insights = []
    
    for col in numeric_columns[:2]:  # Limit to first two numeric columns for brevity
        if df[col].count() > 0:
            max_val = df[col].max()
            min_val = df[col].min()
            avg_val = df[col].mean()
            
            # For the first column, add context from categorical column if available
            if col == numeric_columns[0]:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    cat_col = cat_cols[0]
                    max_row = df.loc[df[col].idxmax()]
                    max_context = max_row[cat_col] if cat_col in max_row else None
                    if max_context:
                        insights.append(f"Highest {col}: {format_number(max_val)} ({max_context})")
                    else:
                        insights.append(f"Highest {col}: {format_number(max_val)}")
                else:
                    insights.append(f"Highest {col}: {format_number(max_val)}")
            else:
                insights.append(f"Highest {col}: {format_number(max_val)}")
                
            insights.append(f"Average {col}: {format_number(avg_val)}")
            insights.append(f"Range: {format_number(min_val)} - {format_number(max_val)}")
    
    # Add row count insight
    insights.append(f"Total records: {len(df)}")
    
    return insights


def format_number(num: float) -> str:
    """Format number for display in insights."""
    if abs(num) >= 1000:
        return f"{num:,.0f}"
    elif abs(num) >= 1:
        return f"{num:.2f}"
    else:
        return f"{num:.4f}"


def is_date_column(series: pd.Series) -> bool:
    """Determine if a series contains date/time data."""
    # Check if it's already a datetime type
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    # For string columns, check if they match date patterns
    if series.dtype == object:
        # Sample first non-null value
        sample = next((x for x in series if x is not None and x != ''), None)
        if sample:
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}', # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}', # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}', # DD-MM-YYYY
                r'^\d{1,2}\s[A-Za-z]{3}\s\d{4}' # 1 Jan 2023
            ]
            return any(re.match(pattern, str(sample)) for pattern in date_patterns)
    
    return False


def enhanced_api_response(
    prompt: str,
    query: str,
    query_output: Any,
    model: str,
    token_usage: Dict[str, int],
    success: bool,
    error_message: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    user_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an enhanced API response that includes chart recommendations.
    
    This function preserves all original fields in the API response and adds chart recommendations.
    
    Args:
        prompt: The original NL prompt
        query: The SQL query
        query_output: Results of the query execution
        model: The LLM model used
        token_usage: Dictionary of token usage
        success: Whether query execution was successful
        error_message: Error message if any
        execution_time_ms: Execution time in ms
        user_hint: Hint for the user
        
    Returns:
        Enhanced API response with chart recommendations
    """
    # Construct the base response with all original fields
    response = {
        "prompt": prompt,
        "query": query,
        "query_output": query_output,
        "model": model,
        "token_usage": token_usage,
        "success": success,
        "error_message": error_message,
        "execution_time_ms": execution_time_ms,
        "user_hint": user_hint
    }
    
    # Only add chart recommendations if query was successful
    if success and query_output is not None:
        # Analyze results and get chart recommendations
        chart_data = analyze_query_results_for_charts(query, query_output)
        
        # Add chart recommendations to response
        response["chart_recommendations"] = chart_data["chart_recommendations"]
        
        # Add chart error message if present
        if "chart_error" in chart_data:
            response["chart_error"] = chart_data["chart_error"]
    else:
        # Add empty chart data if query failed
        response["chart_recommendations"] = []
        response["chart_error"] = "No chart recommendations available - query execution failed"
            
    return response


def process_api_request_with_charts(
    query_generator_fn,
    query: str,
    data_dictionary_path: str,
    execute: bool = True,
    limit_rows: int = 100,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process an API request and include chart recommendations in the response.
    
    Args:
        query_generator_fn: Function to generate SQL from NL query
        query: The natural language query
        data_dictionary_path: Path to data dictionary
        execute: Whether to execute the query
        limit_rows: Maximum rows to return
        model: Model to use for query generation
        **kwargs: Additional keyword arguments for the query generator function
        
    Returns:
        Enhanced API response with chart recommendations
    """
    # Call the original query generator function
    result = query_generator_fn(
        query=query,
        data_dictionary_path=data_dictionary_path,
        execute=execute,
        limit_rows=limit_rows,
        model=model,
        **kwargs
    )
    
    # Extract required fields for enhanced response
    prompt = query
    sql_query = result.get("sql", "")
    query_output = result.get("results", None)
    success = result.get("success", False)
    error_message = result.get("error", None) or result.get("error_execution", None)
    execution_time_ms = result.get("execution_time_ms", None)
    user_hint = result.get("user_hint", None)
    
    # Construct token usage dictionary
    token_usage = {
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0)
    }
    
    # Generate enhanced response with chart recommendations
    enhanced_result = enhanced_api_response(
        prompt=prompt,
        query=sql_query,
        query_output=query_output,
        model=model or result.get("model", "unknown"),
        token_usage=token_usage,
        success=success,
        error_message=error_message,
        execution_time_ms=execution_time_ms,
        user_hint=user_hint
    )
    
    # Preserve any other fields from the original result
    for key, value in result.items():
        if key not in enhanced_result:
            enhanced_result[key] = value
    
    return enhanced_result
