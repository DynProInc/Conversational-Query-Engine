"""
Utilities for generating chart recommendations for edited SQL queries
"""
import os
import json
import anthropic
import openai
from typing import Dict, List, Any, Optional

# Import client manager for client-specific credentials
from config.client_manager import client_manager

# Chart instructions for consistency with claude_query_generator
chart_instructions = """
        ### Chart Type Definitions

        **Bar Charts**: Categorical comparisons (≤15 categories)
        - Required: 1 categorical column + 1+ numeric measures
        - Best for: Category comparisons, time periods
        - Multi-scale: Convert to mixed chart if needed

        **Line Charts**: Trends over continuous ranges
        - Required: 1 continuous axis + 1+ numeric measures  
        - Best for: Time trends, continuous data
        - Multi-scale: Use dual Y-axis configuration

        **Pie Charts**: Composition/parts of whole (≤7 categories)
        - Required: 1 categorical column (≤7 values) + 1 numeric measure
        - Best for: Proportions, part-to-whole relationships
        - Multi-scale: Not applicable

        **Scatter Plots**: Relationships between variables
        - Required: 2+ numeric measures
        - Best for: Correlation analysis, distribution patterns
        - Multi-scale: Use different axis scales

        **Area Charts**: Cumulative totals over time/categories
        - Required: 1 continuous axis + 1+ numeric measures
        - Best for: Cumulative trends, stacked compositions
        - Multi-scale: Convert to mixed chart if needed

        **Mixed/Combo Charts**: Multi-measure comparisons
        - Required: 1 shared axis + multiple measures (different scales)
        - Best for: Different but related metrics
        - Multi-scale: Primary purpose of this chart type

        **KPI Cards**: Single numeric values
        - Required: 1 numeric measure
        - Best for: Key metrics, minimal display
        - Multi-scale: Not applicable

        **Tables**: Categorical data without numeric measures
        - Required: Multiple categorical columns
        - Best for: Detailed data display
        - Multi-scale: Not applicable

        **Histogram Charts**: Distribution analysis of continuous variables
        - Required: Numerical continuous data only (no categorical)
        - Best for: Frequency/spread/skewness/outliers in large datasets
        - Use auto-binning (Sturges/Freedman-Diaconis) for proper bin sizing
        - X-axis: value range, Y-axis: frequency/density
        - Ensure contiguous bins (no gaps)
        - Avoid overlapping distributions (use separate plots/density plots)
        - Skip for small datasets (use box/dot plots instead)

        **Box Plot Charts**: Distribution comparison between groups
        - Required: Numerical data (can group by categorical)
        - Best for: Comparing distributions, showing central tendency/spread/outliers
        - Box = IQR (Q1-Q3), line = median
        - Whiskers = Q1-1.5×IQR to Q3+1.5×IQR
        - Points beyond whiskers = outliers
        - Best for side-by-side comparisons
        - Consider combining with histograms/violin plots for distribution shape details

        ### Multi-Scale Detection (All Chart Types)
        **Scale Detection Rules**:
        - Trigger: Scale ratio ≥ 100x (larger_value / smaller_value ≥ 100)
        - Examples: Sales $50K vs Orders 25 (2000x ratio → secondary axis)
        - Action: Use secondary Y-axis or convert to mixed chart
        - if required Use normalization to bring scales together

        **Chart-Specific Scale Handling**:
        - **Bar/Line/Area**: Convert to mixed chart or use dual Y-axis
        - **Mixed**: Assign measures to appropriate axes
        - **Configuration**: Always include scale detection parameters

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

        **Example with Multi-Scale Detection**:
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

        

        IMPORTANT: DO NOT include any explanations, code blocks, or markdown formatting outside the JSON. Your entire response must be valid JSON that can be parsed directly.
        """

def generate_chart_recommendations_for_edited_sql(
    sql: str, 
    original_prompt: Optional[str] = None,
    model: Optional[str] = None,
    tables_context: Optional[str] = None,
    client_id: str = "mts"
) -> List[Dict[str, Any]]:
    """
    Generate chart recommendations for an edited SQL query using the same
    chart instructions as the original query generation
    
    Args:
        sql: The edited SQL query
        original_prompt: The original natural language query
        model: The model to use (will use env default if not specified)
        tables_context: Optional data dictionary context
        
    Returns:
        List of chart recommendations in the standard format
    """
    # Use the same model as the original query if provided, or default from client settings
    model_provider = None
    
    # Determine the model provider from the model name
    if model:
        # Handle generic provider names and specific model names
        if model.lower() == 'claude' or model.lower() == 'anthropic' or model.lower().startswith('claude-'):
            model_provider = 'anthropic'
            # If model is just 'claude' or 'anthropic', it's a provider name not a specific model
            if model.lower() in ['claude', 'anthropic']:
                # Just mark it as the provider, we'll get the actual model from client config or env
                model = 'anthropic'
        elif model.lower() == 'openai' or model.lower().startswith('gpt'):
            model_provider = 'openai'
        elif model.lower() == 'gemini' or model.lower() == 'google' or model.lower().startswith('gemini-'):
            model_provider = 'gemini'
    
    # If no model specified, use client default
    if not model_provider:
        # Default to anthropic/claude if no model specified
        model_provider = 'anthropic'
    
    # Create a specialized prompt for chart-only generation
    context = ""
    if tables_context:
        context += f"Consider this data dictionary when making recommendations:\n{tables_context}\n\n"
    
    if original_prompt:
        context += f"The original question was: '{original_prompt}'\n\n"
        
    prompt = f"""You are an expert data visualization specialist.

Based on this SQL query:

```sql
{sql}
```

{context}Generate appropriate chart recommendations for visualizing the results.

{chart_instructions}"""

    # Use the appropriate model API based on the model provider and client
    if model_provider == 'anthropic':
        try:
            # Get client-specific API key and model
            llm_config = client_manager.get_llm_config(client_id, 'anthropic')
            api_key = llm_config['api_key']
            
            # Handle the case when model is just "anthropic" (provider name)
            if model and model.lower() == 'anthropic':
                # Use the client's configured model from the client manager
                model_to_use = llm_config.get('model')
                print(f"Using client {client_id}'s configured Anthropic model: {model_to_use}")
                
                # If client doesn't have a model configured, use environment variable
                if not model_to_use:
                    model_to_use = os.getenv("ANTHROPIC_MODEL")
                    print(f"No Anthropic model configured for client {client_id}, using environment variable: {model_to_use}")
                    
                    # If environment variable is not set, raise error
                    if not model_to_use:
                        raise ValueError(f"No Anthropic model configured for client {client_id} and no ANTHROPIC_MODEL environment variable")
            else:
                # Use provided model if specified, otherwise use client default
                if model:
                    model_to_use = model
                else:
                    model_to_use = llm_config.get('model')
                    if not model_to_use:
                        model_to_use = os.getenv("ANTHROPIC_MODEL")
                        if not model_to_use:
                            raise ValueError("No Anthropic model configured and no ANTHROPIC_MODEL environment variable")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=model_to_use,
                max_tokens=2048,
                system="You are a data visualization expert helping generate chart recommendations.",
                messages=[{"role": "user", "content": prompt}]
            )
            
        except ValueError as e:
            # If client doesn't have anthropic config, fall back to global env
            print(f"Client {client_id} anthropic config error: {str(e)}. Falling back to environment variable.")
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                
            client = anthropic.Anthropic(api_key=api_key)
            
            # Handle the case when model is just "anthropic" or "claude" (provider name)
            if model and (model.lower() == 'anthropic' or model.lower() == 'claude'):
                # Try to use global ANTHROPIC_MODEL environment variable
                model_to_use = os.getenv("ANTHROPIC_MODEL")
                if model_to_use:
                    print(f"Using ANTHROPIC_MODEL from environment: {model_to_use}")
                else:
                    # If environment variable is not set, raise error with clear message
                    raise ValueError(f"Cannot use generic model name '{model}'. No specific model configured in ANTHROPIC_MODEL environment variable.")
            else:
                # Use provided model if specified, otherwise check environment variable
                if model and model.lower() not in ['anthropic', 'claude']:
                    model_to_use = model
                else:
                    model_to_use = os.getenv("ANTHROPIC_MODEL")
                    if not model_to_use:
                        raise ValueError("No specific model name provided and no ANTHROPIC_MODEL environment variable found")
            
            response = client.messages.create(
                model=model_to_use,
                max_tokens=2048,
                system="You are a data visualization expert helping generate chart recommendations.",
                messages=[{"role": "user", "content": prompt}]
            )
        
        try:
            # Extract JSON from response
            content = response.content[0].text
            
            # Parse JSON response (could be wrapped in a JSON object or just the array)
            try:
                # Try parsing as complete JSON object with chart_recommendations array
                full_json = json.loads(content)
                if "chart_recommendations" in full_json:
                    return full_json["chart_recommendations"]
            except:
                # Try finding and parsing just the chart recommendations array
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            
            # If we couldn't parse, return empty list
            print("Could not parse chart recommendations from model response")
            return []
            
        except Exception as e:
            print(f"Error parsing chart recommendations: {e}")
            return []
            
    elif model_provider == 'openai':
        try:
            # Get client-specific API key and model
            llm_config = client_manager.get_llm_config(client_id, 'openai')
            api_key = llm_config['api_key']
            
            # Handle the case when model is just "openai" (provider name)
            if model and model.lower() == 'openai':
                # Use the client's configured model from the client manager
                model_to_use = llm_config.get('model')
                print(f"Using client {client_id}'s configured OpenAI model: {model_to_use}")
                
                # If client doesn't have a model configured, use environment variable
                if not model_to_use:
                    model_to_use = os.getenv("OPENAI_MODEL")
                    print(f"No OpenAI model configured for client {client_id}, using environment variable: {model_to_use}")
                    
                    # If environment variable is not set, raise error
                    if not model_to_use:
                        raise ValueError(f"No OpenAI model configured for client {client_id} and no OPENAI_MODEL environment variable")
            else:
                # Use provided model if specified, otherwise use client default
                if model:
                    model_to_use = model
                else:
                    model_to_use = llm_config.get('model')
                    if not model_to_use:
                        model_to_use = os.getenv("OPENAI_MODEL")
                        if not model_to_use:
                            raise ValueError("No OpenAI model configured and no OPENAI_MODEL environment variable")
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_to_use,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert helping generate chart recommendations."},
                    {"role": "user", "content": prompt}
                ]
            )
        except ValueError as e:
            # If client doesn't have openai config, fall back to global env
            print(f"Client {client_id} openai config error: {str(e)}. Falling back to environment variable.")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            # Handle the case when model is just "openai" (provider name)
            if model and model.lower() == 'openai':
                # Try to use global OPENAI_MODEL environment variable
                model_to_use = os.getenv("OPENAI_MODEL")
                if model_to_use:
                    print(f"Using OPENAI_MODEL from environment: {model_to_use}")
                else:
                    # If environment variable is not set, raise error
                    raise ValueError("No OPENAI_MODEL environment variable found")
            else:
                # Use provided model if specified, otherwise check environment variable
                if model:
                    model_to_use = model
                else:
                    model_to_use = os.getenv("OPENAI_MODEL")
                    if not model_to_use:
                        raise ValueError("No model specified and no OPENAI_MODEL environment variable found")
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_to_use,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert helping generate chart recommendations."},
                    {"role": "user", "content": prompt}
                ]
            )
        
        try:
            content = response.choices[0].message.content
            
            # Parse JSON response (same logic as above)
            try:
                full_json = json.loads(content)
                if "chart_recommendations" in full_json:
                    return full_json["chart_recommendations"]
            except:
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            
            print("Could not parse chart recommendations from model response")
            return []
            
        except Exception as e:
            print(f"Error parsing chart recommendations: {e}")
            return []
    
    elif model_provider == 'gemini':
        try:
            # Get client-specific API key and model
            import google.generativeai as genai
            
            llm_config = client_manager.get_llm_config(client_id, 'gemini')
            api_key = llm_config['api_key']
            
            # Handle the case when model is just "gemini" (provider name)
            if model and model.lower() == 'gemini':
                # Use the client's configured model from the client manager
                model_to_use = llm_config.get('model')
                print(f"Using client {client_id}'s configured Gemini model: {model_to_use}")
                
                # If client doesn't have a model configured, use environment variable
                if not model_to_use:
                    model_to_use = os.getenv("GEMINI_MODEL")
                    print(f"No Gemini model configured for client {client_id}, using environment variable: {model_to_use}")
                    
                    # If environment variable is not set, raise error
                    if not model_to_use:
                        raise ValueError(f"No Gemini model configured for client {client_id} and no GEMINI_MODEL environment variable")
            else:
                # Use provided model if specified, otherwise use client default
                if model:
                    model_to_use = model
                else:
                    model_to_use = llm_config.get('model')
                    if not model_to_use:
                        model_to_use = os.getenv("GEMINI_MODEL")
                        if not model_to_use:
                            raise ValueError("No Gemini model configured and no GEMINI_MODEL environment variable")
            
            # Configure gemini
            genai.configure(api_key=api_key)
            
            # Create a gemini model instance
            gemini_model = genai.GenerativeModel(model_to_use)
            
            response = gemini_model.generate_content(
                contents=[
                    {"role": "user", "parts": [prompt]}
                ],
            )
            
            # Extract content from response
            content = response.text
            
            # Parse JSON response similar to other models
            try:
                full_json = json.loads(content)
                if "chart_recommendations" in full_json:
                    return full_json["chart_recommendations"]
            except:
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            
            # If we couldn't parse, return empty list
            print("Could not parse chart recommendations from Gemini response")
            return []
                
        except ValueError as e:
            print(f"Client {client_id} gemini config error: {str(e)}")
            return []
        except Exception as e:
            print(f"Error with Gemini API: {str(e)}")
            return []
    
    # If we get here, we don't have support for the requested model provider
    print(f"Unsupported model provider: {model_provider}")
    return []
