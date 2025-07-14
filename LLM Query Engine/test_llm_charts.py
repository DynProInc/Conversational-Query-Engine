"""
Test script for the updated LLM-generated chart recommendations functionality.

This script demonstrates how to use the updated API with LLM-generated chart recommendations,
where charts are generated directly by the LLM along with the SQL query, rather than
through post-processing the query results.
"""
import os
import json
from dotenv import load_dotenv
from nlq_to_snowflake import nlq_to_snowflake
from nlq_to_snowflake_claude import nlq_to_snowflake_claude
from nlq_to_snowflake_gemini import nlq_to_snowflake_gemini

# Load environment variables
load_dotenv()

def print_chart_recommendations(result):
    """Pretty print the chart recommendations and data insights"""
    print("\n===== CHART RECOMMENDATIONS =====")
    
    if not result.get("chart_recommendations"):
        print("No chart recommendations available.")
        return
    
    # Print each chart recommendation
    for i, chart in enumerate(result.get("chart_recommendations", [])):
        print(f"\n--- Chart {i+1}: {chart.get('chart_type').upper()} CHART ---")
        print(f"Reasoning: {chart.get('reasoning')}")
        print(f"Priority: {chart.get('priority')}")
        
        # Print chart configuration
        config = chart.get('chart_config', {})
        print("\nConfiguration:")
        print(f"  Title: {config.get('title')}")
        print(f"  X-Axis: {config.get('x_axis')}")
        print(f"  Y-Axis: {config.get('y_axis')}")
        print(f"  Color By: {config.get('color_by')}")
        print(f"  Aggregate Function: {config.get('aggregate_function')}")
        print(f"  Chart Library: {config.get('chart_library')}")
        
        # Print additional config if available
        if config.get('additional_config'):
            print("\n  Additional Config:")
            for key, value in config.get('additional_config', {}).items():
                print(f"    {key}: {value}")
    
    # Print data insights
    print("\n===== DATA INSIGHTS =====")
    for insight in result.get("data_insights", []):
        print(f"- {insight}")


def test_openai_chart_recommendations():
    """Test OpenAI-generated chart recommendations"""
    print("\n============================================")
    print("TESTING OPENAI-GENERATED CHART RECOMMENDATIONS")
    print("============================================")
    
    # Example query that would benefit from visualization
    query = "Show me the total sales by store for the top 10 stores in 2023"
    
    # Run the query with chart recommendations
    result = nlq_to_snowflake(
        prompt=query,
        include_charts=True,
        execute_query=True
    )
    
    # Print the SQL query
    print("\nSQL Query:")
    print(result.get("sql", ""))
    
    # Print chart recommendations
    print_chart_recommendations(result)


def test_claude_chart_recommendations():
    """Test Claude-generated chart recommendations"""
    print("\n============================================")
    print("TESTING CLAUDE-GENERATED CHART RECOMMENDATIONS")
    print("============================================")
    
    # Example query that would benefit from visualization
    query = "What are the monthly revenue trends for the past year?"
    
    # Run the query with chart recommendations
    result = nlq_to_snowflake_claude(
        prompt=query,
        include_charts=True,
        execute_query=True
    )
    
    # Print the SQL query
    print("\nSQL Query:")
    print(result.get("sql", ""))
    
    # Print chart recommendations
    print_chart_recommendations(result)


def test_gemini_chart_recommendations():
    """Test Gemini-generated chart recommendations"""
    print("\n============================================")
    print("TESTING GEMINI-GENERATED CHART RECOMMENDATIONS")
    print("============================================")
    
    # Example query that would benefit from visualization
    query = "What's the distribution of sales by product category?"
    
    # Run the query with chart recommendations
    result = nlq_to_snowflake_gemini(
        prompt=query,
        include_charts=True,
        execute_query=True
    )
    
    # Print the SQL query
    print("\nSQL Query:")
    print(result.get("sql", ""))
    
    # Print chart recommendations
    print_chart_recommendations(result)


if __name__ == "__main__":
    # Run all tests
    test_openai_chart_recommendations()
    test_claude_chart_recommendations()
    test_gemini_chart_recommendations()
