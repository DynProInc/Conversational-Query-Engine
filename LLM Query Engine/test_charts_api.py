"""
Test script to demonstrate chart recommendations in API response
"""
import json
from nlq_to_snowflake import nlq_to_snowflake
from nlq_to_snowflake_claude import nlq_to_snowflake_claude
from nlq_to_snowflake_gemini import nlq_to_snowflake_gemini

def test_openai_charts():
    """Test chart recommendations with OpenAI model"""
    # Example query that would generate results good for visualization
    question = "Show me the top 5 stores by total sales in 2023"
    
    # Call the API
    result = nlq_to_snowflake(
        question=question,
        execute=True,
        limit_rows=5,
        include_charts=True  # This enables chart recommendations
    )
    
    # Pretty print just the chart recommendations part
    if 'chart_recommendations' in result:
        print("\n=== CHART RECOMMENDATIONS (OpenAI) ===")
        print(json.dumps(result['chart_recommendations'], indent=2))
        
        print("\n=== DATA INSIGHTS ===")
        print(json.dumps(result['data_insights'], indent=2))
    else:
        print("No chart recommendations available")
    
    return result

def test_claude_charts():
    """Test chart recommendations with Claude model"""
    question = "Give me the quarterly sales by product category for 2023"
    
    # Call the Claude API
    result = nlq_to_snowflake_claude(
        question=question,
        execute=True,
        limit_rows=10,
        include_charts=True
    )
    
    # Pretty print just the chart recommendations part
    if 'chart_recommendations' in result:
        print("\n=== CHART RECOMMENDATIONS (Claude) ===")
        print(json.dumps(result['chart_recommendations'], indent=2))
        
        print("\n=== DATA INSIGHTS ===")
        print(json.dumps(result['data_insights'], indent=2))
    else:
        print("No chart recommendations available")
    
    return result

def test_gemini_charts():
    """Test chart recommendations with Gemini model"""
    question = "Show me the sales trends by month for 2023 for top 3 products"
    
    # Call the Gemini API
    result = nlq_to_snowflake_gemini(
        question=question,
        execute=True,
        limit_rows=12,
        include_charts=True
    )
    
    # Pretty print just the chart recommendations part
    if 'chart_recommendations' in result:
        print("\n=== CHART RECOMMENDATIONS (Gemini) ===")
        print(json.dumps(result['chart_recommendations'], indent=2))
        
        print("\n=== DATA INSIGHTS ===")
        print(json.dumps(result['data_insights'], indent=2))
    else:
        print("No chart recommendations available")
    
    return result

if __name__ == "__main__":
    # Choose which model to test
    print("Testing chart recommendations with OpenAI...")
    openai_result = test_openai_charts()
    
    # Uncomment to test Claude
    # print("\nTesting chart recommendations with Claude...")
    # claude_result = test_claude_charts()
    
    # Uncomment to test Gemini
    # print("\nTesting chart recommendations with Gemini...")
    # gemini_result = test_gemini_charts()
