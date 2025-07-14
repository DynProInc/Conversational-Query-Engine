#!/usr/bin/env python
"""
Simple test script to diagnose output issues
"""
import json
from llm_query_generator import generate_sql_query

def main():
    """Run a simple test and write results to a file"""
    # Test with include_charts=False
    print("Testing with include_charts=False")
    
    try:
        # Get API key from environment variable
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        
        # Generate a simple prompt
        prompt = "Please convert the following question to SQL: Show top 3 products by sales."
        
        # Call function with include_charts=False
        result1 = generate_sql_query(
            api_key=api_key,
            prompt=prompt,
            model="gpt-4o",
            include_charts=False
        )
        
        # Write results to a file
        with open("test_results_false.json", "w") as f:
            f.write(json.dumps(result1, indent=2, default=str))
            
        # Check if chart fields exist
        chart_recs = result1.get("chart_recommendations")
        chart_error = result1.get("chart_error")
        
        with open("test_output.txt", "w") as f:
            f.write(f"RESULT KEYS: {sorted(list(result1.keys()))}\n")
            f.write(f"chart_recommendations present: {chart_recs is not None}\n")
            f.write(f"chart_recommendations type: {type(chart_recs).__name__}\n")
            f.write(f"chart_error present: {chart_error is not None}\n")
            f.write(f"chart_error value: {chart_error}\n")
            
        print("Test completed. Check test_output.txt for results.")
        
    except Exception as e:
        import traceback
        with open("test_error.txt", "w") as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write(traceback.format_exc())
        print("Error occurred. Check test_error.txt for details.")

if __name__ == "__main__":
    main()
