#!/usr/bin/env python
"""
Test script to verify all parameter combinations for the nlq_to_snowflake function
"""
import sys
import json
from nlq_to_snowflake import nlq_to_snowflake

def main():
    """
    Test all combinations of execute_query and include_charts parameters
    """
    test_query = "Show total sales for top 3 products"
    
    # Define all combinations to test
    combinations = [
        {"execute_query": True, "include_charts": True},
        {"execute_query": True, "include_charts": False},
        {"execute_query": False, "include_charts": True},
        {"execute_query": False, "include_charts": False}
    ]
    
    # Run each test combination
    for i, params in enumerate(combinations):
        print(f"\n\n{'='*60}")
        print(f"TEST {i+1}: execute_query={params['execute_query']}, include_charts={params['include_charts']}")
        print(f"{'='*60}")
        
        # Call the function with the current parameter combination
        try:
            result = nlq_to_snowflake(
                prompt=test_query,
                execute_query=params['execute_query'],
                include_charts=params['include_charts'],
                model="gpt-4o"  # Explicitly use the same model for consistency
            )
            
            # Check if chart_recommendations and chart_error fields are present
            chart_rec_type = type(result.get('chart_recommendations', None)).__name__
            chart_error = result.get('chart_error', None)
            
            # Print key response info
            print("\nRESULT STATUS:")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Has SQL: {'sql' in result}")
            print(f"  Has Results: {'results' in result}")
            
            print("\nCHART FIELDS:")
            print(f"  chart_recommendations: {chart_rec_type}")
            print(f"  chart_error: {chart_error}")
            
            print("\nRESPONSE KEYS:")
            print(f"  {list(result.keys())}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            print(f"TRACEBACK: {sys.exc_info()}")

if __name__ == "__main__":
    main()
