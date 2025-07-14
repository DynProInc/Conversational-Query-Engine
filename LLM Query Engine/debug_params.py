#!/usr/bin/env python
"""
Debug script to directly test parameter combinations for SQL generation
"""
import sys
import traceback
from typing import Dict, Any
from llm_query_generator import natural_language_to_sql

# Fixed query to test with
TEST_QUERY = "Show total sales for the top 3 products"

def run_test(execute_query: bool, include_charts: bool) -> Dict[str, Any]:
    """Run a test with specific parameter combinations"""
    print(f"\n\n{'='*50}")
    print(f"TEST: execute_query={execute_query}, include_charts={include_charts}")
    print(f"{'='*50}")
    
    try:
        # Directly call SQL generation without snowflake execution
        result = natural_language_to_sql(
            query=TEST_QUERY,
            model="gpt-4o",  # Use consistent model
            model_provider="openai",
            limit_rows=10,
            include_charts=include_charts
        )
        
        print("\nSQL GENERATION SUCCESSFUL")
        
        # Check key outputs
        print("\nRESULT KEYS:")
        print(f"  {sorted(list(result.keys()))}")
        
        # Check chart fields
        chart_recs = result.get("chart_recommendations")
        chart_error = result.get("chart_error")
        
        print("\nCHART FIELDS:")
        print(f"  chart_recommendations: {type(chart_recs).__name__}, Value: {chart_recs}")
        print(f"  chart_error: {chart_error}")
        
        return result
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("TRACEBACK:")
        traceback.print_exc()
        return {"error": str(e), "success": False}

def main():
    """Run tests for all parameter combinations"""
    # Test all combinations
    print("STARTING PARAMETER COMBINATION TESTS")
    
    # Test 1: execute_query=False, include_charts=False
    result1 = run_test(execute_query=False, include_charts=False)
    
    # Test 2: execute_query=False, include_charts=True
    result2 = run_test(execute_query=False, include_charts=True)
    
    # Test 3: execute_query=True, include_charts=False
    # We don't actually execute in this test script
    result3 = run_test(execute_query=True, include_charts=False)
    
    # Test 4: execute_query=True, include_charts=True
    # We don't actually execute in this test script
    result4 = run_test(execute_query=True, include_charts=True)
    
    # Summary
    print("\n\n" + "="*30 + " TEST SUMMARY " + "="*30)
    tests = [
        ("Test 1: execute_query=False, include_charts=False", result1),
        ("Test 2: execute_query=False, include_charts=True", result2),
        ("Test 3: execute_query=True, include_charts=False", result3),
        ("Test 4: execute_query=True, include_charts=True", result4)
    ]
    
    for test_name, result in tests:
        chart_fields_present = ("chart_recommendations" in result and "chart_error" in result)
        print(f"{test_name}: {'SUCCESS' if chart_fields_present else 'FAILURE'}")
        if not chart_fields_present:
            print(f"  Missing fields. Available keys: {sorted(list(result.keys()))}")

if __name__ == "__main__":
    main()
