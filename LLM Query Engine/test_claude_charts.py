#!/usr/bin/env python
"""
Test script to verify Claude chart recommendations functionality
"""
import os
import json
from dotenv import load_dotenv
from claude_query_generator import natural_language_to_sql_claude

# Load environment variables
load_dotenv()

def test_claude_with_charts():
    """Test Claude's ability to generate chart recommendations"""
    
    print("\n" + "="*80)
    print("TESTING CLAUDE WITH CHART RECOMMENDATIONS")
    print("="*80)
    
    # Simple analytical query that should generate good chart recommendations
    query = "Show total sales by product category over the last 12 months"
    
    try:
        # Call Claude with include_charts=True
        print(f"\nGenerating SQL with charts for query: '{query}'")
        result = natural_language_to_sql_claude(
            query=query,
            include_charts=True  # This is the key parameter we're testing
        )
        
        # Check if we got chart recommendations
        chart_recommendations = result.get('chart_recommendations')
        chart_error = result.get('chart_error')
        
        print("\nRESULT:")
        print(f"SQL Query: {result.get('sql')[:100]}...")  # Show first 100 chars
        
        print("\nCHART DATA:")
        if chart_recommendations:
            print(f"✓ Chart recommendations found: {len(chart_recommendations)} recommendations")
            # Save to file for inspection
            with open('claude_chart_test_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("✓ Results saved to claude_chart_test_results.json")
            
            # Show first chart recommendation
            if chart_recommendations:
                print("\nSample chart recommendation:")
                first_chart = chart_recommendations[0]
                print(f"- Type: {first_chart.get('chart_type')}")
                print(f"- Reasoning: {first_chart.get('reasoning')[:100]}..." if first_chart.get('reasoning') else "- No reasoning provided")
        else:
            print(f"✗ No chart recommendations found")
            print(f"Chart error: {chart_error}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_claude_with_charts()
