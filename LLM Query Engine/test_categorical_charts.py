"""
Test script specifically for categorical data chart recommendations
"""
import pandas as pd
import json
from chart_recommendations import analyze_query_results_for_charts

def print_separator():
    print("\n" + "="*80 + "\n")

def test_purely_categorical_data():
    """Test chart recommendations for purely categorical data"""
    print_separator()
    print("TEST: PURELY CATEGORICAL DATA")
    print_separator()
    
    # Sample data with only categorical columns (no numeric data)
    categorical_data = [
        {
            "CATEGORY_NAME": "OTR TIRE SERVICE LABOR",
            "STORE_NAME": "MCCARTHY TIRE - LANCASTER, PA"
        },
        {
            "CATEGORY_NAME": "FUEL SURCHARGE",
            "STORE_NAME": "MCCARTHY TIRE - BUFFALO, NY"
        },
        {
            "CATEGORY_NAME": "TRAILER & TURF TIRES",
            "STORE_NAME": "MCCARTHY TIRE - RICHMOND, VA"
        }
    ]
    
    # Test with categorical data
    result = analyze_query_results_for_charts(
        "SELECT DISTINCT CATEGORY_NAME, STORE_NAME FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES LIMIT 3;",
        categorical_data
    )
    
    print("QUERY:")
    print("SELECT DISTINCT CATEGORY_NAME, STORE_NAME FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES LIMIT 3;")
    print("\nDATA:")
    for row in categorical_data:
        print(row)
    
    print("\nRESULTS:")
    print(f"Chart Error: {result.get('chart_error')}")
    print(f"Number of recommendations: {len(result.get('chart_recommendations', []))}")
    
    # Print recommendations
    recommendations = result.get('chart_recommendations', [])
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. Chart Type: {rec.get('chart_type')}")
            print(f"   Reasoning: {rec.get('reasoning')}")
            print(f"   Priority: {rec.get('priority')}")
            
            # Print chart config details
            config = rec.get('chart_config', {})
            print(f"   Title: {config.get('title')}")
            if config.get('chart_library'):
                print(f"   Chart Library: {config.get('chart_library')}")
    else:
        print("\nNo recommendations available")
    
    # Verify we get a table recommendation and not bar/pie charts
    chart_types = [rec.get('chart_type') for rec in result.get('chart_recommendations', [])]
    print("\nVERIFICATION:")
    print(f"Contains 'table' recommendation: {'table' in chart_types}")
    print(f"Contains 'bar' recommendation: {'bar' in chart_types}")
    print(f"Contains 'pie' recommendation: {'pie' in chart_types}")
    
    # Assertions
    assert 'table' in chart_types, "Should recommend a table for categorical data"
    assert 'bar' not in chart_types, "Should not recommend a bar chart for purely categorical data"
    assert 'pie' not in chart_types, "Should not recommend a pie chart for purely categorical data"
    
    return result

if __name__ == "__main__":
    print("Testing chart recommendation system for categorical data...")
    test_purely_categorical_data()
    print("\nTest completed successfully!")
