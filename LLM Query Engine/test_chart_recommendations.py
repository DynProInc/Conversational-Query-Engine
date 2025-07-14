"""
Test script for chart recommendations with various data types
"""
import pandas as pd
import json
from chart_recommendations import analyze_query_results_for_charts

def test_categorical_data():
    """Test chart recommendations for purely categorical data"""
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
    
    print("\n=== TEST: Purely Categorical Data ===")
    print(f"Chart Error: {result.get('chart_error')}")
    print(f"Number of recommendations: {len(result.get('chart_recommendations', []))}")
    print("Recommendation types:", [rec.get('chart_type') for rec in result.get('chart_recommendations', [])])
    
    # Print first recommendation in a more readable format
    if result.get('chart_recommendations'):
        first_rec = result.get('chart_recommendations')[0]
        print(f"\nFirst recommendation: {first_rec.get('chart_type')}")
        print(f"Reasoning: {first_rec.get('reasoning')}")
    else:
        print("No recommendations available")
    
    # Verify we get a table recommendation and not bar/pie charts
    chart_types = [rec.get('chart_type') for rec in result.get('chart_recommendations', [])]
    assert 'table' in chart_types, "Should recommend a table for categorical data"
    assert 'bar' not in chart_types, "Should not recommend a bar chart for purely categorical data"
    assert 'pie' not in chart_types, "Should not recommend a pie chart for purely categorical data"
    
def test_categorical_with_numeric():
    """Test chart recommendations for mixed categorical and numeric data"""
    # Sample data with both categorical and numeric columns
    mixed_data = [
        {
            "CATEGORY_NAME": "OTR TIRE SERVICE LABOR",
            "STORE_NAME": "MCCARTHY TIRE - LANCASTER, PA",
            "SALES_AMOUNT": 1500.50
        },
        {
            "CATEGORY_NAME": "FUEL SURCHARGE",
            "STORE_NAME": "MCCARTHY TIRE - BUFFALO, NY",
            "SALES_AMOUNT": 250.75
        },
        {
            "CATEGORY_NAME": "TRAILER & TURF TIRES",
            "STORE_NAME": "MCCARTHY TIRE - RICHMOND, VA",
            "SALES_AMOUNT": 3200.25
        }
    ]
    
    # Test with mixed data
    result = analyze_query_results_for_charts(
        "SELECT CATEGORY_NAME, STORE_NAME, SALES_AMOUNT FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES LIMIT 3;",
        mixed_data
    )
    
    print("\n=== TEST: Mixed Categorical and Numeric Data ===")
    print(f"Chart Error: {result.get('chart_error')}")
    print(f"Number of recommendations: {len(result.get('chart_recommendations', []))}")
    print("Recommendation types:", [rec.get('chart_type') for rec in result.get('chart_recommendations', [])])
    
    # Print first recommendation in a more readable format
    if result.get('chart_recommendations'):
        first_rec = result.get('chart_recommendations')[0]
        print(f"\nFirst recommendation: {first_rec.get('chart_type')}")
        print(f"Reasoning: {first_rec.get('reasoning')}")
    else:
        print("No recommendations available")
    
    # Verify we get appropriate chart recommendations for mixed data
    chart_types = [rec.get('chart_type') for rec in result.get('chart_recommendations', [])]
    assert 'bar' in chart_types, "Should recommend a bar chart for categorical + numeric data"
    
def test_time_series_data():
    """Test chart recommendations for time series data"""
    # Sample data with date and numeric columns
    time_series_data = [
        {
            "DATE": "2023-01-01",
            "SALES_AMOUNT": 1500.50
        },
        {
            "DATE": "2023-01-02",
            "SALES_AMOUNT": 1750.25
        },
        {
            "DATE": "2023-01-03",
            "SALES_AMOUNT": 2100.75
        }
    ]
    
    # Convert string dates to datetime
    df = pd.DataFrame(time_series_data)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Test with time series data
    result = analyze_query_results_for_charts(
        "SELECT DATE, SALES_AMOUNT FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES LIMIT 3;",
        df
    )
    
    print("\n=== TEST: Time Series Data ===")
    print(f"Chart Error: {result.get('chart_error')}")
    print(f"Number of recommendations: {len(result.get('chart_recommendations', []))}")
    print("Recommendation types:", [rec.get('chart_type') for rec in result.get('chart_recommendations', [])])
    
    # Print first recommendation in a more readable format
    if result.get('chart_recommendations'):
        first_rec = result.get('chart_recommendations')[0]
        print(f"\nFirst recommendation: {first_rec.get('chart_type')}")
        print(f"Reasoning: {first_rec.get('reasoning')}")
    else:
        print("No recommendations available")
    
    # Verify we get line chart recommendation for time series data
    chart_types = [rec.get('chart_type') for rec in result.get('chart_recommendations', [])]
    assert 'line' in chart_types, "Should recommend a line chart for time series data"

if __name__ == "__main__":
    print("Testing chart recommendation system...")
    test_categorical_data()
    test_categorical_with_numeric()
    test_time_series_data()
    print("\nAll tests completed!")
