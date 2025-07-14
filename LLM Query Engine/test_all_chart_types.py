"""
Comprehensive test script for chart recommendations with all data types
"""
import pandas as pd
import json
from chart_recommendations import analyze_query_results_for_charts

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_result(result):
    """Print chart recommendation results in a readable format"""
    print(f"Chart Error: {result.get('chart_error', 'None')}")
    recommendations = result.get('chart_recommendations', [])
    print(f"Number of recommendations: {len(recommendations)}")
    
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. Chart Type: {rec.get('chart_type')}")
            print(f"   Reasoning: {rec.get('reasoning')}")
            print(f"   Priority: {rec.get('priority')}")
            
            # Print chart config details
            config = rec.get('chart_config', {})
            print(f"   Title: {config.get('title')}")
            print(f"   X-Axis: {config.get('x_axis')}")
            print(f"   Y-Axis: {config.get('y_axis')}")
            if config.get('chart_library'):
                print(f"   Chart Library: {config.get('chart_library')}")
    else:
        print("\nNo recommendations available")

def test_purely_categorical_data():
    """Test chart recommendations for purely categorical data"""
    print_header("TEST: PURELY CATEGORICAL DATA")
    
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
    
    print_result(result)
    
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
    assert result.get('chart_error') is not None, "Should include an error message for categorical data"
    
    return result

def test_categorical_with_numeric():
    """Test chart recommendations for mixed categorical and numeric data"""
    print_header("TEST: MIXED CATEGORICAL AND NUMERIC DATA")
    
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
    
    print_result(result)
    
    # Verify we get appropriate chart recommendations for mixed data
    chart_types = [rec.get('chart_type') for rec in result.get('chart_recommendations', [])]
    print("\nVERIFICATION:")
    print(f"Contains 'bar' recommendation: {'bar' in chart_types}")
    print(f"Contains 'pie' recommendation: {'pie' in chart_types}")
    
    # Assertions
    assert 'bar' in chart_types, "Should recommend a bar chart for categorical + numeric data"
    assert result.get('chart_error') is None, "Should not include an error message for mixed data"
    
    return result

def test_time_series_data():
    """Test chart recommendations for time series data"""
    print_header("TEST: TIME SERIES DATA")
    
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
    
    print_result(result)
    
    # Verify we get line chart recommendation for time series data
    chart_types = [rec.get('chart_type') for rec in result.get('chart_recommendations', [])]
    print("\nVERIFICATION:")
    print(f"Contains 'line' recommendation: {'line' in chart_types}")
    print(f"Contains 'area' recommendation: {'area' in chart_types}")
    
    # Assertions
    assert 'line' in chart_types, "Should recommend a line chart for time series data"
    assert result.get('chart_error') is None, "Should not include an error message for time series data"
    
    return result

if __name__ == "__main__":
    print("Testing chart recommendation system for all data types...")
    test_purely_categorical_data()
    test_categorical_with_numeric()
    test_time_series_data()
    print("\nAll tests completed successfully!")
