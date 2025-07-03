"""
Create a comprehensive query report CSV that focuses on:
- Date and time
- Model used
- Prompt (natural language query)
- Generated SQL query
- Token usage metrics

This script will generate sample data and create a new CSV file.
"""

import os
import csv
import datetime
import pandas as pd

# Output file path
QUERY_REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_report.csv')

# Sample data - this would normally come from your actual query logs
sample_data = [
    {
        'date': '2025-07-01',
        'time': '10:15:30',
        'model': 'gpt-4o',
        'prompt': 'Show me the top 5 stores with highest sales',
        'sql_query': 'SELECT STORE_NUMBER, STORE_NAME, SUM(DAILY_TOTAL_SALE) AS TOTAL_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES GROUP BY STORE_NUMBER, STORE_NAME ORDER BY TOTAL_SALES DESC LIMIT 5;',
        'prompt_tokens': 1566,
        'completion_tokens': 72,
        'total_tokens': 1638,
        'input_cost': 0.01566,
        'output_cost': 0.00216,
        'total_cost': 0.01782
    },
    {
        'date': '2025-07-01',
        'time': '11:20:45',
        'model': 'claude-3-5-sonnet-20241022',
        'prompt': 'What are the total sales by product category?',
        'sql_query': 'SELECT PRODUCT_CATEGORY, SUM(SALES_AMOUNT) AS TOTAL_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES GROUP BY PRODUCT_CATEGORY ORDER BY TOTAL_SALES DESC;',
        'prompt_tokens': 1918,
        'completion_tokens': 147,
        'total_tokens': 2065,
        'input_cost': 0.05754,
        'output_cost': 0.00882,
        'total_cost': 0.06636
    },
    {
        'date': '2025-07-02',
        'time': '09:05:15',
        'model': 'gpt-4o',
        'prompt': 'Show me sales trends for the past 3 months',
        'sql_query': 'SELECT MONTH(SALE_DATE) AS MONTH, YEAR(SALE_DATE) AS YEAR, SUM(DAILY_TOTAL_SALE) AS MONTHLY_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES WHERE SALE_DATE >= DATEADD(MONTH, -3, CURRENT_DATE()) GROUP BY MONTH(SALE_DATE), YEAR(SALE_DATE) ORDER BY YEAR, MONTH;',
        'prompt_tokens': 1580,
        'completion_tokens': 95,
        'total_tokens': 1675,
        'input_cost': 0.01580,
        'output_cost': 0.00285,
        'total_cost': 0.01865
    },
    {
        'date': '2025-07-02',
        'time': '14:30:22',
        'model': 'claude-3-5-sonnet-20241022',
        'prompt': 'Which customers have made the most purchases?',
        'sql_query': 'SELECT CUSTOMER_ID, CUSTOMER_NAME, COUNT(DISTINCT ORDER_ID) AS NUMBER_OF_ORDERS, SUM(DAILY_TOTAL_SALE) AS TOTAL_SPENT FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES GROUP BY CUSTOMER_ID, CUSTOMER_NAME ORDER BY NUMBER_OF_ORDERS DESC LIMIT 10;',
        'prompt_tokens': 1930,
        'completion_tokens': 160,
        'total_tokens': 2090,
        'input_cost': 0.05790,
        'output_cost': 0.00960,
        'total_cost': 0.06750
    },
    {
        'date': '2025-07-03',
        'time': '08:45:10',
        'model': 'gpt-4o',
        'prompt': 'Show me average sales by day of week',
        'sql_query': 'SELECT DAYNAME(SALE_DATE) AS DAY_OF_WEEK, AVG(DAILY_TOTAL_SALE) AS AVERAGE_SALES FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES GROUP BY DAYNAME(SALE_DATE), DAYOFWEEK(SALE_DATE) ORDER BY DAYOFWEEK(SALE_DATE);',
        'prompt_tokens': 1570,
        'completion_tokens': 88,
        'total_tokens': 1658,
        'input_cost': 0.01570,
        'output_cost': 0.00264,
        'total_cost': 0.01834
    },
    {
        'date': '2025-07-03',
        'time': '16:10:35',
        'model': 'claude-3-5-sonnet-20241022',
        'prompt': 'What products have the highest profit margin?',
        'sql_query': 'SELECT PRODUCT_ID, PRODUCT_NAME, (SUM(SALES_AMOUNT) - SUM(COST_AMOUNT))/SUM(SALES_AMOUNT) * 100 AS PROFIT_MARGIN_PERCENTAGE FROM REPORTING_UAT.GOLD_SALES.V_SMM_DAILY_SALES GROUP BY PRODUCT_ID, PRODUCT_NAME ORDER BY PROFIT_MARGIN_PERCENTAGE DESC LIMIT 20;',
        'prompt_tokens': 1940,
        'completion_tokens': 175,
        'total_tokens': 2115,
        'input_cost': 0.05820,
        'output_cost': 0.01050,
        'total_cost': 0.06870
    }
]

def create_query_report():
    """
    Create a CSV report focusing on prompts and generated SQL queries
    """
    print("Creating query report CSV...")
    
    try:
        # Add current timestamp to the last sample
        current_time = datetime.datetime.now()
        sample_data[-1]['date'] = current_time.strftime('%Y-%m-%d')
        sample_data[-1]['time'] = current_time.strftime('%H:%M:%S')
        
        # Write the report to CSV
        with open(QUERY_REPORT_FILE, 'w', newline='') as f:
            fieldnames = ['date', 'time', 'model', 'prompt', 'sql_query', 
                         'prompt_tokens', 'completion_tokens', 'total_tokens',
                         'input_cost', 'output_cost', 'total_cost']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_data)
        
        print(f"Query report created successfully at: {QUERY_REPORT_FILE}")
        
        # Also display the DataFrame for verification
        df = pd.DataFrame(sample_data)
        print("\nReport preview:")
        pd.set_option('display.max_colwidth', 30)  # Limit column width for display
        print(df[['date', 'time', 'model', 'prompt', 'sql_query']].head())
        
    except Exception as e:
        print(f"Error creating report file: {str(e)}")

if __name__ == '__main__':
    create_query_report()
    print("Done.")
