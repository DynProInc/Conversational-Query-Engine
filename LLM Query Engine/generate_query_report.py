"""
Generate a query report CSV file that focuses on prompts and generated SQL queries.
This script reads from the token_usage.csv file and creates a new CSV with a focus on:
- Date and time
- Model used
- Prompt (natural language query)
- Generated SQL query
"""

import os
import csv
import pandas as pd
import datetime
import json

# Path to the input token usage CSV file
TOKEN_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_usage.csv')
# Path to store the output report
QUERY_REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_report.csv')

def get_sql_query_from_log(prompt, model):
    """
    Try to get the SQL query that was generated for a specific prompt and model.
    This is just a placeholder - in a real implementation, you would retrieve this from logs.
    """
    # In a real implementation, this would look up the query from logs or cache
    # For now, let's just return a placeholder
    return "Not available in logs"

def generate_query_report():
    """
    Generate a CSV report focusing on prompts and generated SQL queries
    """
    print("Generating query report...")
    
    if not os.path.exists(TOKEN_USAGE_FILE):
        print(f"Error: Token usage file not found at {TOKEN_USAGE_FILE}")
        return
    
    # Read the token usage data
    try:
        df = pd.read_csv(TOKEN_USAGE_FILE)
        print(f"Read {len(df)} records from {TOKEN_USAGE_FILE}")
    except Exception as e:
        print(f"Error reading token usage file: {str(e)}")
        return
    
    # Create a new dataframe for the report
    report_data = []
    
    # Process each row in the token usage file
    for _, row in df.iterrows():
        # Extract the fields we're interested in
        timestamp = row.get('timestamp', '')
        model = row.get('model', '')
        prompt = row.get('prompt', '')
        query = get_sql_query_from_log(prompt, model)
        
        # Add to our report data
        report_data.append({
            'date': timestamp.split()[0] if ' ' in timestamp else '',
            'time': timestamp.split()[1] if ' ' in timestamp else '',
            'model': model,
            'prompt': prompt,
            'sql_query': query,
            'prompt_tokens': row.get('prompt_tokens', 0),
            'completion_tokens': row.get('completion_tokens', 0),
            'total_tokens': row.get('total_tokens', 0)
        })
    
    # Write the report to CSV
    try:
        with open(QUERY_REPORT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'time', 'model', 'prompt', 'sql_query', 
                                                  'prompt_tokens', 'completion_tokens', 'total_tokens'])
            writer.writeheader()
            writer.writerows(report_data)
        
        print(f"Query report generated successfully: {QUERY_REPORT_FILE}")
    except Exception as e:
        print(f"Error writing report file: {str(e)}")

if __name__ == '__main__':
    generate_query_report()
    print("Done.")
