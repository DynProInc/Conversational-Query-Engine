"""
Script to remove data_insights from all relevant Python files
"""
import os
import re

def process_file(filename):
    print(f"Processing {filename}...")
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove data_insights from QueryResponse and ComparisonResponse models
    content = re.sub(r'data_insights: Optional\[List\[str\]\] = None(\s+)', r'\1', content)
    
    # Remove data_insights parameter from all QueryResponse initializations
    content = re.sub(r'chart_recommendations=([^,\n]+),\s+data_insights=([^,\n]+)', r'chart_recommendations=\1', content)
    content = re.sub(r'chart_recommendations=None,\s+data_insights=None', r'chart_recommendations=None', content)
    
    # Remove data_insights from responses in chart_recommendations.py
    content = re.sub(r'"data_insights": \[[^\]]*\],?\s+', r'', content)
    content = re.sub(r'response\["data_insights"\] = [^\n]+\n', r'', content)
    
    # Remove data_insights from NLQ-to-Snowflake files
    content = re.sub(r'response\["data_insights"\] = [^\n]+\n', r'', content)
    content = re.sub(r'sql_result\.get\("data_insights", \[\]\)', r'[]', content)
    
    # Remove lines that set data_insights in results
    content = re.sub(r'result\["data_insights"\] = [^\n]+\n', r'', content)
    
    # Write the updated content back to the file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Processed {filename}")

# Process all Python files in the current directory
base_path = os.path.dirname(os.path.abspath(__file__))
files_to_process = [
    os.path.join(base_path, 'api_server.py'),
    os.path.join(base_path, 'chart_recommendations.py'),
    os.path.join(base_path, 'nlq_to_snowflake.py'),
    os.path.join(base_path, 'nlq_to_snowflake_claude.py'),
    os.path.join(base_path, 'nlq_to_snowflake_gemini.py'),
    os.path.join(base_path, 'llm_query_generator.py'),
    os.path.join(base_path, 'claude_query_generator.py'),
    os.path.join(base_path, 'gemini_query_generator.py')
]

for file_path in files_to_process:
    if os.path.exists(file_path):
        process_file(file_path)
    else:
        print(f"File not found: {file_path}")

print("Done removing data_insights from all files.")
