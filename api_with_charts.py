"""
API example showing how to expose chart recommendations through a REST API
"""
from flask import Flask, request, jsonify
import pandas as pd
from nlq_to_snowflake import nlq_to_snowflake
from nlq_to_snowflake_claude import nlq_to_snowflake_claude
from nlq_to_snowflake_gemini import nlq_to_snowflake_gemini

app = Flask(__name__)

@app.route('/api/query', methods=['POST'])
def generate_sql_query():
    """API endpoint that returns SQL query results with chart recommendations"""
    # Get request data
    data = request.json
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question in request'}), 400
    
    # Extract parameters from request
    question = data['question']
    model_type = data.get('model_type', 'claude').lower()  # 'openai', 'claude', or 'gemini'
    execute = data.get('execute', True)
    limit_rows = data.get('limit_rows', 100)
    include_charts = data.get('include_charts', True)
    
    # Set model based on model_type
    model = None
    if model_type == 'openai':
        model = data.get('model', 'gpt-4o')
    elif model_type == 'claude':
        model = data.get('model', 'claude-3-5-sonnet-20241022')
    elif model_type == 'gemini':
        model = data.get('model', 'models/gemini-1.5-flash-latest')
    
    try:
        # Call the appropriate function based on model_type
        if model_type == 'claude':
            result = nlq_to_snowflake_claude(
                question=question,
                execute=execute,
                limit_rows=limit_rows,
                model=model,
                include_charts=include_charts
            )
        elif model_type == 'gemini':
            result = nlq_to_snowflake_gemini(
                question=question,
                execute=execute,
                limit_rows=limit_rows,
                model=model,
                include_charts=include_charts
            )
        else:  # Default to OpenAI
            result = nlq_to_snowflake(
                question=question,
                execute=execute,
                limit_rows=limit_rows,
                model=model,
                include_charts=include_charts
            )
        
        # Convert DataFrame to list of dictionaries if present
        if 'results' in result and isinstance(result['results'], pd.DataFrame):
            result['results'] = result['results'].to_dict(orient='records')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/chart-only', methods=['POST'])
def generate_chart_only():
    """API endpoint that only returns chart recommendations for an already executed query"""
    # Get request data
    data = request.json
    
    if not data or 'query' not in data or 'results' not in data:
        return jsonify({'error': 'Missing query or results in request'}), 400
    
    # Import chart recommendations module
    from chart_recommendations import analyze_query_results_for_charts
    
    # Extract parameters
    query = data['query']
    query_results = data['results']
    
    # Generate chart recommendations
    try:
        chart_data = analyze_query_results_for_charts(query, query_results)
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'chart_recommendations': [],
            'data_insights': []
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
