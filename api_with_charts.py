"""
API example showing how to expose chart recommendations through a REST API
"""
from flask import Flask, request, jsonify
import pandas as pd
from nlq_to_snowflake_claude import nlq_to_snowflake_claude

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
    model_type = data.get('model_type', 'claude').lower()  # 'claude' only
    execute = data.get('execute', True)
    limit_rows = data.get('limit_rows', 100)
    include_charts = data.get('include_charts', True)
    
    # Only support Claude
    model = data.get('model', 'claude-3-5-sonnet-20241022')
    
    try:
        # Use Claude for all requests
        result = nlq_to_snowflake_claude(
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
