"""
Unified API endpoint for chart recommendations
Uses the exact format requested by the user
"""
from flask import Flask, request, jsonify
import os
import pandas as pd
from nlq_to_snowflake import nlq_to_snowflake
from nlq_to_snowflake_claude import nlq_to_snowflake_claude
from nlq_to_snowflake_gemini import nlq_to_snowflake_gemini

app = Flask(__name__)

@app.route('/query/unified', methods=['POST'])
def unified_query():
    """
    Unified endpoint that matches the requested format
    
    Expected request format:
    {
      "prompt": "give me results overall sales in year 2024 in store number 11 for top items and ACCOUNT_TYPE_NAME RETAIL",
      "model": "openai", # or "claude", "gemini"
      "execute_query": false,
      "include_charts": true,
      "data_dictionary_path": null
    }
    """
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request'}), 400
    
    # Extract parameters from request
    prompt = data['prompt']
    model_name = data.get('model', 'claude').lower()
    execute_query = data.get('execute_query', False)
    include_charts = data.get('include_charts', True)
    data_dictionary_path = data.get('data_dictionary_path')
    
    # Default limit for queries
    limit_rows = data.get('limit_rows', 100)
    
    try:
        # Call the appropriate function based on model
        if model_name == 'claude':
            result = nlq_to_snowflake_claude(
                question=prompt,
                execute=execute_query,
                limit_rows=limit_rows,
                data_dictionary_path=data_dictionary_path,
                include_charts=include_charts
            )
        elif model_name == 'gemini':
            result = nlq_to_snowflake_gemini(
                question=prompt,
                execute=execute_query,
                limit_rows=limit_rows,
                data_dictionary_path=data_dictionary_path,
                include_charts=include_charts
            )
        else:  # Default to OpenAI
            result = nlq_to_snowflake(
                question=prompt,
                execute=execute_query,
                limit_rows=limit_rows,
                data_dictionary_path=data_dictionary_path,
                include_charts=include_charts
            )
            
        # Convert DataFrame to list of dictionaries if present
        if 'results' in result and isinstance(result['results'], pd.DataFrame):
            result['results'] = result['results'].to_dict(orient='records')
            
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }), 500

if __name__ == '__main__':
    # Run on port 8000 as specified in the example URL
    app.run(host='0.0.0.0', port=8000, debug=True)
