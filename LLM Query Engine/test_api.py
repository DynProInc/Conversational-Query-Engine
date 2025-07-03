#!/usr/bin/env python3
"""
Test client for the API Server
"""
import requests
import json
import sys
import os
import argparse
from token_logger import TokenLogger

def make_api_request(prompt, model="openai", limit_rows=100, data_dictionary_path=None, specific_model=None):
    """Make a request to the API Server
    
    Args:
        prompt: Natural language query
        model: Which backend to use - "openai", "claude", or "compare" 
        limit_rows: Maximum rows to return
        data_dictionary_path: Optional path to data dictionary
        specific_model: Optional specific model name (e.g., "gpt-4o" or "claude-3-opus")
    """
    # Constants
    API_BASE_URL = "http://localhost:8000"
    API_OPENAI_ENDPOINT = f"{API_BASE_URL}/query"
    API_CLAUDE_ENDPOINT = f"{API_BASE_URL}/query/claude"
    API_COMPARE_ENDPOINT = f"{API_BASE_URL}/query/compare"
    API_MODELS_ENDPOINT = f"{API_BASE_URL}/models"

    print(f"\nDEBUG: Making API request to {model.lower()} model")
    
    # Select the appropriate endpoint based on model choice
    if model.lower() == "claude":
        endpoint = API_CLAUDE_ENDPOINT
        print(f"DEBUG: Using Claude endpoint: {endpoint}")
    elif model.lower() == "compare":
        endpoint = API_COMPARE_ENDPOINT
        print(f"DEBUG: Using Compare endpoint: {endpoint}")
    else:  # Default to OpenAI
        endpoint = API_OPENAI_ENDPOINT
        print(f"DEBUG: Using OpenAI endpoint: {endpoint}")
    
    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "limit_rows": limit_rows
    }
    
    # Add optional parameters if provided
    if data_dictionary_path:
        payload["data_dictionary_path"] = data_dictionary_path
        
    if specific_model:
        payload["model"] = specific_model

    print(f"DEBUG: Request payload: {payload}")
    
    try:
        # Make the API request
        response = requests.post(endpoint, json=payload)
        print(f"DEBUG: Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"DEBUG: Error response content: {response.text}")
            
        return response
    except Exception as e:
        print(f"DEBUG: Exception making API request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise   

def get_available_models():
    """Get the list of available models from the API"""
    API_BASE_URL = "http://localhost:8000"
    API_MODELS_ENDPOINT = f"{API_BASE_URL}/models"
    try:
        response = requests.get(API_MODELS_ENDPOINT)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve models: {response.status_code}")
            return {"openai": ["gpt-4o"], "claude": ["claude-3-5-sonnet"]}
    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        return {"openai": ["gpt-4o"], "claude": ["claude-3-5-sonnet"]}

def display_results(response, model_type):
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()
        
        print("\n=== API Response ===")
        print(f"Prompt: {result['prompt']}")
        
        # Handle comparison mode differently
        if model_type == "compare":
            # OpenAI results
            if result.get('openai'):
                print("\n=== OpenAI Results ===")
                openai_result = result['openai']
                print(f"Model: {openai_result.get('model', 'Unknown')}")
                print(f"Generated SQL Query: \n{openai_result.get('query', 'No query generated')}")
                
                print("\nQuery Output:")
                if openai_result.get('query_output'):
                    # Print the results in a formatted way
                    for row in openai_result['query_output'][:5]:  # Limit to 5 rows for readability
                        print(json.dumps(row, indent=2))
                    if len(openai_result['query_output']) > 5:
                        print(f"...and {len(openai_result['query_output']) - 5} more rows")
                else:
                    print("No results returned or query was not executed.")
                
                # Print token usage if available
                if openai_result.get('token_usage'):
                    print("\nToken Usage:")
                    print(f"  Prompt tokens: {openai_result['token_usage'].get('prompt_tokens', 0)}")
                    print(f"  Completion tokens: {openai_result['token_usage'].get('completion_tokens', 0)}")
                    print(f"  Total tokens: {openai_result['token_usage'].get('total_tokens', 0)}")
                
                # Handle the case where execution_time_ms might be None
                exec_time = openai_result.get('execution_time_ms')
                if exec_time is not None:
                    print(f"Execution time: {exec_time:.2f} ms")
                else:
                    print("Execution time: Not available")
            
            # Claude results
            if result.get('claude'):
                print("\n=== Claude Results ===")
                claude_result = result['claude']
                print(f"Model: {claude_result.get('model', 'Unknown')}")
                print(f"Generated SQL Query: \n{claude_result.get('query', 'No query generated')}")
                
                print("\nQuery Output:")
                if claude_result.get('query_output'):
                    # Print the results in a formatted way
                    for row in claude_result['query_output'][:5]:  # Limit to 5 rows for readability
                        print(json.dumps(row, indent=2))
                    if len(claude_result['query_output']) > 5:
                        print(f"...and {len(claude_result['query_output']) - 5} more rows")
                else:
                    print("No results returned or query was not executed.")
                
                # Print token usage if available
                if claude_result.get('token_usage'):
                    print("\nToken Usage:")
                    print(f"  Prompt tokens: {claude_result['token_usage'].get('prompt_tokens', 0)}")
                    print(f"  Completion tokens: {claude_result['token_usage'].get('completion_tokens', 0)}")
                    print(f"  Total tokens: {claude_result['token_usage'].get('total_tokens', 0)}")
                
                # Handle the case where execution_time_ms might be None
                exec_time = claude_result.get('execution_time_ms')
                if exec_time is not None:
                    print(f"Execution time: {exec_time:.2f} ms")
                else:
                    print("Execution time: Not available")
        else:
            # Handle regular (single model) response
            print(f"Model: {result.get('model', 'Unknown')}")
            print(f"Generated SQL Query: \n{result.get('query', 'No query generated')}")
            
            print("\nQuery Output:")
            if result.get('query_output'):
                # Print the results in a formatted way
                for row in result['query_output'][:5]:  # Limit to 5 rows for readability
                    print(json.dumps(row, indent=2))
                if len(result['query_output']) > 5:
                    print(f"...and {len(result['query_output']) - 5} more rows")
            else:
                print("No results returned or query was not executed.")
            
            # Print token usage if available
            if result.get('token_usage'):
                print("\nToken Usage:")
                print(f"  Prompt tokens: {result['token_usage'].get('prompt_tokens', 0)}")
                print(f"  Completion tokens: {result['token_usage'].get('completion_tokens', 0)}")
                print(f"  Total tokens: {result['token_usage'].get('total_tokens', 0)}")
            
            # Handle the case where execution_time_ms might be None
            exec_time = result.get('execution_time_ms')
            if exec_time is not None:
                print(f"Execution time: {exec_time:.2f} ms")
            else:
                print("Execution time: Not available")
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Setup parameters
    prompt = args.prompt if args.prompt else input("Enter your natural language query: ")
    model = args.model
    specific_model = args.specific_model
    limit_rows = args.limit
    data_dict = args.data_dictionary
    
    # Initialize token logger
    logger = TokenLogger()
    
    # Display what's being sent
    print(f"\nSending query: '{prompt}'")
    print(f"Using model: {model}{' ('+specific_model+')' if specific_model else ''}")
    print(f"Row limit: {limit_rows}")
    
    try:
        # Make the API request
        response = make_api_request(
            prompt=prompt,
            model=model,
            limit_rows=limit_rows,
            specific_model=specific_model,
            data_dictionary_path=data_dict
        )
        
        # Display results
        display_results(response, model)
        
        # Log the token usage with SQL query
        if response.status_code == 200:
            result = response.json()
            actual_model = result.get('model', model)
            sql_query = result.get('query', '')
            token_usage = result.get('token_usage', {})
            
            if token_usage:
                # Log the token usage with the SQL query
                logger.log_usage(
                    model=actual_model,
                    query=prompt,
                    usage=token_usage,
                    prompt=prompt,
                    sql_query=sql_query
                )
        
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()

def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test the NL-to-SQL API with different models")
    parser.add_argument("--prompt", "-p", help="Natural language query")
    parser.add_argument("--limit", "-l", type=int, default=100, help="Maximum number of rows to return")
    parser.add_argument("--model", "-m", default="openai", choices=["openai", "claude", "compare"],
                        help="Model to use for generation (openai, claude, or compare)")
    parser.add_argument("--specific-model", "-s", help="Specific model version to use (e.g., gpt-4o, claude-3-opus)")
    parser.add_argument("--data-dictionary", "-d", help="Path to data dictionary")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
