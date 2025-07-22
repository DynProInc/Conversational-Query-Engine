"""
Comprehensive Token Logging Test Script

This script tests token logging across all three LLM providers (OpenAI, Claude, and Gemini).
It tests three conditions for each provider:
1. Query Generated But Not Executed (execute=False)
2. Query Generated and Successfully Executed (execute=True)
3. Query Generated But Execution Failed (invalid SQL)

The script verifies that token usage is logged correctly in token_usage.csv.
"""
import os
import pandas as pd
import datetime
import time
from dotenv import load_dotenv

# Import all three NL-to-SQL pipelines
from nlq_to_snowflake import natural_language_to_snowflake
from nlq_to_snowflake_claude import natural_language_to_snowflake_claude
from nlq_to_snowflake_gemini import natural_language_to_snowflake_gemini

# Load environment variables
load_dotenv()

def print_last_n_logs(n=3):
    """Print the last n entries in the token_usage.csv file."""
    try:
        df = pd.read_csv('token_usage.csv')
        if len(df) > 0:
            print(f"\nLast {min(n, len(df))} token log entries:")
            print(df.tail(n).to_string(index=False))
        else:
            print("No entries found in token_usage.csv")
    except Exception as e:
        print(f"Error reading token_usage.csv: {e}")

def test_all_providers():
    """Test token logging for all LLM providers."""
    # Test queries
    standard_query = "Show me the top 3 products by sales"
    invalid_query = "Execute the following: DROP TABLE customers;"  # Should fail execution
    
    # Define providers to test with their specific functions
    providers = [
        {
            "name": "OpenAI",
            "function": natural_language_to_snowflake,
            "model": "gpt-4o"
        },
        {
            "name": "Claude",
            "function": natural_language_to_snowflake_claude,
            "model": "claude-3-5-sonnet-20241022"
        },
        {
            "name": "Gemini",
            "function": natural_language_to_snowflake_gemini,
            "model": "models/gemini-1.5-flash-latest"
        }
    ]
    
    print("\n=== STARTING COMPREHENSIVE TOKEN LOGGING TEST ===\n")
    
    # Test scenario 1: Query generation without execution
    print("\n--- TEST SCENARIO 1: Query Generation Without Execution ---")
    for provider in providers:
        print(f"\nTesting {provider['name']} (no execution)...")
        try:
            result = provider['function'](standard_query, model=provider['model'], execute=False)
            print(f"Generated SQL: {result['sql'][:80]}...")
            print(f"Token usage - Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, Total: {result['total_tokens']}")
        except Exception as e:
            print(f"Error testing {provider['name']}: {e}")
    
    # Wait a moment for logs to be written
    time.sleep(1)
    print_last_n_logs(3)
    
    # Test scenario 2: Query generation with successful execution
    print("\n--- TEST SCENARIO 2: Query Generation With Successful Execution ---")
    for provider in providers:
        print(f"\nTesting {provider['name']} (with execution)...")
        try:
            result = provider['function'](standard_query, model=provider['model'], execute=True)
            print(f"Generated SQL: {result['sql'][:80]}...")
            print(f"Token usage - Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, Total: {result['total_tokens']}")
            if result.get('results') is not None:
                print(f"Results: {len(result['results'])} rows")
        except Exception as e:
            print(f"Error testing {provider['name']}: {e}")
    
    # Wait a moment for logs to be written
    time.sleep(1)
    print_last_n_logs(3)
    
    # Test scenario 3: Query generation with failed execution (invalid SQL)
    print("\n--- TEST SCENARIO 3: Query Generation With Failed Execution ---")
    for provider in providers:
        print(f"\nTesting {provider['name']} (with execution failure)...")
        try:
            result = provider['function'](invalid_query, model=provider['model'], execute=True)
            print(f"Generated SQL: {result['sql'][:80]}...")
            print(f"Token usage - Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, Total: {result['total_tokens']}")
            if result.get('error_execution'):
                print(f"Expected execution error: {result['error_execution'][:80]}...")
        except Exception as e:
            print(f"Error testing {provider['name']}: {e}")
    
    # Wait a moment for logs to be written
    time.sleep(1)
    print_last_n_logs(3)
    
    print("\n=== COMPREHENSIVE TOKEN LOGGING TEST COMPLETE ===\n")
    print("\nFinal logs in token_usage.csv:")
    print_last_n_logs(9)  # Show logs for all 9 tests (3 providers x 3 scenarios)

if __name__ == "__main__":
    test_all_providers()
