"""
Test script to verify that Claude respects numeric limits in user queries.

This script tests the Claude query generator with queries containing explicit numeric limits,
both as digits (5) and as words (five).
"""
import os
from dotenv import load_dotenv
from nlq_to_snowflake_claude import nlq_to_snowflake_claude
from claude_query_generator import extract_limit_from_query

# Load environment variables
load_dotenv()

def test_limit_extraction():
    """Test the limit extraction function directly."""
    
    test_cases = [
        # Digit-based limits
        ("Give me top 5 sales person as per region", 5),
        ("Show me the first 3 customers by revenue", 3),
        # Word-based limits
        ("Give me top five sales person as per region", 5),
        ("Show me the first three customers by revenue", 3),
        # Compound word-based limits
        ("Show me twenty-five top products", 25)
    ]
    
    print("=== TESTING LIMIT EXTRACTION ===\n")
    
    for query, expected in test_cases:
        extracted = extract_limit_from_query(query)
        result = "✅ PASS" if extracted == expected else f"❌ FAIL (got {extracted}, expected {expected})"
        print(f"Query: '{query}'\nExtracted limit: {extracted} - {result}\n")
    
    print("=== LIMIT EXTRACTION TEST COMPLETE ===\n")

def test_claude_limit_handling():
    """Test Claude's handling of numeric limits in queries."""
    
    # Test queries with explicit numeric limits (both digits and words)
    test_queries = [
        # Digit-based limits
        "Give me top 5 sales person as per region",
        "Show me the first 3 customers by revenue",
        # Word-based limits
        "Give me top five sales person as per region",
        "Show me the first three customers by revenue",
        # Compound word-based limits
        "Show me twenty-five top products"
    ]
    
    print("=== TESTING CLAUDE LIMIT HANDLING ===\n")
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            # First check what limit we expect to extract
            expected_limit = extract_limit_from_query(query)
            print(f"Expected limit: {expected_limit}")
            
            # Generate SQL without execution
            result = nlq_to_snowflake_claude(query, execute=False)
            
            # Extract and print the SQL
            sql = result.get("sql", "")
            print(f"\nGenerated SQL:\n{sql}\n")
            
            # Check if LIMIT is present in the SQL
            if "LIMIT" in sql.upper():
                print("✅ LIMIT clause found in SQL")
                # Extract the limit value
                import re
                limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
                if limit_match:
                    limit_value = int(limit_match.group(1))
                    print(f"   Limit value: {limit_value}")
                    
                    # Check if the extracted limit matches what we expect
                    if expected_limit is not None and limit_value == expected_limit:
                        print(f"   ✅ Correct limit value ({expected_limit})")
                    else:
                        print(f"   ❌ Incorrect limit value (expected {expected_limit})")
            else:
                print("❌ No LIMIT clause found in SQL")
                
        except Exception as e:
            print(f"Error testing query: {str(e)}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    # Test the limit extraction function first
    test_limit_extraction()
    
    # Then test the full Claude pipeline
    test_claude_limit_handling()
