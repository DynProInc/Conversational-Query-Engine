"""
Utilities for analyzing SQL structure changes using sqlglot
"""
import re
import json
import sqlglot
from sqlglot import parse, exp, ParseError
from sqlglot.errors import ErrorLevel, TokenError
from typing import Dict, List, Tuple, Union, Any, Optional, Set

def get_sql_structure(sql: str) -> Dict:
    """
    Extract SQL structure using sqlglot parser
    Returns dict with relevant structural elements
    """
    try:
        # Parse SQL using sqlglot
        parsed = sqlglot.parse_one(sql)
        
        # Get SELECT columns
        select_columns = []
        if hasattr(parsed, 'expressions'):
            for expr in parsed.expressions:
                if hasattr(expr, 'alias') and expr.alias:
                    select_columns.append(expr.alias)
                elif hasattr(expr, 'name'):
                    select_columns.append(expr.name)
        
        # Get GROUP BY columns
        group_by = []
        if hasattr(parsed, 'group') and parsed.group:
            for expr in parsed.group:
                if hasattr(expr, 'name'):
                    group_by.append(expr.name)
        
        return {
            "select_columns": select_columns,
            "group_by": group_by
        }
    except Exception as e:
        print(f"Error parsing SQL with sqlglot: {e}")
        # Fallback to simpler approach
        return extract_sql_parts_regex(sql)

def extract_sql_parts_regex(sql: str) -> Dict[str, str]:
    """Fallback function to extract SQL parts for comparison using regex"""
    sql = sql.upper()
    where_idx = sql.find("WHERE")
    group_idx = sql.find("GROUP BY")
    
    select_from = ""
    where_clause = ""
    group_order_limit = ""
    
    if where_idx > 0:
        select_from = sql[:where_idx].strip()
        rest = sql[where_idx:]
        
        if group_idx > where_idx:
            where_clause = sql[where_idx:group_idx].strip()
            group_order_limit = sql[group_idx:].strip()
        else:
            group_idx = sql.find("ORDER BY")
            if group_idx > where_idx:
                where_clause = sql[where_idx:group_idx].strip()
                group_order_limit = sql[group_idx:].strip()
            else:
                group_idx = sql.find("LIMIT")
                if group_idx > where_idx:
                    where_clause = sql[where_idx:group_idx].strip()
                    group_order_limit = sql[group_idx:].strip()
                else:
                    where_clause = rest
    else:
        select_from = sql
    
    return {
        "select_from": select_from,
        "where_clause": where_clause,
        "group_order_limit": group_order_limit
    }

def is_column_structure_changed(original_sql: str, edited_sql: str) -> bool:
    """
    Determine if the edited SQL has a different column structure
    using sqlglot for parsing
    
    Returns True if column structure has changed, False if only filtering changed
    """
    try:
        # Get structure using sqlglot
        original = get_sql_structure(original_sql)
        edited = get_sql_structure(edited_sql)
        
        # Compare select columns (if parsing was successful)
        if "select_columns" in original and "select_columns" in edited:
            if set(original["select_columns"]) != set(edited["select_columns"]):
                return True
                
        # Compare group by
        if "group_by" in original and "group_by" in edited:
            if set(original["group_by"]) != set(edited["group_by"]):
                return True
                
        # If we couldn't parse properly, compare raw SQL excluding WHERE clauses
        if "raw_sql" in original or "raw_sql" in edited:
            # Extract parts before WHERE and after ORDER BY/LIMIT
            orig_parts = extract_sql_parts_regex(original_sql)
            edit_parts = extract_sql_parts_regex(edited_sql)
            
            if orig_parts["select_from"] != edit_parts["select_from"]:
                return True
            if orig_parts["group_order_limit"] != edit_parts["group_order_limit"]:
                return True
        
        # If we get here, structure is similar enough
        return False
        
    except Exception as e:
        print(f"Error comparing SQL structure: {e}")
        return True  # If we can't determine, assume structure changed

# We'll use the existing chart instructions from claude_query_generator.py
# No need for a separate chart prompt generation function


class SQLValidationError(Exception):
    """Custom exception for SQL validation errors"""
    pass


def validate_read_only_sql(sql: str) -> Union[bool, Dict[str, str]]:
    """
    Validates that a SQL query only contains read operations and follows security rules.
    Uses both regex pattern matching and sqlglot SQL parser for robust validation.
    
    Rules enforced:
    1. Only SELECT statements and WITH clauses are allowed
    2. WITH clauses (Common Table Expressions) are allowed if they only contain SELECT operations
    3. No SELECT * allowed
    4. LIMIT required for non-aggregate/non-GROUP BY queries (will be added by the caller)
    5. For security reasons, SHOW, DESCRIBE, and EXPLAIN are not allowed
    
    Args:
        sql: The SQL query to validate
        
    Returns:
        If valid: True
        If invalid: Dict with 'error' key containing error message
    """
    # Normalize SQL for validation
    sql = sql.strip()
    
    # Convert to uppercase for easier pattern matching, but keep original for error messages
    sql_upper = sql.upper()
    
    # 1. Check for forbidden operations
    # These need to be checked only when they appear as commands at beginning of statements
    # We don't want to block them when they appear as part of table or column names
    forbidden_operations = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "MERGE", "COPY", "PUT", "GET", "GRANT", "REVOKE",
        "SHOW", "EXPLAIN"
    ]
    
    # For DESC/DESCRIBE, we handle them separately to allow ORDER BY ... DESC
    
    # Special handling for commands at the BEGINNING OF STATEMENTS ONLY
    # Using word boundary pattern to ensure we're only matching commands at the start,
    # not when they appear as part of table or column names
    
    # Check for DESCRIBE/DESC as commands (at beginning of statement only)
    if re.search(r"^\s*DESC(RIBE)?\b[\s(][\"\'\w\d_\.]", sql_upper):
        return {
            "error": "DESCRIBE/DESC commands are not allowed. Only SELECT statements and WITH clauses are permitted."
        }
        
    # Check for SHOW as a command (at beginning of statement only)
    if re.search(r"^\s*SHOW\b[\s(][\"\'\w\d_\.]", sql_upper):
        return {
            "error": "SHOW commands are not allowed. Only SELECT statements and WITH clauses are permitted."
        }
        
    # Check for EXPLAIN as a command (at beginning of statement only)
    if re.search(r"^\s*EXPLAIN\b[\s(][\"\'\w\d_\.]", sql_upper):
        return {
            "error": "EXPLAIN commands are not allowed. Only SELECT statements and WITH clauses are permitted."
        }
    
    # Check if query starts with any other forbidden operation
    for op in forbidden_operations:
        # Skip the ones we've already handled with special logic
        if op in ["DESC", "DESCRIBE", "SHOW", "EXPLAIN"]:
            continue
            
        # Using word boundary \b to ensure we only match the complete command word
        # This ensures we don't match when these words appear as part of other words in table/column names
        pattern = fr"^\s*{op}\b[\s(][\"\'\w\d_\.]"
        if re.search(pattern, sql_upper):
            return {
                "error": f"Operation not allowed: {op} statements are forbidden. Only SELECT statements and WITH clauses are permitted."
            }
    
    # 2. Check if query is allowed type
    allowed_prefixes = ["SELECT", "WITH"]
    is_allowed = False
    
    for prefix in allowed_prefixes:
        if sql_upper.strip().startswith(prefix):
            is_allowed = True
            break
    
    if not is_allowed:
        return {
            "error": "Invalid query. Only SELECT statements and WITH clauses are allowed."
        }
        
    # Use sqlglot to parse the query and validate it more robustly
    try:
        # Parse the SQL query with Snowflake dialect
        parsed = sqlglot.parse(sql, dialect='snowflake')
        
        # Check if any expressions in the parsed SQL are forbidden operations
        for expression in parsed:
            # Only allow SELECT and WITH statements
            if isinstance(expression, sqlglot.exp.Select):
                # Check for SELECT * (not allowed)
                for col in expression.find_all(sqlglot.exp.Star):
                    if not col.table:  # Allow t.* but not *
                        return {"error": "'SELECT *' is not allowed. Please specify explicit column names."}
                        
            elif isinstance(expression, sqlglot.exp.With):
                # Ensure WITH clause contains only SELECT operations
                selects_found = False
                for cte in expression.find_all(sqlglot.exp.CTE):
                    query = cte.args.get('query')
                    if query and isinstance(query, sqlglot.exp.Select):
                        selects_found = True
                    else:
                        return {"error": "WITH clauses must only contain SELECT operations."}
                        
                # Check that there's a SELECT after the WITH clause
                if not selects_found and not expression.find(sqlglot.exp.Select):
                    return {"error": "WITH clause must be followed by a SELECT statement."}
            else:
                # Not a SELECT or WITH statement
                return {
                    "error": f"Operation not allowed: {type(expression).__name__} statements are forbidden. Only SELECT statements and WITH clauses are permitted."
                }
    except ParseError as e:
        # SQL syntax error - provide a user-friendly message
        return {"error": f"SQL syntax error: {str(e)}"}
    except Exception as e:
        # Fallback to regex validation if sqlglot parsing fails
        # But log the exception for debugging
        print(f"sqlglot parsing failed: {str(e)}, falling back to regex validation")
        
    # 2a. For WITH clauses, ensure they only contain SELECT operations
    if sql_upper.startswith("WITH"):
        # Check for forbidden operations within the WITH clause
        for op in forbidden_operations:
            # Use a more flexible pattern that works with multi-line SQL and formatting
            if re.search(fr"WITH[\s\S]*AS[\s\S]*\([\s\S]*{op}\s", sql_upper):
                return {
                    "error": f"WITH clauses must only contain SELECT operations. Found forbidden operation: {op}."
                }
        
        # More flexible pattern for checking WITH clause structure
        # First check if there's an AS ( pattern after WITH
        if not re.search(r"WITH\s+.*\s+AS\s*\(", sql_upper, re.DOTALL):
            return {
                "error": "WITH clause must define a table expression with 'AS ('."
            }
            
        # Then check if there's a SELECT after a closing parenthesis
        if not re.search(r"\)[\s\S]*SELECT", sql_upper, re.DOTALL):
            return {
                "error": "WITH clause must be followed by a SELECT statement."
            }
    
    # 3. For SELECT statements, perform additional validations
    if sql_upper.startswith("SELECT"):
        # 3a. Check for SELECT *
        # Check for SELECT * FROM or SELECT * pattern
        select_star_pattern = r"SELECT\s+\*\s+FROM|SELECT\s+\*\s*$"
        if re.search(select_star_pattern, sql_upper):
            return {
                "error": "'SELECT *' is not allowed. Please specify explicit column names."
            }
    
    # If all checks pass, return True
    return True
