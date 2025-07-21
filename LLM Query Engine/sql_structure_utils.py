"""
Utilities for analyzing SQL structure changes using sqlglot
"""
import re
import sqlglot
from typing import Dict, List, Set, Tuple, Optional

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
