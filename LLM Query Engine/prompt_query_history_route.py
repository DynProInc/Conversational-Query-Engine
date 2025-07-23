import os
import csv
import datetime
from typing import List, Dict, Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

TOKEN_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_usage.csv')

@router.get("/prompt_query_history", operation_id="prompt_query_history_route_get")
def prompt_query_history(page: int = 1, page_size: int = 100) -> JSONResponse:
    """
    Returns the last 30 days of prompt/query history from token_usage.csv as JSON.
    Supports pagination to handle large responses.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page (max 1000)
    """
    # Validate pagination parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 1000:
        page_size = 100
    results: List[Dict[str, Any]] = []
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=30)

    if not os.path.exists(TOKEN_USAGE_FILE):
        return JSONResponse(content={"error": "token_usage.csv not found"}, status_code=404)

    results = []
    
    # Handle CSV file with potential encoding issues - try multiple encodings
    encodings_to_try = ['utf-8-sig', 'latin-1', 'cp1252']
    success = False
    error_message = ""
    
    for encoding in encodings_to_try:
        try:
            with open(TOKEN_USAGE_FILE, 'r', newline='', encoding=encoding) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Try both ISO and legacy datetime formats
                    timestamp_str = row.get('timestamp') or row.get('date')
                    if not timestamp_str:
                        # Try first column if unnamed
                        timestamp_str = next(iter(row.values()), None)
                    if not timestamp_str:
                        continue
                    try:
                        # Try parsing as ISO first, fallback to common datetime format
                        try:
                            timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        except Exception:
                            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        continue
                    if timestamp >= cutoff:
                        results.append(row)
            # If we got here without exception, we succeeded
            success = True
            break
        except UnicodeDecodeError as e:
            error_message = f"Failed with {encoding}: {str(e)}"
            continue
        except Exception as e:
            error_message = f"Unexpected error with {encoding}: {str(e)}"
            break
            
    if not success and not results:
        return JSONResponse(
            content={"error": f"Could not read token_usage.csv with any encoding: {error_message}"}, 
            status_code=500
        )

    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda x: x.get('timestamp', x.get('date', '')), reverse=True)
    
    # Calculate pagination values
    total_records = len(results)
    total_pages = (total_records + page_size - 1) // page_size
    start_index = (page - 1) * page_size
    end_index = min(start_index + page_size, total_records)
    
    # Get paginated subset
    paginated_results = results[start_index:end_index]
    
    # Create response with pagination metadata
    response = {
        "data": paginated_results,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages
        }
    }
    
    return JSONResponse(content=response)
