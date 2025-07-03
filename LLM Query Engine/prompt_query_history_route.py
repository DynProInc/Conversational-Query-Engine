import os
import csv
import datetime
from typing import List, Dict, Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

TOKEN_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_usage.csv')

@router.get("/prompt_query_history")
def prompt_query_history() -> JSONResponse:
    """
    Returns the last 30 days of prompt/query history from token_usage.csv as JSON.
    """
    results: List[Dict[str, Any]] = []
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=30)

    if not os.path.exists(TOKEN_USAGE_FILE):
        return JSONResponse(content={"error": "token_usage.csv not found"}, status_code=404)

    with open(TOKEN_USAGE_FILE, 'r', newline='', encoding='utf-8') as csvfile:
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

    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda x: x.get('timestamp', x.get('date', '')), reverse=True)
    return JSONResponse(content=results)
