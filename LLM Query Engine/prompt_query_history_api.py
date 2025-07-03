"""
FastAPI app to serve prompt/query history from token_usage.csv for the last 30 days.
"""
import os
import csv
import datetime
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

TOKEN_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_usage.csv')

@app.get("/prompt_query_history")
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
            # Expecting 'timestamp' field in ISO format
            timestamp_str = row.get('timestamp')
            if not timestamp_str:
                continue
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except Exception:
                continue
            if timestamp >= cutoff:
                results.append(row)

    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prompt_query_history_api:app", host="0.0.0.0", port=8081, reload=True)
