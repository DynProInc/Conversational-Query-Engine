"""
FastAPI app to serve prompt/query history from token_usage.csv for the last 30 days.
"""
import os
import csv
import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

# Create router for better integration with main API
router = APIRouter()
app = FastAPI()  # Keep for standalone usage

TOKEN_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_usage.csv')

@router.get("/prompt_query_history", operation_id="prompt_query_history_api_get")
@app.get("/prompt_query_history", operation_id="prompt_query_history_api_app_get")  # Keep for standalone usage
def prompt_query_history() -> JSONResponse:
    """
    Returns the last 30 days of prompt/query history from token_usage.csv as JSON.
    """
    results: List[Dict[str, Any]] = []
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=30)

    if not os.path.exists(TOKEN_USAGE_FILE):
        return JSONResponse(content={"error": "token_usage.csv not found"}, status_code=404)

    # Try with different encodings since the file might have mixed encodings
    results = []
    try:
        with open(TOKEN_USAGE_FILE, 'r', newline='', encoding='utf-8-sig') as csvfile:
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
    except UnicodeDecodeError:
        # Try with latin-1 encoding which is more forgiving
        try:
            with open(TOKEN_USAGE_FILE, 'r', newline='', encoding='latin-1') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamp_str = row.get('timestamp')
                    if not timestamp_str:
                        continue
                    try:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str)
                    except Exception:
                        continue
                    if timestamp >= cutoff:
                        results.append(row)
        except Exception as e:
            return JSONResponse(content={"error": f"Error reading token_usage.csv: {str(e)}"}, status_code=500)

    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return JSONResponse(content=results)

# Include router in app for standalone usage
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prompt_query_history_api:app", host="0.0.0.0", port=8081, reload=True)
