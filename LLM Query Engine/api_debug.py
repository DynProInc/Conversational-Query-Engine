"""
API Debug Endpoint - Capture prints and logs for monitoring
"""
import io
import sys
import threading
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Create a router
router = APIRouter(tags=["debug"])

# In-memory log storage
class LogCapture:
    logs = []
    lock = threading.Lock()
    
    @classmethod
    def add_log(cls, message):
        with cls.lock:
            cls.logs.append(message)
            # Keep only the last 1000 log entries
            if len(cls.logs) > 1000:
                cls.logs = cls.logs[-1000:]
    
    @classmethod
    def get_logs(cls, limit=100):
        with cls.lock:
            return cls.logs[-limit:] if cls.logs else []
    
    @classmethod
    def clear_logs(cls):
        with cls.lock:
            cls.logs = []

# Original print function
original_print = print

# Override print to capture logs
def capturing_print(*args, **kwargs):
    # Call original print
    original_print(*args, **kwargs)
    
    # Capture the output
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    message = output.getvalue()
    LogCapture.add_log(message)

# Replace print with our capturing version
sys.print = capturing_print

# API Models
class LogResponse(BaseModel):
    logs: List[str]
    count: int

@router.get("/logs", response_model=LogResponse)
def get_logs(limit: int = 100):
    """Get the captured logs"""
    logs = LogCapture.get_logs(limit)
    return {"logs": logs, "count": len(logs)}

@router.post("/logs/clear")
def clear_logs():
    """Clear all logs"""
    LogCapture.clear_logs()
    return {"status": "logs cleared"}

# Enable capturing all print statements
def enable_log_capture():
    # Monkey patch print
    sys.print = capturing_print
