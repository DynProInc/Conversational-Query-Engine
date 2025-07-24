"""API routes for the Feedback Loop system.

These routes allow frontend / Swagger users to submit feedback on query
results and retrieve aggregated feedback for inspection.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.feedback_service import get_feedback_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    client_id: str = Field(..., description="Client identifier – matches the one used in /query/unified")
    prompt: str = Field(..., description="Original natural language prompt")
    model: Optional[str] = Field(None, description="LLM model that produced the response")
    generated_sql: Optional[str] = Field(None, description="SQL produced by the system")
    feedback: str = Field(..., description="Free-form user feedback / correction / explanation")
    corrected_sql: Optional[str] = Field(None, description="If user provides the correct SQL")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Optional star rating 1–5")
    user: Optional[str] = Field(None, description="Username or identifier of the human providing feedback")

class FeedbackResponse(BaseModel):
    status: str
    message: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Endpoint for users to submit feedback on a query result."""
    service = get_feedback_service()
    try:
        payload = req.dict()
        # Swagger UI uses the literal string "string" as the default example for
        # optional text fields. If the user leaves those untouched they should
        # be treated as missing, NOT as the actual value. Remove such placeholders.
        for key, value in list(payload.items()):
            if isinstance(value, str) and value.strip().lower() == "string":
                payload[key] = None
                if key in ("corrected_sql", "generated_sql"):
                    # Drop keys that would otherwise short-circuit normal flow
                    payload.pop(key, None)
        payload["timestamp"] = datetime.utcnow().isoformat()
        service.record(payload)

        # Invalidate existing cached result for this prompt if model provided
        if req.model:
            try:
                from services.cache_service import CacheService
                CacheService().invalidate_query_cache(
                    query=req.prompt,
                    client_id=req.client_id,
                    model=req.model
                )
            except Exception:
                logger.warning("Failed to invalidate cache for feedback prompt")

        # Additionally, clear file-based cache decorator entry for this client
        try:
            from cache.cache_decorator import clear_cache
            clear_cache(client_id=req.client_id)
        except Exception:
            logger.warning("Failed to clear file cache for client after feedback")

        return FeedbackResponse(status="success", message="Feedback recorded. Thank you!")
    except Exception as e:
        logger.exception("Failed to save feedback")
        raise HTTPException(status_code=500, detail="Could not save feedback") from e


class FeedbackListResponse(BaseModel):
    feedback: List[Dict[str, Any]]

@router.get("/list", response_model=FeedbackListResponse)
async def list_feedback(client_id: Optional[str] = None, limit: int = 100):
    """Get recent feedback entries (optionally filtered by client)."""
    service = get_feedback_service()
    entries = service.list(client_id=client_id, limit=limit)
    return FeedbackListResponse(feedback=entries)
