from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import uuid
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from utils.file_csv_logger import DATA_DIR
from services.feedback_manager import FeedbackManager
import sqlparse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackIn(BaseModel):
    execution_id: str = Field(..., description="UUID returned by /query/unified")
    type: str = Field(..., pattern="^(thumbs_up|thumbs_down|correction|suggestion)$")
    text: str | None = None
    corrected_query: str | None = None
    user_id: str | None = None

class FeedbackEntry(BaseModel):
    feedback_id: str
    execution_id: str
    timestamp: str
    type: str
    text: str
    corrected_query: str
    user_id: str
    prompt: Optional[str] = None
    model: Optional[str] = None

class ExecutionEntry(BaseModel):
    execution_id: str
    timestamp: str
    client_id: str
    prompt: str
    model: str
    generated_query: str
    formatted_query: str
    success: bool
    error_message: str

class FeedbackHistoryResponse(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    items: List[FeedbackEntry]

class ExecutionHistoryResponse(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    items: List[ExecutionEntry]

@router.post("/", status_code=201)
async def submit_feedback(pb: FeedbackIn):
    fb_id = str(uuid.uuid4())
    FeedbackManager.log_feedback(
        fb_id=fb_id,
        exec_id=pb.execution_id,
        fb_type=pb.type,
        text=pb.text or "",
        corrected_query=pb.corrected_query or "",
        user_id=pb.user_id or "anon"
    )
    return {"feedback_id": fb_id, "msg": "stored"}

@router.get("/history", response_model=FeedbackHistoryResponse)
async def get_feedback_history(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Number of items per page", ge=1, le=100),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    include_executions: bool = Query(False, description="Include execution details with each feedback entry")
):
    """
    Get feedback history with optional filtering by client ID.
    Returns the most recent feedback entries first.
    """
    feedback_path = DATA_DIR / "feedback.csv"
    executions_path = DATA_DIR / "executions.csv"
    
    if not feedback_path.exists():
        return FeedbackHistoryResponse(
            total_items=0,
            total_pages=0,
            current_page=page,
            page_size=page_size,
            items=[]
        )
    
    try:
        # Read feedback data
        feedback_df = pd.read_csv(feedback_path)
        
        # Sort by timestamp (newest first)
        feedback_df = feedback_df.sort_values("timestamp", ascending=False)
        
        # If execution details are requested, merge with executions data
        if include_executions and executions_path.exists():
            try:
                # Load execution data
                executions_df = pd.read_csv(executions_path)
                
                # Filter by client_id if specified for executions
                if client_id:
                    filtered_executions_df = executions_df[executions_df["client_id"] == client_id]
                else:
                    filtered_executions_df = executions_df
                
                # Create a dictionary of execution details for faster lookup
                execution_details = {}
                for _, row in executions_df.iterrows():  # Use full executions_df to ensure all execution details are available
                    exec_id = row['execution_id']
                    execution_details[exec_id] = {
                        'prompt': row['prompt'] if pd.notna(row['prompt']) else "",
                        'model': row['model'] if pd.notna(row['model']) else ""
                    }
                
                logger.info(f"Loaded {len(execution_details)} execution details")
                
                # Add execution details to feedback entries directly
                merged_df = feedback_df.copy()
                merged_df['prompt'] = ""
                merged_df['model'] = ""
                
                # Manually populate prompt and model fields
                populated_count = 0
                for i, row in merged_df.iterrows():
                    exec_id = row['execution_id']
                    if exec_id in execution_details:
                        merged_df.at[i, 'prompt'] = execution_details[exec_id]['prompt']
                        merged_df.at[i, 'model'] = execution_details[exec_id]['model']
                        populated_count += 1
                
                logger.info(f"Feedback entries with execution details populated: {populated_count} out of {len(merged_df)}")
                logger.info(f"Sample populated entry - prompt length: {len(merged_df['prompt'].iloc[0]) if len(merged_df) > 0 else 0}, model: {merged_df['model'].iloc[0] if len(merged_df) > 0 else 'none'}")
                
                # Debug: Check if any execution IDs from feedback are missing in executions
                feedback_exec_ids = set(feedback_df['execution_id'])
                execution_exec_ids = set(executions_df['execution_id'])
                missing_ids = feedback_exec_ids - execution_exec_ids
                if missing_ids:
                    logger.warning(f"Found {len(missing_ids)} execution IDs in feedback that are missing in executions: {list(missing_ids)[:5]}...")
                else:
                    logger.info("All feedback execution IDs found in executions data")
                
                # Ensure all fields are strings to avoid serialization issues
                merged_df['prompt'] = merged_df['prompt'].astype(str)
                merged_df['model'] = merged_df['model'].astype(str)
                
            except Exception as e:
                logger.error(f"Error adding execution details: {str(e)}")
                # If the process fails, continue without execution details
                merged_df = feedback_df.copy()
                merged_df["prompt"] = ""
                merged_df["model"] = ""
            
            # Count total entries after client filtering
            total_items = len(merged_df)
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            paginated_df = merged_df.iloc[offset:offset+page_size]
            
            # Convert to list of dictionaries
            entries = paginated_df.fillna("").to_dict("records")
        else:
            try:
                # Even if include_executions is false, we'll still try to get execution details
                # but we won't filter by client_id to ensure we get as many matches as possible
                if executions_path.exists():
                    # Load execution data
                    executions_df = pd.read_csv(executions_path)
                    
                    # Create a dictionary of execution details for faster lookup
                    execution_details = {}
                    for _, row in executions_df.iterrows():
                        exec_id = row['execution_id']
                        execution_details[exec_id] = {
                            'prompt': row['prompt'] if pd.notna(row['prompt']) else "",
                            'model': row['model'] if pd.notna(row['model']) else ""
                        }
                    
                    logger.info(f"Loaded {len(execution_details)} execution details for basic feedback history")
                    
                    # Apply pagination without filtering by client_id
                    total_items = len(feedback_df)
                    total_pages = (total_items + page_size - 1) // page_size
                    offset = (page - 1) * page_size
                    paginated_df = feedback_df.iloc[offset:offset+page_size]
                    
                    # Add execution details to feedback entries directly
                    merged_df = paginated_df.copy()
                    merged_df['prompt'] = ""
                    merged_df['model'] = ""
                    
                    # Manually populate prompt and model fields
                    populated_count = 0
                    for i, row in merged_df.iterrows():
                        exec_id = row['execution_id']
                        if exec_id in execution_details:
                            merged_df.at[i, 'prompt'] = execution_details[exec_id]['prompt']
                            merged_df.at[i, 'model'] = execution_details[exec_id]['model']
                            populated_count += 1
                    
                    logger.info(f"Basic feedback entries with execution details populated: {populated_count} out of {len(merged_df)}")
                    
                    # Ensure all fields are strings to avoid serialization issues
                    merged_df['prompt'] = merged_df['prompt'].astype(str)
                    merged_df['model'] = merged_df['model'].astype(str)
                    
                    entries = merged_df.fillna("").to_dict("records")
                else:
                    # If executions file doesn't exist, proceed without execution details
                    total_items = len(feedback_df)
                    total_pages = (total_items + page_size - 1) // page_size
                    offset = (page - 1) * page_size
                    paginated_df = feedback_df.iloc[offset:offset+page_size]
                    entries = paginated_df.fillna("").to_dict("records")
                    
                    # Add empty prompt and model fields
                    for entry in entries:
                        entry["prompt"] = ""
                        entry["model"] = ""
            except Exception as e:
                logger.error(f"Error adding execution details to basic feedback: {str(e)}")
                # If the process fails, continue without execution details
                total_items = len(feedback_df)
                total_pages = (total_items + page_size - 1) // page_size
                offset = (page - 1) * page_size
                paginated_df = feedback_df.iloc[offset:offset+page_size]
                entries = paginated_df.fillna("").to_dict("records")
                
                # Add empty prompt and model fields
                for entry in entries:
                    entry["prompt"] = ""
                    entry["model"] = ""
        
        # Create FeedbackEntry objects with explicit prompt and model fields
        feedback_entries = []
        for entry in entries:
            # Ensure all fields are properly handled, especially prompt and model
            try:
                feedback_entries.append(FeedbackEntry(
                    feedback_id=entry["feedback_id"],
                    execution_id=entry["execution_id"],
                    timestamp=entry["timestamp"],
                    type=entry["type"],
                    text=entry["text"],
                    corrected_query=entry["corrected_query"] if "corrected_query" in entry and entry["corrected_query"] else "",
                    user_id=entry["user_id"] if "user_id" in entry and entry["user_id"] else "anonymous",
                    prompt=entry["prompt"] if "prompt" in entry and entry["prompt"] else "",
                    model=entry["model"] if "model" in entry and entry["model"] else ""
                ))
                
                # Log the first entry for debugging
                if len(feedback_entries) == 1:
                    logger.info(f"First feedback entry - prompt length: {len(feedback_entries[0].prompt)}, model: {feedback_entries[0].model}")
            except Exception as e:
                logger.error(f"Error creating FeedbackEntry: {str(e)} for entry: {entry}")
                # Create a minimal entry if there's an error
                feedback_entries.append(FeedbackEntry(
                    feedback_id=entry.get("feedback_id", "unknown"),
                    execution_id=entry.get("execution_id", "unknown"),
                    timestamp=entry.get("timestamp", ""),
                    type=entry.get("type", ""),
                    text=entry.get("text", ""),
                    corrected_query="",
                    user_id="anonymous",
                    prompt="",
                    model=""
                ))
        
        return FeedbackHistoryResponse(
            total_items=total_items,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
            items=feedback_entries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback history: {str(e)}")

@router.get("/executions", response_model=ExecutionHistoryResponse)
async def get_execution_history(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Number of items per page", ge=1, le=100),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    model: Optional[str] = Query(None, description="Filter by model"),
    success_only: bool = Query(False, description="Only show successful executions")
):
    """
    Get execution history with optional filtering.
    Returns the most recent executions first with properly formatted SQL queries.
    """
    executions_path = DATA_DIR / "executions.csv"
    
    if not executions_path.exists():
        return ExecutionHistoryResponse(
            total_items=0,
            total_pages=0,
            current_page=page,
            page_size=page_size,
            items=[]
        )
    
    try:
        # Read executions data
        executions_df = pd.read_csv(executions_path)
        
        # Apply filters
        if client_id:
            executions_df = executions_df[executions_df["client_id"] == client_id]
        
        if model:
            executions_df = executions_df[executions_df["model"].str.contains(model, case=False)]
        
        if success_only:
            executions_df = executions_df[executions_df["success"] == True]
        
        # Get total count after filtering
        total_items = len(executions_df)
        total_pages = (total_items + page_size - 1) // page_size
        
        # Sort by timestamp (newest first)
        executions_df = executions_df.sort_values("timestamp", ascending=False)
        
        # Apply pagination
        offset = (page - 1) * page_size
        paginated_df = executions_df.iloc[offset:offset+page_size]
        
        # Format SQL queries
        entries = []
        for _, row in paginated_df.iterrows():
            entry_dict = row.fillna("").to_dict()
            
            # Format the SQL query if it exists
            if entry_dict["generated_query"]:
                try:
                    formatted_query = sqlparse.format(
                        entry_dict["generated_query"],
                        reindent=True,
                        keyword_case='upper',
                        indent_width=4
                    )
                    entry_dict["formatted_query"] = formatted_query
                except Exception:
                    # If formatting fails, use the original query
                    entry_dict["formatted_query"] = entry_dict["generated_query"]
            else:
                entry_dict["formatted_query"] = ""
            
            entries.append(entry_dict)
        
        return ExecutionHistoryResponse(
            total_items=total_items,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
            items=[ExecutionEntry(**entry) for entry in entries]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving execution history: {str(e)}")

@router.get("/execution/{execution_id}")
async def get_execution_details(execution_id: str):
    """
    Get detailed information about a specific execution including any feedback received.
    """
    executions_path = DATA_DIR / "executions.csv"
    feedback_path = DATA_DIR / "feedback.csv"
    
    if not executions_path.exists():
        raise HTTPException(status_code=404, detail="Executions data not found")
    
    try:
        # Read executions data
        executions_df = pd.read_csv(executions_path)
        
        # Find the specific execution
        execution = executions_df[executions_df["execution_id"] == execution_id]
        
        if execution.empty:
            raise HTTPException(status_code=404, detail=f"Execution with ID {execution_id} not found")
        
        # Convert to dictionary
        execution_dict = execution.iloc[0].fillna("").to_dict()
        
        # Format the SQL query
        if execution_dict["generated_query"]:
            try:
                formatted_query = sqlparse.format(
                    execution_dict["generated_query"],
                    reindent=True,
                    keyword_case='upper',
                    indent_width=4
                )
                execution_dict["formatted_query"] = formatted_query
            except Exception:
                execution_dict["formatted_query"] = execution_dict["generated_query"]
        else:
            execution_dict["formatted_query"] = ""
        
        # Get associated feedback if available
        feedback_entries = []
        if feedback_path.exists():
            feedback_df = pd.read_csv(feedback_path)
            feedback = feedback_df[feedback_df["execution_id"] == execution_id]
            
            if not feedback.empty:
                feedback_entries = [FeedbackEntry(**entry) for entry in feedback.fillna("").to_dict("records")]
        
        # Create response with execution details and feedback entries
        response = {
            "execution_id": execution_dict["execution_id"],
            "timestamp": execution_dict["timestamp"],
            "client_id": execution_dict["client_id"],
            "prompt": execution_dict["prompt"],
            "model": execution_dict["model"],
            "generated_query": execution_dict["generated_query"],
            "formatted_query": execution_dict["formatted_query"],
            "success": bool(execution_dict["success"]),
            "error_message": execution_dict["error_message"],
            "feedback_entries": feedback_entries
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving execution details: {str(e)}")

