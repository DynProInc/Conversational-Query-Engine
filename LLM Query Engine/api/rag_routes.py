"""
API routes for RAG functionality.
This module provides FastAPI routes for document processing, retrieval, and other RAG operations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import logging
import json
import os
import tempfile
import shutil

from services.rag_service import RAGService
from services.rag_client import get_rag_service
from services.cache_service import CacheService, cache_llm_query

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}}
)

# Models for API
class Document(BaseModel):
    """Model for a document to process."""
    text: str = Field(..., description="Document content as plain text to embed and index.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata key/value pairs.")
    id: Optional[str] = Field(default=None, description="Optional unique identifier for the document. If omitted, the system will generate one.")

class ProcessDocumentsRequest(BaseModel):
    """Request model for processing documents."""
    documents: List[Document] = Field(..., description="List of Document objects to be processed and ingested.")
    collection_name: str = Field(..., description="Target vector-store collection name.")
    chunk_size: Optional[int] = Field(default=None, description="Optional chunk size (characters) when splitting documents. Uses default if not provided.")
    chunk_overlap: Optional[int] = Field(default=None, description="Optional overlap size (characters) between consecutive chunks.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata key/value pairs.")

class SchemaDefinition(BaseModel):
    """Model for a database schema definition."""
    tables: Dict[str, Any] = Field(..., description="Dictionary of table definitions keyed by table name.")
    relationships: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional list of table relationship definitions.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata key/value pairs.")

class ProcessSchemaRequest(BaseModel):
    """Request model for processing schema."""
    schema: SchemaDefinition = Field(..., description="Schema definition object describing tables and relationships.")
    collection_name: str = Field(..., description="Target vector-store collection name.")
    include_examples: Optional[bool] = Field(default=True, description="Whether to auto-generate example rows/statements for the schema.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata key/value pairs.")

class RetrieveContextRequest(BaseModel):
    """Request model for retrieving context."""
    query: str = Field(..., description="End-user natural language query.")
    collection_name: Optional[str] = None
    top_k: Optional[int] = Field(default=None, description="Number of top documents to retrieve.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters to apply during retrieval.")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional previous user/assistant turns for conversational context.")
    client_id: str = Field(default="default", description="Client identifier used for multi-tenant isolation.")

class EnhancePromptRequest(BaseModel):
    """Request model for enhancing prompts."""
    query: str = Field(..., description="End-user natural language query.")
    retrieved_context: Dict[str, Any] = Field(..., description="Context object returned by /rag/retrieve containing relevant documents.")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional previous user/assistant turns for conversational context.")
    system_prompt: Optional[str] = Field(default=None, description="Optional base system prompt to prepend to the enhanced prompt or query.")
    template_name: str = Field(default="default", description="Name of the prompt template to apply when formatting the enhanced prompt.")

class QueryWithRAGRequest(BaseModel):
    """Request model for querying with RAG."""
    query: str = Field(..., description="End-user natural language query.")
    collection_name: Optional[str] = None
    top_k: Optional[int] = Field(default=5, description="Number of documents to retrieve during the RAG step.")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional previous user/assistant turns for conversational context.")
    client_id: str = Field(default="default", description="Client identifier used for multi-tenant isolation.")
    system_prompt: Optional[str] = Field(default=None, description="Optional base system prompt to prepend to the enhanced prompt or query.")
    model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name to use for generation.")
    temperature: float = Field(default=0.7, description="Sampling temperature for the language model.")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate. Uses model default if None.")
    use_cache: bool = Field(default=True, description="If true, attempt to serve the result from cache and store new responses.")

# Get RAG service instance
def get_rag_service():
    """Get RAG service instance."""
    return RAGService()

# Get cache service instance
def get_cache_service():
    """Get cache service instance."""
    return CacheService()

@router.post("/documents", response_model=Dict[str, Any])
async def process_documents(
    request: ProcessDocumentsRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process documents for RAG.
    
    Args:
        request: Process documents request
        rag_service: RAG service
        
    Returns:
        Processing results
    """
    try:
        # Convert Document models to dictionaries
        docs = [
            {
                "text": doc.text,
                "metadata": doc.metadata or {},
                "id": doc.id
            }
            for doc in request.documents
        ]
        
        # Process documents
        result = rag_service.process_documents(
            documents=docs,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            metadata=request.metadata
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@router.post("/schema", response_model=Dict[str, Any])
async def process_schema(
    request: ProcessSchemaRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process database schema definition for RAG.
    
    Args:
        request: Process schema request
        rag_service: RAG service
        
    Returns:
        Processing results
    """
    try:
        # Convert SchemaDefinition model to dictionary
        schema_dict = {
            "tables": request.schema.tables,
            "relationships": request.schema.relationships or [],
            "metadata": request.schema.metadata or {}
        }
        
        # Process schema
        result = rag_service.process_schema(
            schema_definition=schema_dict,
            collection_name=request.collection_name,
            include_examples=request.include_examples,
            metadata=request.metadata
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing schema: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing schema: {str(e)}")

@router.post("/data-dictionary", response_model=Dict[str, Any])
async def upload_data_dictionary(
    collection_name: str = Form(..., description="Target vector-store collection name."),
    client_id: str = Form("default"),
    csv_file: UploadFile = File(...),
    table_col: Optional[str] = Form(None, description="Header name for table names (auto-detected if omitted)"),
    column_col: Optional[str] = Form(None, description="Header name for column names (auto-detected if omitted)"),
    description_col: Optional[str] = Form(None, description="Header name for column descriptions (auto-detected if omitted)"),
    background_tasks: BackgroundTasks = None,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload and process CSV data dictionary for RAG.
    
    Args:
        collection_name: Name of the dictionary collection
        client_id: Client identifier (defaults to "default")
        csv_file: CSV file to upload
        table_col: Column name for table names (defaults to "TABLE_NAME")
        column_col: Column name for column names (defaults to "COLUMN_NAME")
        description_col: Column name for descriptions (defaults to "DESCRIPTION")
        background_tasks: Background tasks
        rag_service: RAG service
        
    Returns:
        Processing results
    """
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_path = temp_file.name
        temp_file.close()
        
        # Save uploaded file to temporary file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)
        
        # -------------------------------------------
        # Auto-detect column names if any are missing
        # -------------------------------------------
        # Detect headers if any of the three are missing or blank
        if not table_col or not column_col or not description_col:
            import pandas as pd
            try:
                df_headers = pd.read_csv(temp_path, nrows=0)
            except Exception as read_err:
                logger.error(f"Failed reading CSV headers for auto-detection: {read_err}")
                raise HTTPException(status_code=400, detail="Could not read CSV file to detect headers")
            detected_columns = [c.strip() for c in df_headers.columns]
            logger.info(f"Detected CSV headers: {detected_columns}")

            def _detect(possible_keywords):
                for kw in possible_keywords:
                    for col in detected_columns:
                        normalized = col.lower().replace(" ", "").replace("_", "")
                        if kw in normalized:
                            return col
                return None

            if table_col is None:
                table_col = _detect(["tablename", "table_name", "table"])
            if column_col is None:
                column_col = _detect(["columnname", "column_name", "column", "field"])
            if description_col is None:
                description_col = _detect(["description", "desc", "definition", "column_description"])
            logger.info(f"Auto-detected headers: table={table_col}, column={column_col}, description={description_col}")

            # Validate detection
            if not table_col or not column_col or not description_col:
                missing = [n for n,v in zip(["table", "column", "description"], [table_col, column_col, description_col]) if v is None]
                raise HTTPException(status_code=400, detail=f"Could not auto-detect required headers: {', '.join(missing)}. Please specify them explicitly.")

        logger.info(f"Processing data dictionary with columns: {table_col}, {column_col}, {description_col}")
        
        # Process data dictionary
        metadata = {
            "client_id": client_id,
            "table_col": table_col,
            "column_col": column_col,
            "description_col": description_col
        }
        
        result = rag_service.process_data_dictionary(
            csv_path=temp_path,
            collection_name=collection_name,
            metadata=metadata
        )
        
        # Clean up temporary file in background
        if background_tasks:
            background_tasks.add_task(os.remove, temp_path)
        
        # Check if the processing was successful
        if not result.get("success", False):
            error_message = result.get("error", "Unknown error during data dictionary processing")
            logger.error(f"Data dictionary processing failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
        
        # Add additional debug information
        result["debug_info"] = {
            "vector_store_populated": True,  # This should be True if documents were added to vector store
            "next_step": "Documents should now be available for RAG queries",
            "test_query": f"Try querying with collection_name='{collection_name}' and client_id='{client_id}'"
        }
        
        return result
    except Exception as e:
        logger.error(f"Error processing data dictionary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure file is closed
        csv_file.file.close()

@router.get("/debug/collections/{collection_name}", response_model=Dict[str, Any])
async def debug_collection(
    collection_name: str = Path(..., description="Target vector-store collection name."),
    client_id: str = Query("default", description="Client identifier used for multi-tenant isolation."),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Debug endpoint to check what's in a vector store collection.
    
    Args:
        collection_name: Name of the collection to debug
        client_id: Client identifier for filtering
        rag_service: RAG service
        
    Returns:
        Debug information about the collection
    """
    try:
        # Try to retrieve some context with a generic query
        test_query = "table column description"
        context = rag_service.retrieve_context(
            query=test_query,
            collection_name=collection_name,
            top_k=5,
            filters={"client_id": client_id}
        )
        
        # Handle the dictionary structure returned by retrieve_context
        context_list = []
        if context and isinstance(context, dict):
            # Extract documents from the result dictionary
            context_list = context.get("documents", [])
        elif context and isinstance(context, list):
            context_list = context
        
        # Get vector store info
        debug_info = {
            "collection_name": collection_name,
            "client_id": client_id,
            "test_query": test_query,
            "context_type": str(type(context)),
            "full_context_keys": list(context.keys()) if isinstance(context, dict) else "Not a dict",
            "documents_found": len(context_list),
            "sample_documents": []
        }
        
        # Add sample documents (first 2)
        if context_list:
            for i, doc in enumerate(context_list[:2]):
                try:
                    if isinstance(doc, dict):
                        text = doc.get("text", str(doc))
                        metadata = doc.get("metadata", {})
                    else:
                        text = str(doc)
                        metadata = {}
                    
                    debug_info["sample_documents"].append({
                        "index": i,
                        "text_preview": text[:200] + "..." if len(text) > 200 else text,
                        "metadata": metadata,
                        "doc_type": str(type(doc))
                    })
                except Exception as doc_error:
                    debug_info["sample_documents"].append({
                        "index": i,
                        "error": f"Error processing document: {str(doc_error)}",
                        "doc_type": str(type(doc))
                    })
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error debugging collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve", response_model=Dict[str, Any])
async def retrieve_context(
    request: RetrieveContextRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Retrieve relevant context for a query.
    
    Args:
        request: Retrieve context request
        rag_service: RAG service
        
    Returns:
        Retrieved context
    """
    try:
        # Retrieve context
        result = rag_service.retrieve_context(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            filters=request.filters,
            conversation_history=request.conversation_history,
            client_id=request.client_id
        )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

@router.post("/enhance", response_model=Dict[str, Any])
async def enhance_prompt(
    request: EnhancePromptRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Build an enhanced prompt with retrieved context.
    
    Args:
        request: Enhance prompt request
        rag_service: RAG service
        
    Returns:
        Enhanced prompt
    """
    try:
        # Build enhanced prompt
        result = rag_service.build_enhanced_prompt(
            query=request.query,
            retrieved_context=request.retrieved_context,
            conversation_history=request.conversation_history,
            system_prompt=request.system_prompt,
            template_name=request.template_name
        )
        
        return result
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error enhancing prompt: {str(e)}")

@router.post("/query", response_model=Dict[str, Any])
@cache_llm_query(ttl_seconds=3600 * 24)  # Cache for 1 day
async def query_with_rag(
    request: QueryWithRAGRequest,
    rag_service: RAGService = Depends(get_rag_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Query with RAG.
    
    This endpoint combines retrieval, context enhancement, and LLM generation
    in a single call. The result is cached if use_cache is true.
    
    Args:
        request: Query with RAG request
        rag_service: RAG service
        cache_service: Cache service
        
    Returns:
        Query result with enhanced context and LLM response
    """
    try:
        # Check if result is cached
        if request.use_cache:
            cached_result = cache_service.get_query_result(
                query=request.query,
                client_id=request.client_id,
                model=request.model,
                context=json.dumps(request.conversation_history) if request.conversation_history else None
            )
            
            if cached_result:
                return {
                    "query": request.query,
                    "response": cached_result["response"],
                    "model": cached_result["model"],
                    "tokens_used": cached_result["tokens_used"],
                    "cached": True,
                    "cache_age": cached_result["cache_age"]
                }
        
        # Retrieve context
        retrieved_context = rag_service.retrieve_context(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            conversation_history=request.conversation_history,
            client_id=request.client_id
        )
        
        # Build enhanced prompt
        enhanced_prompt_result = rag_service.build_enhanced_prompt(
            query=request.query,
            retrieved_context=retrieved_context,
            conversation_history=request.conversation_history,
            system_prompt=request.system_prompt
        )
        
        # TODO: Call LLM with enhanced prompt
        # This would call a function in a separate LLM service module
        # For now, return the enhanced prompt for testing
        result = {
            "query": request.query,
            "enhanced_prompt": enhanced_prompt_result["enhanced_prompt"],
            "retrieved_docs_count": len(retrieved_context["documents"]),
            "retrieval_time": retrieved_context["retrieval_time_seconds"],
            "enhancement_time": enhanced_prompt_result["enhancement_time_seconds"],
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "cached": False
            # "response": llm_response would come from the LLM call
            # "tokens_used": tokens_used would come from the LLM call
        }
        
        return result
    except Exception as e:
        logger.error(f"Error querying with RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying with RAG: {str(e)}")

@router.get("/stats", response_model=Dict[str, Any])
async def get_rag_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get RAG statistics.
    
    Args:
        rag_service: RAG service
        
    Returns:
        RAG statistics
    """
    try:
        return rag_service.get_stats()
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting RAG stats: {str(e)}")

@router.get("/collections", response_model=List[str])
async def get_collections(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get available collections.
    
    Args:
        rag_service: RAG service
        
    Returns:
        List of collection names
    """
    try:
        collections = rag_service.rag_integrator.retriever.get_available_collections()
        return collections
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")
