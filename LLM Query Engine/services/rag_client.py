"""
RAG Client - Simple integration module for using RAG in the unified API endpoint.
This provides a clean interface to call the RAG functionality from other parts of the application.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from rag.rag_integrator import RAGIntegrator
from services.cache_service import CacheService

logger = logging.getLogger(__name__)

# Create singleton instance that will be accessed from other modules
_rag_integrator = None

def get_rag_service() -> RAGIntegrator:
    """
    Get the RAG integrator instance (singleton pattern)
    
    Returns:
        RAGIntegrator instance
    """
    global _rag_integrator
    
    try:
        if _rag_integrator is None:
            logger.info("Initializing RAG service...")
            vector_store_path = os.environ.get("VECTOR_STORE_PATH", "vector_stores")
            # Initialize without custom embedding parameters - use defaults
            _rag_integrator = RAGIntegrator(vector_store_path=vector_store_path)
            logger.info("RAG service initialized")
        
        return _rag_integrator
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        # Create an emergency stub if initialization fails
        from unittest.mock import MagicMock
        mock = MagicMock()
        # Make sure the mock returns empty results when methods are called
        mock.retrieve.return_value = {"documents": [], "query": "", "error": str(e)}
        return mock

def retrieve_context(query: str, collection_name: str, client_id: Optional[str] = None, 
                top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Retrieve relevant context from the vector store based on the query.
    
    Args:
        query: The query to search for
        collection_name: Name of the collection to search in
        client_id: Client ID for filtering (optional)
        top_k: Number of documents to retrieve
        filters: Additional filters for document retrieval
        
    Returns:
        Dictionary containing retrieved documents and metadata
    """
    try:
        # Get rag integrator
        rag_integrator = get_rag_service()
        
        # Add client_id to filters if provided
        if client_id:
            if not filters:
                filters = {}
            filters["client_id"] = client_id
        
        # Log the retrieval attempt
        logger.info(f"Retrieving context for '{query}' from collection '{collection_name}'")
        logger.info(f"Using filters: {filters}")
        
        # Use the rag integrator to retrieve documents
        result = rag_integrator.retrieve_context(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            filters=filters
        )
        
        if result and result.get("documents"):
            logger.info(f"Retrieved {len(result['documents'])} documents from collection '{collection_name}'")
        else:
            logger.warning(f"No documents retrieved from collection '{collection_name}'")
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        # Return empty result on error
        return {"documents": [], "query": query, "error": str(e)}

def _prune_documents_for_query(documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Heuristically drop column_definition chunks whose column name is unrelated to the query.
    Returns a pruned document list; falls back to original list if everything would be removed."""
    try:
        if not documents:
            return documents
        query_tokens = set(query.lower().replace("_", " ").split())
        pruned: List[Dict[str, Any]] = []
        for doc in documents:
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            ctype = meta.get("content_type")
            if ctype == "column_definition":
                col_name = (meta.get("column_name") or "").lower()
                normalized = col_name.replace("_", " ")
                if any(t in normalized for t in query_tokens):
                    pruned.append(doc)
            else:
                # Keep table definitions or other contexts untouched
                pruned.append(doc)
        # Ensure we don't return empty result (would harm accuracy)
        return pruned if pruned else documents
    except Exception:
        # Safety: on any error fallback to original docs
        return documents


def enhance_prompt(query: str, retrieved_context: Dict[str, Any], 
                system_prompt: Optional[str] = None,
                prune_columns: bool = False) -> Dict[str, Any]:
    """
    Enhance a prompt with retrieved context.
    
    Args:
        query: The original query
        retrieved_context: Context retrieved from the vector store
        system_prompt: Optional system prompt to use for enhancement
        
    Returns:
        Dictionary containing the enhanced prompt
    """
    try:
        # Get rag integrator
        rag_integrator = get_rag_service()
        
        # Log the enhancement attempt
        logger.info(f"Enhancing prompt for query: '{query}'")
        
        # Use the context enhancer to create an enhanced prompt
        documents = retrieved_context.get("documents", [])
        # Optional column pruning
        if prune_columns:
            logger.info("Column pruning enabled – pruning irrelevant columns from context")
            documents = _prune_documents_for_query(documents, query)
            logger.info(f"Context size after pruning: {len(documents)} documents")
        if not documents:
            logger.warning("No documents provided for prompt enhancement")
            return {"enhanced_prompt": query, "original_query": query}
        
        # Log document count
        logger.info(f"Using {len(documents)} documents for prompt enhancement")
        
        # Build context text using ContextEnhancer utilities
        try:
            context_text = rag_integrator.context_enhancer.format_structured_context(documents)
        except Exception:
            # Fallback to simple formatter
            context_text = rag_integrator.context_enhancer.format_retrieved_documents(documents)
        
        # Combine system prompt (if any) and query with context
        if system_prompt:
            enhanced_prompt = f"{system_prompt}\n\n{context_text}\n\n{query}"
        else:
            enhanced_prompt = f"{context_text}\n\n{query}" if context_text else query
        
        return {
            "enhanced_prompt": enhanced_prompt,
            "original_query": query,
            "num_documents_used": len(documents)
        }
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        # Return original query on error
        return {"enhanced_prompt": query, "original_query": query, "error": str(e)}
