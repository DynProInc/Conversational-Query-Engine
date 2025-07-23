"""
RAG Service module for the Conversational Query Engine.
This module provides RAG functionality to the FastAPI application.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from functools import wraps

from rag.rag_integrator import RAGIntegrator
from rag.embeddings_manager import EmbeddingsManager
from services.cache_service import CacheService

# Setup logging
logger = logging.getLogger(__name__)

class RAGService:
    """
    Service class that provides RAG functionality to the FastAPI application.
    Initializes and manages the RAG integrator.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one RAG service instance."""
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        cache_service: Optional[CacheService] = None,
        vector_store_path: str = "vector_stores",
        default_top_k: int = 5,
        enable_hybrid_search: bool = True,
        enable_contextual_retrieval: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize the RAG service.
        
        Args:
            embeddings_manager: Embeddings manager instance
            cache_service: Cache service instance
            vector_store_path: Path to store vector data
            default_top_k: Default number of documents to retrieve
            enable_hybrid_search: Whether to use hybrid search
            enable_contextual_retrieval: Whether to use contextual retrieval
            enable_caching: Whether to enable caching
        """
        # Only initialize once (singleton)
        if self._initialized:
            return
        
        self.vector_store_path = vector_store_path
        self.default_top_k = default_top_k
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_contextual_retrieval = enable_contextual_retrieval
        self.enable_caching = enable_caching
        
        # Read configuration from environment if available
        self._configure_from_env()
        
        # Create directory for vector stores if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize components
        self.embeddings_manager = embeddings_manager or self._init_embeddings_manager()
        self.cache_service = cache_service or CacheService(
            cache_dir=os.path.join(self.vector_store_path, "cache")
        )
        
        # Initialize RAG integrator
        self.rag_integrator = RAGIntegrator(
            embeddings_manager=self.embeddings_manager,
            cache_service=self.cache_service,
            vector_store_path=self.vector_store_path,
            default_top_k=self.default_top_k,
            enable_hybrid_search=self.enable_hybrid_search,
            enable_contextual_retrieval=self.enable_contextual_retrieval,
            enable_caching=self.enable_caching
        )
        
        self._initialized = True
        logger.info("RAG service initialized")
    
    def _configure_from_env(self):
        """Configure RAG service from environment variables."""
        self.vector_store_path = os.environ.get("VECTOR_STORE_PATH", self.vector_store_path)
        
        # Feature flags
        self.enable_hybrid_search = os.environ.get("ENABLE_HYBRID_SEARCH", "").lower() in ["true", "1", "yes"]
        self.enable_contextual_retrieval = os.environ.get("ENABLE_CONTEXTUAL_RETRIEVAL", "").lower() in ["true", "1", "yes"]
        self.enable_caching = os.environ.get("ENABLE_RAG_CACHING", "").lower() in ["true", "1", "yes"]
        
        # Retrieval parameters
        if "DEFAULT_TOP_K" in os.environ:
            self.default_top_k = int(os.environ.get("DEFAULT_TOP_K"))
    
    def _init_embeddings_manager(self) -> EmbeddingsManager:
        """
        Initialize the embeddings manager.
        
        Returns:
            Configured embeddings manager
        """
        cache_dir = os.path.join(self.vector_store_path, "embeddings_cache")
        cache_duration = int(os.environ.get("EMBEDDINGS_CACHE_DURATION", 86400 * 30))  # 30 days by default
        
        try:
            return EmbeddingsManager(
                cache_dir=cache_dir,
                cache_duration_seconds=cache_duration,
                use_disk_cache=True,
                use_memory_cache=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings manager: {e}")
            raise e
    
    def process_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process documents for RAG.
        
        Args:
            documents: List of documents to process
            collection_name: Name of the document collection
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            metadata: Optional metadata to add to documents
            
        Returns:
            Processing results
        """
        return self.rag_integrator.process_documents(
            documents=documents,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata
        )
    
    def process_schema(
        self,
        schema_definition: Dict[str, Any],
        collection_name: str,
        include_examples: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process database schema definition for RAG.
        
        Args:
            schema_definition: Database schema definition
            collection_name: Name of the schema collection
            include_examples: Whether to include example data
            metadata: Optional metadata to add
            
        Returns:
            Processing results
        """
        return self.rag_integrator.process_schema(
            schema_definition=schema_definition,
            collection_name=collection_name,
            include_examples=include_examples,
            metadata=metadata
        )
    
    def process_data_dictionary(
        self,
        csv_path: str,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process CSV data dictionary for RAG.
        
        Args:
            csv_path: Path to CSV data dictionary
            collection_name: Name of the dictionary collection
            metadata: Optional metadata to add
            
        Returns:
            Processing results
        """
        return self.rag_integrator.process_data_dictionary(
            csv_path=csv_path,
            collection_name=collection_name,
            metadata=metadata
        )
    
    def retrieve_context(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            collection_name: Optional collection name filter
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            conversation_history: Optional conversation history
            client_id: Client identifier
            
        Returns:
            Retrieved context
        """
        return self.rag_integrator.retrieve_context(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            filters=filters,
            conversation_history=conversation_history,
            client_id=client_id
        )
    
    def build_enhanced_prompt(
        self,
        query: str,
        retrieved_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        template_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Build an enhanced prompt with retrieved context.
        
        Args:
            query: User query
            retrieved_context: Retrieved context from retrieve_context method
            conversation_history: Optional conversation history
            system_prompt: Optional system prompt
            template_name: Prompt template name
            
        Returns:
            Enhanced prompt
        """
        return self.rag_integrator.build_enhanced_prompt(
            query=query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            template_name=template_name
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG service statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.rag_integrator.get_stats()
    
    def generate_query_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text
            
        Returns:
            Query embedding
        """
        return self.embeddings_manager.generate_embedding(text)

# Create singleton instance for import
rag_service = RAGService()

def get_rag_service() -> RAGService:
    """
    Get the singleton instance of the RAG service.
    
    Returns:
        RAGService: The singleton RAG service instance
    """
    return rag_service
