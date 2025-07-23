"""
RAG Integrator module for the Conversational Query Engine.
This module connects all RAG components and provides a unified interface.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from functools import wraps

from rag.document_processor import DocumentProcessor
from rag.retriever import BaseRetriever, VectorStoreRetriever, HybridRetriever, ContextualRetriever
from rag.multi_collection_retriever import MultiCollectionRetriever
from rag.context_enhancer import ContextEnhancer
from rag.embeddings_manager import EmbeddingsManager
from services.cache_service import CacheService

# Setup logging
logger = logging.getLogger(__name__)

class RAGIntegrator:
    """
    Class that integrates all RAG components and provides a unified interface
    for the application to use the RAG pipeline.
    """
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessor] = None,
        retriever: Optional[BaseRetriever] = None,
        context_enhancer: Optional[ContextEnhancer] = None,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        cache_service: Optional[CacheService] = None,
        vector_store_path: str = "vector_stores",
        default_top_k: int = 5,
        enable_hybrid_search: bool = True,
        enable_contextual_retrieval: bool = True,
        enable_caching: bool = True,
        retrieval_cache_ttl: int = 3600,  # 1 hour
        default_chunk_size: int = 500,
        default_chunk_overlap: int = 50,
        max_context_tokens: int = 4000
    ):
        """
        Initialize the RAG integrator.
        
        Args:
            document_processor: Document processor instance
            retriever: Base retriever instance
            context_enhancer: Context enhancer instance
            embeddings_manager: Embeddings manager instance
            cache_service: Cache service instance
            vector_store_path: Path to store vector stores
            default_top_k: Default number of documents to retrieve
            enable_hybrid_search: Whether to use hybrid search
            enable_contextual_retrieval: Whether to use contextual retrieval
            enable_caching: Whether to enable caching
            retrieval_cache_ttl: TTL for retrieval cache in seconds
            default_chunk_size: Default chunk size for documents
            default_chunk_overlap: Default chunk overlap for documents
            max_context_tokens: Maximum tokens for context
        """
        self.vector_store_path = vector_store_path
        self.default_top_k = default_top_k
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_contextual_retrieval = enable_contextual_retrieval
        self.enable_caching = enable_caching
        self.retrieval_cache_ttl = retrieval_cache_ttl
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.max_context_tokens = max_context_tokens
        
        # Initialize components
        self.embeddings_manager = embeddings_manager or self._init_embeddings_manager()
        self.cache_service = cache_service or CacheService(
            cache_dir=os.path.join(self.vector_store_path, "cache")
        )
        self.document_processor = document_processor or self._init_document_processor()
        self.retriever = retriever or self._init_retriever()
        self.context_enhancer = context_enhancer or self._init_context_enhancer()
        
        # Create directory for vector stores
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize statistics from persisted file or create new
        stats_path = os.path.join(self.vector_store_path, "rag_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"Loaded RAG stats from {stats_path}")
            except Exception as e:
                logger.error(f"Error loading RAG stats: {e}")
                self.stats = self._init_empty_stats()
        else:
            self.stats = self._init_empty_stats()
            self._save_stats()
            
    def _init_empty_stats(self):
        """Initialize empty statistics dictionary"""
        return {
            "total_queries": 0,
            "retrieval_time": 0,
            "documents_retrieved": 0,
            "avg_retrieval_time": 0,
            "context_enhancement_time": 0,
            "avg_context_enhancement_time": 0,
            "processing_time": 0,
            "documents_processed": 0
        }
        
    def _save_stats(self):
        """Save statistics to disk"""
        stats_path = os.path.join(self.vector_store_path, "rag_stats.json")
        try:
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"Saved RAG stats to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving RAG stats: {e}")
            
    def update_stats(self, stat_type, value):
        """
        Update statistics for tracking and monitoring.
        
        Args:
            stat_type: Type of statistic to update
            value: Value to add to the statistic
        """
        if stat_type in self.stats:
            self.stats[stat_type] += value
            
            # Recalculate averages
            if stat_type == "retrieval_time" and self.stats["total_queries"] > 0:
                self.stats["avg_retrieval_time"] = self.stats["retrieval_time"] / self.stats["total_queries"]
            elif stat_type == "context_enhancement_time" and self.stats["total_queries"] > 0:
                self.stats["avg_context_enhancement_time"] = self.stats["context_enhancement_time"] / self.stats["total_queries"]
                
            # Save stats to disk after each update
            self._save_stats()
        
        logger.info("RAG integrator initialized")
    
    def _init_embeddings_manager(self) -> EmbeddingsManager:
        """
        Initialize the embeddings manager.
        
        Returns:
            Configured embeddings manager
        """
        # Try to use Sentence Transformers as default
        # Fall back to OpenAI if not available
        try:
            # Use the correct parameters according to EmbeddingsManager __init__ signature
            # (no provider parameter)
            return EmbeddingsManager(
                embedding_generator=None,  # Use default
                cache_dir="cache/embeddings",
                use_disk_cache=True,
                use_memory_cache=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings manager: {e}")
            # Create a basic manager with minimal parameters
            return EmbeddingsManager(cache_dir="cache/embeddings")
    
    def _init_document_processor(self) -> DocumentProcessor:
        """
        Initialize the document processor.
        
        Returns:
            Configured document processor
        """
        # Extract the embedding_generator from the embeddings_manager if available
        embedding_generator = getattr(self.embeddings_manager, 'embedding_generator', None)
        
        return DocumentProcessor(
            embedding_generator=embedding_generator,
            chunk_size=self.default_chunk_size,
            chunk_overlap=self.default_chunk_overlap,
            use_semantic_chunking=True
        )
    
    def _init_retriever(self) -> BaseRetriever:
        """
        Initialize the retriever based on configuration.
        Uses MultiCollectionRetriever for dynamic collection handling.
        
        Returns:
            Configured retriever
        """
        # Get vector store type from environment
        vector_store_type = os.environ.get("VECTOR_STORE_TYPE", "faiss")
        
        # Extract the embedding_generator from embeddings_manager if available
        embedding_generator = getattr(self.embeddings_manager, 'embedding_generator', None)
        
        # Create multi-collection retriever
        multi_retriever = MultiCollectionRetriever(
            vector_store_path=self.vector_store_path,
            embedding_generator=embedding_generator,
            store_type=vector_store_type,
            score_threshold=0.0
        )
        
        # If contextual retrieval is enabled, wrap with contextual retriever
        if self.enable_contextual_retrieval:
            return ContextualRetriever(
                base_retriever=multi_retriever,
                context_window_size=3  # Default context window size
            )
        
        return multi_retriever
    
    def _init_context_enhancer(self) -> ContextEnhancer:
        """
        Initialize the context enhancer.
        
        Returns:
            Configured context enhancer
        """
        # First we need a retriever to pass to the ContextEnhancer
        # We'll use the retriever we already initialized
        if not hasattr(self, 'retriever') or self.retriever is None:
            self.retriever = self._init_retriever()
            
        return ContextEnhancer(
            retriever=self.retriever,
            max_context_length=self.max_context_tokens,
            relevance_threshold=0.5,  # Default threshold
            include_metadata=True
        )
    
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
        start_time = time.time()
        
        # Use default chunk parameters if not provided
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        # Process documents
        result = self.document_processor.process_documents(
            documents=documents,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata
        )
        
        # Add processed documents to vector store if processing was successful
        if result.get('success', False) and result.get('processed_documents'):
            try:
                processed_docs = result['processed_documents']
                
                # Get the vector store for this collection
                if hasattr(self.retriever, '_get_vector_store'):
                    vector_store = self.retriever._get_vector_store(collection_name)
                    if not vector_store:
                        raise ValueError(f"Could not create vector store for collection: {collection_name}")
                else:
                    raise ValueError("Retriever does not support multi-collection operations")
                
                # Prepare data for vector store
                texts = [doc["text"] for doc in processed_docs]
                metadatas = [doc["metadata"] for doc in processed_docs]
                embeddings = [doc["embedding"] for doc in processed_docs]
                
                # Add to vector store
                if hasattr(vector_store, "add_texts"):
                    logger.info(f"Adding {len(texts)} chunks to vector store collection '{collection_name}'")
                    vector_store.add_texts(
                        texts=texts,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                    logger.info(f"Successfully added {len(texts)} chunks to vector store collection '{collection_name}'")
                else:
                    logger.error(f"Vector store of type {type(vector_store).__name__} doesn't support adding texts")
                    result['vector_store_error'] = "Vector store doesn't support adding texts"
                    
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                result['vector_store_error'] = str(e)
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats["processing_time"] += processing_time
        self.stats["documents_processed"] += len(documents)
        
        return {
            **result,
            "processing_time_seconds": processing_time
        }
    
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
        start_time = time.time()
        
        # Process schema
        result = self.document_processor.process_schema_definition(
            schema_definition=schema_definition,
            collection_name=collection_name,
            include_examples=include_examples,
            metadata=metadata
        )
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats["processing_time"] += processing_time
        self.stats["documents_processed"] += 1
        
        return {
            **result,
            "processing_time_seconds": processing_time
        }
    
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
            Processing results with stats and success status
        """
        """
        Process CSV data dictionary for RAG.
        
        Args:
            csv_path: Path to CSV data dictionary
            collection_name: Name of the dictionary collection
            metadata: Optional metadata to add
            
        Returns:
            Processing results
        """
        start_time = time.time()
        documents = []
        client_name = "default"
        
        try:
            # Process data dictionary
            client_name = metadata.get("client_id", "default") if metadata else "default"
            logger.info(f"Processing data dictionary for client: {client_name}, collection: {collection_name}")
            
            # Extract column names from metadata or use defaults
            table_col = metadata.get("table_col", "TABLE_NAME") if metadata else "TABLE_NAME"
            column_col = metadata.get("column_col", "COLUMN_NAME") if metadata else "COLUMN_NAME"
            description_col = metadata.get("description_col", "DESCRIPTION") if metadata else "DESCRIPTION"
            
            logger.info(f"Using column mappings: table={table_col}, column={column_col}, description={description_col}")
            
            # Process the CSV file
            documents = self.document_processor.process_csv_data_dictionary(
                file_path=csv_path,
                client_name=client_name,
                table_col=table_col,
                column_col=column_col,
                description_col=description_col
            )
            
            logger.info(f"Processed {len(documents)} documents from data dictionary")
        
            # Store documents in vector store
            if documents:
                try:
                    # Extract only the necessary fields for vector store
                    vector_docs = [{
                        "text": doc["text"],
                        "embedding": doc["embedding"],
                        "metadata": doc["metadata"]
                    } for doc in documents]
                    
                    logger.info(f"Preparing to add {len(vector_docs)} documents to vector store collection '{collection_name}'")
                    
                    # Log sample document content for debugging
                    if vector_docs:
                        sample_doc = vector_docs[0]
                        logger.info(f"Sample document text preview: {sample_doc['text'][:200]}...")
                        logger.info(f"Sample document metadata: {sample_doc['metadata']}")
                    else:
                        logger.warning("No vector documents to add to vector store!")
                    
                    # Add to vector store - handle different vector store implementations
                    # Extract the data we need from vector_docs
                    texts = [doc["text"] for doc in vector_docs]
                    metadatas = [doc["metadata"] for doc in vector_docs]
                    embeddings = [doc["embedding"] for doc in vector_docs]
                    
                    # Get the vector store for this collection from the multi-collection retriever
                    if hasattr(self.retriever, '_get_vector_store'):
                        # Multi-collection retriever
                        vector_store = self.retriever._get_vector_store(collection_name)
                        if not vector_store:
                            raise ValueError(f"Could not create vector store for collection: {collection_name}")
                    else:
                        # Fallback for other retriever types
                        raise ValueError("Retriever does not support multi-collection operations")
                    
                    # Check which method the vector store supports and call accordingly
                    if hasattr(vector_store, "add_documents"):
                        # For vector stores with add_documents method
                        logger.info("Using add_documents method for vector store")
                        vector_store.add_documents(
                            collection_name=collection_name,
                            documents=vector_docs
                        )
                    elif hasattr(vector_store, "add_texts"):
                        # For FAISS and similar vector stores
                        logger.info("Using add_texts method for vector store (FAISS)")
                        vector_store.add_texts(
                            texts=texts,
                            metadatas=metadatas,
                            embeddings=embeddings
                        )
                    else:
                        logger.error(f"Vector store of type {type(vector_store).__name__} doesn't support adding documents")
                        raise ValueError("Vector store doesn't support adding documents or texts")
                    
                    logger.info(f"Successfully added documents to vector store collection '{collection_name}'")
                except Exception as e:
                    logger.error(f"Error adding documents to vector store: {str(e)}")
                    raise
        
            # Update stats
            processing_time = time.time() - start_time
            self.stats["processing_time"] += processing_time
            self.stats["documents_processed"] += len(documents) if documents else 0
            
            return {
                "success": True,
                "collection_name": collection_name,
                "documents_processed": len(documents) if documents else 0,
                "processing_time_seconds": processing_time,
                "client_id": client_name
            }
            
        except Exception as e:
            # Log the error and return failure status
            processing_time = time.time() - start_time
            logger.error(f"Error in process_data_dictionary: {str(e)}")
            
            return {
                "success": False,
                "collection_name": collection_name,
                "error": str(e),
                "processing_time_seconds": processing_time,
                "client_id": client_name
            }
    
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
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Check if we have cached results
        cache_key = f"retrieval:{client_id}:{query}:{collection_name}:{top_k}"
        if self.enable_caching and hasattr(self.cache_service, 'cache_integrator'):
            # Access the cache_integrator directly
            cached_result = self.cache_service.cache_integrator.get_generic_cached(
                key=cache_key,
                namespace="retrieval_cache"
            )
            
            if cached_result and cached_result[0]:
                return cached_result[0]
        
        # If not cached, perform retrieval
        top_k = top_k or self.default_top_k
        
        # Create appropriate retriever for this query
        retriever = self.retriever
        
        # If conversation history is provided and contextual retrieval is enabled,
        # ensure we're using a ContextualRetriever
        if conversation_history and self.enable_contextual_retrieval:
            if not isinstance(retriever, ContextualRetriever):
                retriever = ContextualRetriever(
                    base_retriever=retriever,
                    embeddings_manager=self.embeddings_manager
                )
        
        # Retrieve documents
        retrieved_docs = retriever.retrieve(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            filters=filters,
            conversation_history=conversation_history
        )
        
        # Enhance context
        enhanced_context, _ = self.context_enhancer.enhance_query_context(
            query=query,
            k=len(retrieved_docs),
            structured=True
        )
        
        # Build result
            
        result = {
            "query": query,
            "documents": retrieved_docs,
            "enhanced_context": enhanced_context,
            "retrieval_time_seconds": time.time() - start_time,
            "num_documents": len(retrieved_docs)
        }
        
        # Update stats
        retrieval_time = time.time() - start_time
        self.update_stats("retrieval_time", retrieval_time)
        self.update_stats("documents_retrieved", len(retrieved_docs))
        
        # Cache result
        if self.enable_caching and hasattr(self.cache_service, 'cache_integrator'):
            self.cache_service.cache_integrator.cache_generic(
                key=cache_key,
                value=result,
                namespace="retrieval_cache",
                ttl_seconds=self.retrieval_cache_ttl,
                metadata={
                    "query": query,
                    "collection_name": collection_name,
                    "top_k": top_k
                }
            )
        
        return result
    
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
        start_time = time.time()
        
        # Build enhanced prompt
        enhanced_prompt = self.context_enhancer.build_enhanced_prompt(
            query=query,
            enhanced_context=retrieved_context["enhanced_context"],
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            template_name=template_name
        )
        
        # Update stats
        enhancement_time = time.time() - start_time
        self.stats["context_enhancement_time"] += enhancement_time
        
        if self.stats["total_queries"] > 0:
            self.stats["avg_context_enhancement_time"] = (
                self.stats["context_enhancement_time"] / self.stats["total_queries"]
            )
        
        return {
            "enhanced_prompt": enhanced_prompt,
            "enhancement_time_seconds": enhancement_time
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Get base stats
        stats = self.stats.copy()
        
        # Add component-specific stats if available
        if hasattr(self.embeddings_manager, "get_stats"):
            stats["embeddings_manager"] = self.embeddings_manager.get_stats()
        
        if hasattr(self.retriever, "get_stats"):
            stats["retriever"] = self.retriever.get_stats()
        
        return stats
