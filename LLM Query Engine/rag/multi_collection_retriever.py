"""
Multi-collection retriever for handling multiple vector store collections.
This allows the RAG system to work with client-specific collections.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from rag.retriever import BaseRetriever
from rag.vector_store import FAISSVectorStore, ChromaVectorStore, VectorStoreFactory
from utils.embedding_utils import EmbeddingGenerator

logger = logging.getLogger(__name__)

class MultiCollectionRetriever(BaseRetriever):
    """
    Retriever that can work with multiple collections dynamically.
    """
    
    def __init__(
        self,
        vector_store_path: str = "vector_stores",
        embedding_generator: Optional[EmbeddingGenerator] = None,
        store_type: str = "faiss",
        score_threshold: float = 0.0
    ):
        """
        Initialize the multi-collection retriever.
        
        Args:
            vector_store_path: Base path for vector stores
            embedding_generator: Generator for query embeddings
            store_type: Type of vector store ("faiss" or "chroma")
            score_threshold: Minimum score threshold for results
        """
        self.vector_store_path = vector_store_path
        self.embedding_generator = embedding_generator
        self.store_type = store_type
        self.score_threshold = score_threshold
        
        # Cache for vector store instances
        self._vector_stores: Dict[str, Any] = {}
        
        # Ensure vector store directory exists
        os.makedirs(self.vector_store_path, exist_ok=True)
    
    def _get_vector_store(self, collection_name: str):
        """
        Get or create a vector store for the given collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Vector store instance
        """
        if collection_name in self._vector_stores:
            return self._vector_stores[collection_name]
        
        try:
            if self.store_type.lower() == "faiss":
                # For FAISS, use collection-specific file paths
                index_path = os.path.join(self.vector_store_path, f"{collection_name}.index")
                metadata_path = os.path.join(self.vector_store_path, f"{collection_name}_metadata.json")
                content_path = os.path.join(self.vector_store_path, f"{collection_name}_content.json")
                
                vector_store = FAISSVectorStore(
                    collection_name=collection_name,
                    embedding_dim=384,  # Default dimension for sentence transformers
                    index_path=index_path,
                    metadata_path=metadata_path,
                    content_path=content_path
                )
            else:
                # For ChromaDB, use collection-specific directory
                persist_directory = os.path.join(self.vector_store_path, collection_name)
                vector_store = ChromaVectorStore(
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                    embedding_dim=384
                )
            
            # Cache the vector store
            self._vector_stores[collection_name] = vector_store
            logger.info(f"Created vector store for collection: {collection_name}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store for collection {collection_name}: {e}")
            return None
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        if self.store_type.lower() == "faiss":
            index_path = os.path.join(self.vector_store_path, f"{collection_name}.index")
            return os.path.exists(index_path)
        else:
            persist_directory = os.path.join(self.vector_store_path, collection_name)
            return os.path.exists(persist_directory)
    
    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the specified collection.
        
        Args:
            query: Query text
            collection_name: Name of the collection to search
            k: Number of documents to retrieve
            filters: Optional filters to apply
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving from collection: {collection_name}")
        
        # Check if collection exists
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return []
        
        # Get vector store for the collection
        vector_store = self._get_vector_store(collection_name)
        if not vector_store:
            logger.error(f"Failed to get vector store for collection: {collection_name}")
            return []
        
        try:
            # Generate query embedding if generator is available
            if self.embedding_generator:
                query_embedding = self.embedding_generator.generate_embedding(query)
                
                # Retrieve documents from vector store
                docs = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters,
                    query_embedding=query_embedding
                )
            else:
                # If no embedding generator, try to let vector store handle it
                logger.warning(f"No embedding generator available for collection {collection_name}")
                docs = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )
            
            # Filter by score threshold if provided
            if self.score_threshold > 0:
                docs = [doc for doc in docs if doc.get("score", 0) >= self.score_threshold]
            
            logger.info(f"Retrieved {len(docs)} documents from collection {collection_name}")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving from collection {collection_name}: {e}")
            return []
    
    def get_available_collections(self) -> List[str]:
        """
        Get list of available collections.
        
        Returns:
            List of collection names
        """
        collections = []
        
        if not os.path.exists(self.vector_store_path):
            return collections
        
        try:
            if self.store_type.lower() == "faiss":
                # Look for .index files
                for file in os.listdir(self.vector_store_path):
                    if file.endswith(".index"):
                        collection_name = file[:-6]  # Remove .index extension
                        collections.append(collection_name)
            else:
                # Look for directories (ChromaDB collections)
                for item in os.listdir(self.vector_store_path):
                    item_path = os.path.join(self.vector_store_path, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        collections.append(item)
        
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
        
        return collections
