"""
Vector store implementation for the RAG system.
This module provides classes for managing vector databases using ChromaDB or FAISS.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import os
import json
import logging
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import uuid

# Setup logging
logger = logging.getLogger(__name__)

class VectorStore:
    """Base class for vector stores."""
    
    def __init__(self, collection_name: str, embedding_dim: int = 384):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of the embeddings
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs for the added texts
        """
        raise NotImplementedError("Subclasses must implement add_texts")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter to apply
            query_embedding: Optional pre-computed query embedding
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        raise NotImplementedError("Subclasses must implement similarity_search")
    
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of IDs to delete
            filter: Optional filter to apply
        """
        raise NotImplementedError("Subclasses must implement delete")
    
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents by ID.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        raise NotImplementedError("Subclasses must implement get")

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "chroma_db",
        embedding_function: Optional[Any] = None,
        client_settings: Optional[Dict[str, Any]] = None,
        client: Optional[chromadb.Client] = None,
        embedding_dim: int = 384,
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_function: Function to generate embeddings
            client_settings: Optional settings for the ChromaDB client
            client: Optional existing ChromaDB client
            embedding_dim: Dimension of the embeddings
        """
        super().__init__(collection_name, embedding_dim)
        self.persist_directory = persist_directory
        
        # Ensure the persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        if client is not None:
            self.client = client
        else:
            settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
                is_persistent=True,
                **(client_settings or {})
            )
            self.client = chromadb.Client(settings=settings)
        
        # Set embedding function
        self.embedding_function = embedding_function
        
        # Create or get collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get an existing collection or create a new one."""
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.info(f"Creating new collection {self.collection_name}: {e}")
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add texts to the ChromaDB collection.
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs for the added texts
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadatas exists for all texts
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add documents to collection
        if embeddings:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in ChromaDB.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter to apply
            query_embedding: Optional pre-computed query embedding
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        # Perform search
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter
            )
        
        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {},
                "score": float(results["distances"][0][i]) if "distances" in results else 1.0
            }
            documents.append(doc)
        
        return documents
    
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete documents from the ChromaDB collection.
        
        Args:
            ids: Optional list of IDs to delete
            filter: Optional filter to apply
        """
        if ids:
            self.collection.delete(ids=ids)
        elif filter:
            self.collection.delete(where=filter)
    
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents by ID from ChromaDB.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        results = self.collection.get(ids=ids)
        
        documents = []
        for i in range(len(results["ids"])):
            doc = {
                "id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i] if "metadatas" in results and results["metadatas"] else {}
            }
            documents.append(doc)
        
        return documents

class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_dim: int = 384,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        content_path: Optional[str] = None,
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of the embeddings
            index_path: Path to save the FAISS index
            metadata_path: Path to save metadata
            content_path: Path to save content
        """
        super().__init__(collection_name, embedding_dim)
        
        # Import FAISS here to allow the module to be imported even if FAISS is not installed
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss python package. "
                "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )
        
        # Set paths
        self.index_path = index_path or f"faiss_{collection_name}.index"
        self.metadata_path = metadata_path or f"faiss_{collection_name}_metadata.json"
        self.content_path = content_path or f"faiss_{collection_name}_content.json"
        
        # Initialize index
        self.index = self._initialize_index()
        
        # Initialize dictionaries for metadata and content
        self.metadata = self._load_metadata()
        self.content = self._load_content()
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.metadata.get("ids", []))}
    
    def _initialize_index(self):
        """Initialize or load FAISS index."""
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading FAISS index from {self.index_path}")
                return self.faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
        
        logger.info(f"Creating new FAISS index for {self.collection_name}")
        return self.faiss.IndexFlatL2(self.embedding_dim)
    
    def _load_metadata(self):
        """Load metadata from disk."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        return {"ids": [], "metadatas": []}
    
    def _load_content(self):
        """Load content from disk."""
        if os.path.exists(self.content_path):
            try:
                with open(self.content_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading content: {e}")
        
        return {"documents": []}
    
    def _save_index(self):
        """Save the FAISS index to disk."""
        self.faiss.write_index(self.index, self.index_path)
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def _save_content(self):
        """Save content to disk."""
        with open(self.content_path, 'w') as f:
            json.dump(self.content, f)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add texts to the FAISS index.
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs for the added texts
        """
        if embeddings is None:
            raise ValueError("FAISS vector store requires pre-computed embeddings")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadatas exists for all texts
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Update metadata and content
        start_idx = len(self.metadata["ids"])
        for i, (id_, text, metadata) in enumerate(zip(ids, texts, metadatas)):
            self.id_to_index[id_] = start_idx + i
            self.metadata["ids"].append(id_)
            self.metadata["metadatas"].append(metadata)
            self.content["documents"].append(text)
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        self._save_content()
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in FAISS.
        
        Args:
            query: Query text (not used, provided for API compatibility)
            k: Number of results to return
            filter: Optional filter to apply
            query_embedding: Pre-computed query embedding (required)
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        if query_embedding is None:
            raise ValueError("FAISS vector store requires a pre-computed query embedding")
        
        # Convert query_embedding to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Perform search
        distances, indices = self.index.search(query_array, k)
        
        # Apply filter if provided
        filtered_results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata["ids"]):
                continue  # Skip invalid indices
            
            doc_id = self.metadata["ids"][idx]
            metadata = self.metadata["metadatas"][idx]
            
            # Skip if doesn't match filter
            if filter and not self._matches_filter(metadata, filter):
                continue
            
            doc = {
                "id": doc_id,
                "content": self.content["documents"][idx],
                "metadata": metadata,
                "score": float(distances[0][i])
            }
            filtered_results.append(doc)
        
        return filtered_results[:k]
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter."""
        for key, value in filter.items():
            # Handle alias between 'client_id' and 'client'
            if key not in metadata:
                if key == "client_id" and "client" in metadata:
                    if metadata["client"] != value:
                        return False
                    else:
                        continue  # alias matched
                else:
                    return False
            if metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete documents from the FAISS index.
        Note: FAISS doesn't support direct removal, so this implementation rebuilds the index.
        
        Args:
            ids: Optional list of IDs to delete
            filter: Optional filter to apply
        """
        if not ids and not filter:
            return
        
        # Identify indices to keep
        keep_indices = []
        for i, id_ in enumerate(self.metadata["ids"]):
            metadata = self.metadata["metadatas"][i]
            
            keep = True
            if ids and id_ in ids:
                keep = False
            elif filter and self._matches_filter(metadata, filter):
                keep = False
            
            if keep:
                keep_indices.append(i)
        
        # Extract data for kept indices
        kept_ids = [self.metadata["ids"][i] for i in keep_indices]
        kept_metadatas = [self.metadata["metadatas"][i] for i in keep_indices]
        kept_documents = [self.content["documents"][i] for i in keep_indices]
        
        # Extract embeddings for kept indices
        all_embeddings = []
        for i in range(self.index.ntotal):
            embedding = np.zeros(self.embedding_dim, dtype='float32')
            self.index.reconstruct(i, embedding)
            all_embeddings.append(embedding)
        
        kept_embeddings = [all_embeddings[i] for i in keep_indices]
        
        # Create new index
        self.index = self.faiss.IndexFlatL2(self.embedding_dim)
        
        # Update metadata and content
        self.metadata = {"ids": kept_ids, "metadatas": kept_metadatas}
        self.content = {"documents": kept_documents}
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.metadata["ids"])}
        
        # Add kept embeddings to new index
        if kept_embeddings:
            self.index.add(np.array(kept_embeddings))
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        self._save_content()
    
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents by ID from FAISS.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        documents = []
        for id_ in ids:
            if id_ in self.id_to_index:
                idx = self.id_to_index[id_]
                doc = {
                    "id": id_,
                    "content": self.content["documents"][idx],
                    "metadata": self.metadata["metadatas"][idx]
                }
                documents.append(doc)
        
        return documents

class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create_vector_store(
        store_type: str,
        collection_name: str,
        **kwargs
    ) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store ("chroma" or "faiss")
            collection_name: Name of the collection
            **kwargs: Additional arguments for specific vector store
            
        Returns:
            VectorStore instance
        """
        if store_type.lower() == "chroma":
            return ChromaVectorStore(collection_name, **kwargs)
        elif store_type.lower() == "faiss":
            return FAISSVectorStore(collection_name, **kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

def get_default_embedding_function():
    """
    Get the default embedding function for ChromaDB.
    
    Returns:
        ChromaDB embedding function
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
