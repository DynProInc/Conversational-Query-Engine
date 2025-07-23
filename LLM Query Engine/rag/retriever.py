"""
Retrieval module for the RAG system.
This module provides classes for retrieving relevant content from vector stores.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import re
from collections import defaultdict

from utils.embedding_utils import EmbeddingGenerator, get_top_k_similar_indices, batch_cosine_similarity
from utils.text_processing import preprocess_user_query, extract_keywords
from rag.vector_store import VectorStore

# Setup logging
logger = logging.getLogger(__name__)

class BaseRetriever:
    """Base class for retrievers."""
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        raise NotImplementedError("Subclasses must implement retrieve")

class VectorStoreRetriever(BaseRetriever):
    """Class for retrieving documents from a vector store."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        score_threshold: float = 0.0
    ):
        """
        Initialize the vector store retriever.
        
        Args:
            vector_store: Vector store to retrieve from
            embedding_generator: Generator for query embeddings
            score_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the vector store.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            filter: Optional filter to apply
            
        Returns:
            List of retrieved documents
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Retrieve documents from vector store
        docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
            query_embedding=query_embedding
        )
        
        # Filter by score threshold if provided
        if self.score_threshold > 0:
            docs = [doc for doc in docs if doc.get("score", 0) >= self.score_threshold]
        
        return docs

class KeywordRetriever(BaseRetriever):
    """Class for retrieving documents based on keyword matching."""
    
    def __init__(self, documents: List[Dict[str, str]]):
        """
        Initialize the keyword retriever.
        
        Args:
            documents: List of documents with "text" and other fields
        """
        self.documents = documents
        # Build keyword index
        self.keyword_index = self._build_keyword_index()
    
    def _build_keyword_index(self) -> Dict[str, List[int]]:
        """
        Build a keyword index for the documents.
        
        Returns:
            Dictionary mapping keywords to document indices
        """
        keyword_index = defaultdict(list)
        
        for i, doc in enumerate(self.documents):
            text = doc.get("text", "")
            # Extract keywords from text
            keywords = extract_keywords(text)
            
            # Add document index to each keyword
            for keyword in keywords:
                keyword_index[keyword].append(i)
        
        return keyword_index
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on keyword matching.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Extract keywords from query
        _, query_keywords = preprocess_user_query(query)
        
        # Count keyword matches for each document
        doc_scores = defaultdict(int)
        
        for keyword in query_keywords:
            for doc_idx in self.keyword_index.get(keyword, []):
                doc_scores[doc_idx] += 1
        
        # Sort documents by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k documents
        top_docs = []
        for doc_idx, score in sorted_docs[:k]:
            doc = self.documents[doc_idx].copy()
            doc["score"] = score / max(len(query_keywords), 1)  # Normalize score
            top_docs.append(doc)
        
        return top_docs

class HybridRetriever(BaseRetriever):
    """Class combining vector search and keyword search."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        score_threshold: float = 0.0
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search
            embedding_generator: Generator for query embeddings
            documents: List of documents for keyword search
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            score_threshold: Minimum combined score threshold
        """
        self.vector_retriever = VectorStoreRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        if documents:
            self.keyword_retriever = KeywordRetriever(documents=documents)
        else:
            self.keyword_retriever = None
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using both vector and keyword search.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            filter: Optional filter to apply
            
        Returns:
            List of retrieved documents
        """
        # Get more results than needed to allow for reranking
        n_results = k * 2
        
        # Get vector search results
        vector_docs = self.vector_retriever.retrieve(
            query=query,
            k=n_results,
            filter=filter
        )
        
        # If no keyword retriever, return vector results
        if not self.keyword_retriever:
            return vector_docs[:k]
        
        # Get keyword search results
        keyword_docs = self.keyword_retriever.retrieve(
            query=query,
            k=n_results
        )
        
        # Combine results
        doc_scores = {}
        
        # Add vector search results
        for doc in vector_docs:
            doc_id = doc["id"]
            vector_score = doc.get("score", 0)
            doc_scores[doc_id] = {
                "doc": doc,
                "combined_score": vector_score * self.vector_weight,
                "vector_score": vector_score,
                "keyword_score": 0
            }
        
        # Add keyword search results
        for doc in keyword_docs:
            doc_id = doc["id"]
            keyword_score = doc.get("score", 0)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]["keyword_score"] = keyword_score
                doc_scores[doc_id]["combined_score"] += keyword_score * self.keyword_weight
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "combined_score": keyword_score * self.keyword_weight,
                    "vector_score": 0,
                    "keyword_score": keyword_score
                }
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Filter by threshold
        if self.score_threshold > 0:
            sorted_docs = [d for d in sorted_docs if d["combined_score"] >= self.score_threshold]
        
        # Format results
        results = []
        for item in sorted_docs[:k]:
            doc = item["doc"]
            doc["score"] = item["combined_score"]
            doc["vector_score"] = item["vector_score"]
            doc["keyword_score"] = item["keyword_score"]
            results.append(doc)
        
        return results

class MultiQueryRetriever(BaseRetriever):
    """Class that generates multiple query variations and combines results."""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        query_generator: Any,  # Should be an LLM-based query generator
        num_queries: int = 3,
        deduplication_threshold: float = 0.95
    ):
        """
        Initialize the multi-query retriever.
        
        Args:
            base_retriever: Base retriever to use
            query_generator: Generator for query variations
            num_queries: Number of query variations to generate
            deduplication_threshold: Threshold for deduplicating results
        """
        self.base_retriever = base_retriever
        self.query_generator = query_generator
        self.num_queries = num_queries
        self.deduplication_threshold = deduplication_threshold
    
    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of the query.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        # Use query generator to create variations
        variations = self.query_generator.generate_variations(
            query, self.num_queries
        )
        
        # Always include original query
        if query not in variations:
            variations.insert(0, query)
        
        return variations
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multiple query variations.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Generate query variations
        query_variations = self.generate_query_variations(query)
        
        # Retrieve documents for each variation
        all_docs = []
        for query_var in query_variations:
            docs = self.base_retriever.retrieve(query=query_var, k=k, **kwargs)
            all_docs.extend(docs)
        
        # Deduplicate results by document ID
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc["id"]
            if doc_id not in unique_docs or doc["score"] > unique_docs[doc_id]["score"]:
                unique_docs[doc_id] = doc
        
        # Sort by score
        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_docs[:k]

class ContextualRetriever(BaseRetriever):
    """Class that incorporates conversation context in retrieval."""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        context_window_size: int = 3,
        query_weight: float = 0.8,
        context_weight: float = 0.2
    ):
        """
        Initialize the contextual retriever.
        
        Args:
            base_retriever: Base retriever to use
            context_window_size: Number of previous exchanges to consider
            query_weight: Weight for current query results
            context_weight: Weight for context query results
        """
        self.base_retriever = base_retriever
        self.context_window_size = context_window_size
        self.query_weight = query_weight
        self.context_weight = context_weight
        self.conversation_history = []
    
    def add_to_history(self, query: str, response: str):
        """
        Add a query-response pair to conversation history.
        
        Args:
            query: User query
            response: System response
        """
        self.conversation_history.append((query, response))
        # Trim history to window size
        if len(self.conversation_history) > self.context_window_size:
            self.conversation_history = self.conversation_history[-self.context_window_size:]
    
    def get_context_query(self) -> str:
        """
        Generate a context query from conversation history.
        
        Returns:
            Context query text
        """
        if not self.conversation_history:
            return ""
        
        context = ""
        for query, response in self.conversation_history:
            context += f"User: {query}\nSystem: {response}\n\n"
        
        return context.strip()
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents considering conversation context.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Retrieve documents for current query
        query_docs = self.base_retriever.retrieve(query=query, k=k, **kwargs)
        
        # If no history, return query results
        if not self.conversation_history:
            return query_docs
        
        # Generate context query
        context_query = self.get_context_query()
        
        # Retrieve documents for context query
        context_docs = self.base_retriever.retrieve(query=context_query, k=k, **kwargs)
        
        # Combine results with weighting
        doc_scores = {}
        
        # Add query results
        for doc in query_docs:
            doc_id = doc["id"]
            query_score = doc.get("score", 0)
            doc_scores[doc_id] = {
                "doc": doc,
                "combined_score": query_score * self.query_weight,
                "query_score": query_score,
                "context_score": 0
            }
        
        # Add context results
        for doc in context_docs:
            doc_id = doc["id"]
            context_score = doc.get("score", 0)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]["context_score"] = context_score
                doc_scores[doc_id]["combined_score"] += context_score * self.context_weight
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "combined_score": context_score * self.context_weight,
                    "query_score": 0,
                    "context_score": context_score
                }
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Format results
        results = []
        for item in sorted_docs[:k]:
            doc = item["doc"]
            doc["score"] = item["combined_score"]
            doc["query_score"] = item["query_score"]
            doc["context_score"] = item["context_score"]
            results.append(doc)
        
        return results
