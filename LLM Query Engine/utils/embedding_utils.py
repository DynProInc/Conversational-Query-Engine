"""
Embedding utilities for Conversational Query Engine.
This module provides functions for generating and manipulating embeddings
for text data using different models.
"""
from typing import List, Dict, Any, Union, Optional
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Constants
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"  # Small but effective model for local embedding
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_EMBEDDING_DIMENSION = 384  # Default for all-MiniLM-L6-v2

class EmbeddingGenerator:
    """Class for generating embeddings using different models."""
    
    def __init__(
        self, 
        model_name: str = DEFAULT_LOCAL_MODEL,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use for local embeddings
            use_openai: Whether to use OpenAI's embedding API
            openai_api_key: OpenAI API key (if use_openai is True)
            openai_model: OpenAI embedding model name (if use_openai is True)
            embedding_dimension: Dimension of the embeddings
        """
        self.use_openai = use_openai
        self.embedding_dimension = embedding_dimension
        
        if use_openai:
            if not openai_api_key and "OPENAI_API_KEY" in os.environ:
                openai_api_key = os.environ["OPENAI_API_KEY"]
            if not openai_api_key:
                raise ValueError("OpenAI API key must be provided when use_openai=True")
            openai.api_key = openai_api_key
            self.openai_model = openai_model
            # Adjust embedding dimension based on OpenAI model
            if openai_model == "text-embedding-ada-002":
                self.embedding_dimension = 1536
        else:
            # Load local model
            self.model = SentenceTransformer(model_name)
            # Update embedding dimension based on the loaded model
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension
        
        if self.use_openai:
            # Use OpenAI API for embeddings
            response = openai.embeddings.create(
                input=text.strip(),
                model=self.openai_model
            )
            return response.data[0].embedding
        else:
            # Use local sentence-transformers model
            return self.model.encode(text.strip()).tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            return [[0.0] * self.embedding_dimension]
        
        if self.use_openai:
            # Use OpenAI API for embeddings (in batches to avoid rate limits)
            BATCH_SIZE = 100  # OpenAI recommends batches of ~100
            all_embeddings = []
            
            for i in range(0, len(valid_texts), BATCH_SIZE):
                batch_texts = valid_texts[i:i+BATCH_SIZE]
                response = openai.embeddings.create(
                    input=batch_texts,
                    model=self.openai_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        else:
            # Use local sentence-transformers model
            return self.model.encode(valid_texts).tolist()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (between -1 and 1)
    """
    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)
    
    norm1 = np.linalg.norm(vec1_array)
    norm2 = np.linalg.norm(vec2_array)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1_array, vec2_array) / (norm1 * norm2))

def batch_cosine_similarity(query_vec: List[float], vectors: List[List[float]]) -> List[float]:
    """
    Calculate cosine similarities between a query vector and multiple vectors.
    
    Args:
        query_vec: Query vector
        vectors: List of vectors to compare against
        
    Returns:
        List of similarity scores
    """
    query_array = np.array(query_vec)
    query_norm = np.linalg.norm(query_array)
    
    if query_norm == 0:
        return [0.0] * len(vectors)
    
    vectors_array = np.array(vectors)
    vectors_norm = np.linalg.norm(vectors_array, axis=1)
    
    # Avoid division by zero
    vectors_norm[vectors_norm == 0] = 1.0
    
    dot_products = np.dot(vectors_array, query_array)
    similarities = dot_products / (vectors_norm * query_norm)
    
    return similarities.tolist()

def get_top_k_similar_indices(query_vec: List[float], vectors: List[List[float]], k: int = 5) -> List[int]:
    """
    Get indices of top k most similar vectors to the query vector.
    
    Args:
        query_vec: Query vector
        vectors: List of vectors to compare against
        k: Number of top similar vectors to return
        
    Returns:
        List of indices for the top k most similar vectors
    """
    similarities = batch_cosine_similarity(query_vec, vectors)
    
    # Get indices sorted by similarity in descending order
    indices = np.argsort(similarities)[::-1]
    
    # Return top k indices
    return indices[:k].tolist()
