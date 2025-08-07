"""
Reranker Module for RAG System

This module provides a reranking service for RAG (Retrieval-Augmented Generation) results.
It enhances search relevance by using transformer-based reranking models.

Features:
- Supports multiple reranker models with fallback mechanism
- Combines original vector search scores with reranker scores
- Handles errors gracefully with automatic fallback
- Thread-safe singleton pattern for efficient resource usage

Dependencies:
- transformers
- torch
- numpy
"""

import os
import sys
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logger = logging.getLogger("RAGReranker")

class RAGReranker:
    """
    A reranker for RAG results with fallback models.
    Tries models in order of preference and falls back to next if one fails.
    """
    
    # List of models to try in order of preference
    RERANKER_MODELS = [
        "BAAI/bge-reranker-large",  # First choice - larger model, more accurate
        "BAAI/bge-reranker-base",   # Second choice - good balance of speed and accuracy
        "Qwen/Qwen3-Reranker-0.6B"  # Fallback - our original model
    ]
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker = None
        self.tokenizer = None
        self.initialized = False
        self.current_model = None
        
        # Try to initialize with the first available model
        self._initialize_with_fallback()
    
    def _initialize_with_fallback(self):
        """Try to initialize with each model until one succeeds"""
        for model_name in self.RERANKER_MODELS:
            try:
                logger.info(f"Attempting to initialize reranker with model: {model_name}")
                self._initialize_model(model_name)
                self.current_model = model_name
                logger.info(f"Successfully initialized reranker with model: {model_name}")
                self.initialized = True
                return
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {str(e)}")
                continue
        
        logger.error("All reranker models failed to initialize. Reranking will be disabled.")
        self.initialized = False
    
    def _initialize_model(self, model_name: str):
        """Initialize a specific model"""
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info(f"Loading model {model_name}")
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        
        # Set to evaluation mode
        self.reranker.eval()
    
    def rerank(self, query: str, matches: List[Dict[str, Any]], top_k: int = None, enable_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Rerank the given matches using the current reranker model.
        
        Args:
            query: The search query
            matches: List of match dictionaries containing 'text' and 'score' keys
            top_k: Number of top results to return (None returns all)
            
        Returns:
            List of reranked matches with updated scores
        """
        if not enable_reranking or not self.initialized or not matches:
            return matches[:top_k] if top_k else matches
        
        try:
            # Prepare query-document pairs
            pairs = [(query, match.get('combined_text', '')) for match in matches]
            
            # Get scores from reranker
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.reranker(**inputs)
                
                # Handle different model output formats
                if hasattr(outputs, 'logits'):
                    # BAAI models return logits directly
                    if len(outputs.logits.shape) == 2 and outputs.logits.shape[1] == 1:
                        # Single score per example (regression)
                        scores = outputs.logits.squeeze(-1).cpu().numpy()
                    else:
                        # Classification scores (softmax over classes)
                        scores = F.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                else:
                    # Some models return scores directly
                    scores = outputs.scores.cpu().numpy()
                
                # Ensure we have the right number of scores
                if len(scores) != len(matches):
                    logger.warning(f"Number of scores ({len(scores)}) doesn't match number of matches ({len(matches)})")
                    return matches[:top_k] if top_k else matches
            
            # Update matches with new scores
            for i, match in enumerate(matches):
                if i >= len(scores):  # Safety check
                    continue
                    
                original_score = match.get('score', 0)
                reranker_score = float(scores[i])
                
                # Combine scores (adjust weights as needed)
                combined_score = (0.3 * original_score) + (0.7 * reranker_score)
                
                # Update match with scores
                match.update({
                    'original_score': original_score,
                    'reranker_score': reranker_score,
                    'score': combined_score,
                    'reranker_model': self.current_model
                })
            
            # Sort by combined score
            reranked_matches = sorted(
                matches,
                key=lambda x: x.get('score', 0),
                reverse=True
            )
            
            return reranked_matches[:top_k] if top_k else reranked_matches
            
        except Exception as e:
            logger.error(f"Error during reranking with {self.current_model}: {str(e)}", exc_info=True)
            # Fall back to original matches if reranking fails
            return matches[:top_k] if top_k else matches

# Create a singleton instance
_reranker_instance = None

def get_reranker(enable_reranking: bool = True) -> RAGReranker:
    """Get or create the singleton reranker instance
    
    Args:
        enable_reranking: Whether reranking should be enabled
        
    Returns:
        RAGReranker instance (may be a dummy if reranking is disabled)
    """
    global _reranker_instance
    
    if not enable_reranking:
        # Return a dummy reranker that just passes through results
        class DummyReranker:
            def rerank(self, query, matches, top_k=None, **kwargs):
                return matches[:top_k] if top_k else matches
        return DummyReranker()
        
    if _reranker_instance is None:
        _reranker_instance = RAGReranker()
    return _reranker_instance

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    reranker = get_reranker(enable_reranking=True)
    
    test_matches = [
        {"combined_text": "Sales data for Q2 2023", "score": 0.85},
        {"combined_text": "Quarterly revenue report", "score": 0.78},
        {"combined_text": "Customer satisfaction survey", "score": 0.65}
    ]
    
    query = "Show me sales reports from last quarter"
    reranked = reranker.rerank(query, test_matches)
    
    print("Original order:")
    for i, m in enumerate(test_matches):
        print(f"{i+1}. {m['combined_text']} (score: {m['score']:.3f})")
    
    print("\nReranked order:")
    for i, m in enumerate(reranked):
        print(f"{i+1}. {m['combined_text']} (score: {m['score']:.3f}, reranker: {m['reranker_score']:.3f})")
        
    # Test with reranking disabled
    print("\n=== Testing with reranking disabled ===")
    reranker = get_reranker(enable_reranking=False)
    non_reranked = reranker.rerank(query, test_matches.copy())
    
    print("\nOriginal order (reranking disabled):")
    for i, m in enumerate(non_reranked):
        print(f"{i+1}. {m['combined_text']} (score: {m['score']:.3f})")