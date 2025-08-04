#!/usr/bin/env python3
"""
Model RAG Integration
====================

Integrates the Multi-Client RAG system with existing LLM query generators.
Maintains strict client isolation and proper error handling while enhancing
prompts with relevant schema context.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRAGIntegration")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import client RAG manager
from client_rag_manager import get_rag_manager

# Track original query generators for fallback
_original_generators = {}

def _safe_import(module_name: str, function_name: str = "generate_query") -> Optional[Callable]:
    """
    Safely import a function from a module
    
    Args:
        module_name: Name of the module to import
        function_name: Name of the function to import
        
    Returns:
        Imported function or None if import fails
    """
    try:
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error importing {function_name} from {module_name}: {e}")
        return None


def _enhance_with_rag(
    original_generator: Callable,
    generator_name: str,
    client_id: str, 
    query: str, 
    **kwargs
) -> Dict[str, Any]:
    """
    Enhance query generator with RAG capabilities
    
    Args:
        original_generator: Original query generator function
        generator_name: Name of the generator (for logging)
        client_id: Client identifier
        query: User query
        **kwargs: Additional arguments for the generator
        
    Returns:
        Generated query result with RAG enhancement
    """
    start_time = time.time()
    
    # Get RAG manager
    rag_manager = get_rag_manager()
    
    try:
        # Use RAG to get optimized prompt if client is valid
        optimized_query = rag_manager.get_optimized_prompt(client_id, query)
        
        # Use optimized query instead of original query
        result = original_generator(client_id=client_id, query=optimized_query, **kwargs)
        
        # Add RAG metadata to result
        if isinstance(result, dict):
            # Add RAG metadata to the result
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"]["rag_enhanced"] = True
            result["metadata"]["rag_processing_time"] = round(time.time() - start_time, 3)
            
            # Attempt to get token savings if available
            try:
                token_savings = rag_manager.rag_system.estimate_token_savings(client_id, query)
                if token_savings:
                    result["metadata"]["rag_token_savings"] = token_savings
            except:
                pass
        
        logger.info(f"Successfully enhanced {generator_name} query with RAG for client '{client_id}'")
        return result
        
    except Exception as e:
        logger.warning(f"RAG enhancement failed for {generator_name}: {e}. Falling back to original generator.")
        
        # Fall back to original query generator without RAG enhancement
        return original_generator(client_id=client_id, query=query, **kwargs)


def enhance_openai_generator():
    """Enhance OpenAI query generator with RAG capabilities"""
    global _original_generators
    
    # Import the OpenAI query generator
    openai_generator = _safe_import("llm_query_generator", "generate_query")
    if not openai_generator:
        logger.error("Could not import OpenAI query generator")
        return False
    
    # Store original generator for fallback
    _original_generators["openai"] = openai_generator
    
    # Replace with enhanced version
    def enhanced_openai_generator(client_id: str, query: str, **kwargs) -> Dict[str, Any]:
        return _enhance_with_rag(
            _original_generators["openai"], 
            "OpenAI", 
            client_id, 
            query, 
            **kwargs
        )
    
    # Update the module
    import llm_query_generator
    llm_query_generator.generate_query = enhanced_openai_generator
    logger.info("Enhanced OpenAI query generator with RAG capabilities")
    return True


def enhance_claude_generator():
    """Enhance Claude query generator with RAG capabilities"""
    global _original_generators
    
    # Import the Claude query generator
    claude_generator = _safe_import("claude_query_generator", "generate_query")
    if not claude_generator:
        logger.error("Could not import Claude query generator")
        return False
    
    # Store original generator for fallback
    _original_generators["claude"] = claude_generator
    
    # Replace with enhanced version
    def enhanced_claude_generator(client_id: str, query: str, **kwargs) -> Dict[str, Any]:
        return _enhance_with_rag(
            _original_generators["claude"], 
            "Claude", 
            client_id, 
            query, 
            **kwargs
        )
    
    # Update the module
    import claude_query_generator
    claude_query_generator.generate_query = enhanced_claude_generator
    logger.info("Enhanced Claude query generator with RAG capabilities")
    return True


def enhance_gemini_generator():
    """Enhance Gemini query generator with RAG capabilities"""
    global _original_generators
    
    # Import the Gemini query generator
    gemini_generator = _safe_import("gemini_query_generator", "generate_query")
    if not gemini_generator:
        logger.error("Could not import Gemini query generator")
        return False
    
    # Store original generator for fallback
    _original_generators["gemini"] = gemini_generator
    
    # Replace with enhanced version
    def enhanced_gemini_generator(client_id: str, query: str, **kwargs) -> Dict[str, Any]:
        return _enhance_with_rag(
            _original_generators["gemini"], 
            "Gemini", 
            client_id, 
            query, 
            **kwargs
        )
    
    # Update the module
    import gemini_query_generator
    gemini_query_generator.generate_query = enhanced_gemini_generator
    logger.info("Enhanced Gemini query generator with RAG capabilities")
    return True


def initialize_rag_integration() -> Dict[str, bool]:
    """
    Initialize RAG integration with all LLM query generators
    
    Returns:
        Dictionary with integration status for each LLM provider
    """
    # Preload RAG manager
    rag_manager = get_rag_manager()
    
    # Enhance generators
    results = {
        "openai": enhance_openai_generator(),
        "claude": enhance_claude_generator(),
        "gemini": enhance_gemini_generator()
    }
    
    # Log overall status
    success_count = sum(1 for status in results.values() if status)
    if success_count == len(results):
        logger.info("Successfully enhanced all query generators with RAG capabilities")
    else:
        logger.warning(f"Enhanced {success_count}/{len(results)} query generators with RAG capabilities")
    
    return results


def restore_original_generators():
    """Restore original query generators (useful for testing)"""
    global _original_generators
    
    if "openai" in _original_generators:
        import llm_query_generator
        llm_query_generator.generate_query = _original_generators["openai"]
    
    if "claude" in _original_generators:
        import claude_query_generator
        claude_query_generator.generate_query = _original_generators["claude"]
    
    if "gemini" in _original_generators:
        import gemini_query_generator
        gemini_query_generator.generate_query = _original_generators["gemini"]
    
    logger.info("Restored original query generators")


# Easy access function for other modules
def rag_enhanced(active: bool = True) -> bool:
    """
    Enable or disable RAG enhancement for all generators
    
    Args:
        active: True to enable, False to disable
        
    Returns:
        True if successful, False otherwise
    """
    if active:
        return all(initialize_rag_integration().values())
    else:
        restore_original_generators()
        return True


if __name__ == "__main__":
    # Initialize RAG integration when run directly
    integration_status = initialize_rag_integration()
    
    # Display status
    for model, status in integration_status.items():
        print(f"{model.capitalize()}: {'✓ Enhanced' if status else '× Failed'}")
        
    # Show test instructions
    print("\nTo test the enhanced generators, import them directly:")
    print("from llm_query_generator import generate_query as openai_query")
    print("from claude_query_generator import generate_query as claude_query")
    print("from gemini_query_generator import generate_query as gemini_query")
