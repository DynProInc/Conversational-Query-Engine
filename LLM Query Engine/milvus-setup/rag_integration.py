#!/usr/bin/env python3
"""
RAG Integration with LLM Query Engine
=====================================

This module integrates the Milvus RAG system with the existing LLM Query Engine.
It provides API endpoint extensions and query generator enhancements to use
client-specific RAG collections for token optimization and improved accuracy.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import Milvus RAG system
from milvus_setup.multi_client_rag import MultiClientMilvusRAG

# Import query generators from parent directory
try:
    from llm_query_generator import generate_query as openai_generate_query
    from claude_query_generator import generate_query as claude_generate_query
    from gemini_query_generator import generate_query as gemini_generate_query
    from client_manager import get_client_context, verify_client
    from token_logger import log_token_usage
except ImportError as e:
    print(f"Error importing LLM Query Engine modules: {e}")
    print("Please run this script from the LLM Query Engine directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "rag_integration.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGIntegration")

class RAGQueryEnhancer:
    """
    Enhances query generators with RAG capabilities for token optimization
    while maintaining client isolation
    """
    
    def __init__(self, milvus_host="localhost", milvus_port="19530"):
        """Initialize the RAG Query Enhancer"""
        logger.info("Initializing RAG Query Enhancer")
        self.rag_system = MultiClientMilvusRAG(milvus_host, milvus_port)
        
        # Track loaded clients
        self.client_setup_status = {}
        
    def setup_client(self, client_id):
        """Set up a client in the RAG system if not already set up"""
        if client_id in self.client_setup_status:
            return self.client_setup_status[client_id]
            
        # Get client data dictionary path
        data_dict_path = Path(f"{parent_dir}/config/clients/data_dictionaries/{client_id}/data_dictionary.csv")
        
        if not data_dict_path.exists():
            logger.warning(f"Data dictionary not found for client {client_id} at {data_dict_path}")
            self.client_setup_status[client_id] = False
            return False
        
        # Set up client in RAG system
        success = self.rag_system.setup_client(client_id, str(data_dict_path))
        self.client_setup_status[client_id] = success
        
        if success:
            logger.info(f"Successfully set up RAG for client {client_id}")
        else:
            logger.error(f"Failed to set up RAG for client {client_id}")
            
        return success
        
    def enhance_openai_query_generator(self):
        """Enhance the OpenAI query generator with RAG capabilities"""
        original_generate_query = openai_generate_query
        
        def enhanced_generate_query(client_id, query, model=None, data_dictionary_path=None):
            """Enhanced query generator with RAG optimization"""
            # Verify client exists
            if not verify_client(client_id):
                raise ValueError(f"Client {client_id} does not exist")
            
            # Set up client in RAG system if not already set up
            self.setup_client(client_id)
            
            # Get optimized prompt with RAG
            try:
                optimized_prompt = self.rag_system.get_optimized_prompt(client_id, query)
                
                # Log token savings
                token_savings = self.rag_system.estimate_token_savings(client_id, query)
                logger.info(
                    f"OpenAI RAG Optimization - Client: {client_id}, "
                    f"Saved: {token_savings['savings_tokens']} tokens "
                    f"({token_savings['savings_percentage']}%)"
                )
                
                # Log the optimized token usage
                log_token_usage(
                    client_id=client_id,
                    model=model or "gpt-4",  # Use provided model or default
                    tokens_in=token_savings['optimized_tokens'],
                    tokens_out=0,  # Will be updated after API call
                    query_type="nlq_to_sql"
                )
                
                # Call original function but with our optimized prompt
                return original_generate_query(client_id, optimized_prompt, model, data_dictionary_path)
                
            except Exception as e:
                logger.error(f"RAG optimization failed for client {client_id}: {e}")
                logger.info(f"Falling back to standard query generation")
                # Fall back to original method if RAG fails
                return original_generate_query(client_id, query, model, data_dictionary_path)
        
        # Replace original function with enhanced version
        openai_generate_query.__code__ = enhanced_generate_query.__code__
        logger.info("Enhanced OpenAI query generator with RAG capabilities")
        
    def enhance_claude_query_generator(self):
        """Enhance the Claude query generator with RAG capabilities"""
        original_generate_query = claude_generate_query
        
        def enhanced_generate_query(client_id, query, model=None, data_dictionary_path=None):
            """Enhanced query generator with RAG optimization"""
            # Verify client exists
            if not verify_client(client_id):
                raise ValueError(f"Client {client_id} does not exist")
            
            # Set up client in RAG system if not already set up
            self.setup_client(client_id)
            
            # Get optimized prompt with RAG
            try:
                optimized_prompt = self.rag_system.get_optimized_prompt(client_id, query)
                
                # Log token savings
                token_savings = self.rag_system.estimate_token_savings(client_id, query)
                logger.info(
                    f"Claude RAG Optimization - Client: {client_id}, "
                    f"Saved: {token_savings['savings_tokens']} tokens "
                    f"({token_savings['savings_percentage']}%)"
                )
                
                # Log the optimized token usage
                log_token_usage(
                    client_id=client_id,
                    model=model or "claude-3-5-sonnet",  # Use provided model or default
                    tokens_in=token_savings['optimized_tokens'],
                    tokens_out=0,  # Will be updated after API call
                    query_type="nlq_to_sql"
                )
                
                # Call original function but with our optimized prompt
                return original_generate_query(client_id, optimized_prompt, model, data_dictionary_path)
                
            except Exception as e:
                logger.error(f"RAG optimization failed for client {client_id}: {e}")
                logger.info(f"Falling back to standard query generation")
                # Fall back to original method if RAG fails
                return original_generate_query(client_id, query, model, data_dictionary_path)
        
        # Replace original function with enhanced version
        claude_generate_query.__code__ = enhanced_generate_query.__code__
        logger.info("Enhanced Claude query generator with RAG capabilities")
        
    def enhance_gemini_query_generator(self):
        """Enhance the Gemini query generator with RAG capabilities"""
        original_generate_query = gemini_generate_query
        
        def enhanced_generate_query(client_id, query, model=None, data_dictionary_path=None):
            """Enhanced query generator with RAG optimization"""
            # Verify client exists
            if not verify_client(client_id):
                raise ValueError(f"Client {client_id} does not exist")
            
            # Set up client in RAG system if not already set up
            self.setup_client(client_id)
            
            # Get optimized prompt with RAG
            try:
                optimized_prompt = self.rag_system.get_optimized_prompt(client_id, query)
                
                # Log token savings
                token_savings = self.rag_system.estimate_token_savings(client_id, query)
                logger.info(
                    f"Gemini RAG Optimization - Client: {client_id}, "
                    f"Saved: {token_savings['savings_tokens']} tokens "
                    f"({token_savings['savings_percentage']}%)"
                )
                
                # Log the optimized token usage
                log_token_usage(
                    client_id=client_id,
                    model=model or "gemini-1.5-pro",  # Use provided model or default
                    tokens_in=token_savings['optimized_tokens'],
                    tokens_out=0,  # Will be updated after API call
                    query_type="nlq_to_sql"
                )
                
                # Call original function but with our optimized prompt
                return original_generate_query(client_id, optimized_prompt, model, data_dictionary_path)
                
            except Exception as e:
                logger.error(f"RAG optimization failed for client {client_id}: {e}")
                logger.info(f"Falling back to standard query generation")
                # Fall back to original method if RAG fails
                return original_generate_query(client_id, query, model, data_dictionary_path)
        
        # Replace original function with enhanced version
        gemini_generate_query.__code__ = enhanced_generate_query.__code__
        logger.info("Enhanced Gemini query generator with RAG capabilities")

    def enhance_all_generators(self):
        """Enhance all query generators with RAG capabilities"""
        self.enhance_openai_query_generator()
        self.enhance_claude_query_generator()
        self.enhance_gemini_query_generator()
        logger.info("Enhanced all query generators with RAG capabilities")

def initialize_rag_system():
    """Initialize the RAG system and enhance query generators"""
    try:
        enhancer = RAGQueryEnhancer()
        enhancer.enhance_all_generators()
        logger.info("Successfully initialized RAG system and enhanced query generators")
        return enhancer
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return None

if __name__ == "__main__":
    # Simple test to ensure integration works
    enhancer = initialize_rag_system()
    
    if enhancer:
        print("RAG Query Enhancer initialized successfully!")
        print("Checking client setup...")
        
        # Test with existing clients
        for client_id in ["mts", "penguin"]:
            success = enhancer.setup_client(client_id)
            print(f"Client {client_id} setup: {'✅' if success else '❌'}")
            
        print("\nRAG system is ready for use with the LLM Query Engine")
        print("Token optimization is active for all query generators")
    else:
        print("Failed to initialize RAG Query Enhancer")
        print("Please check the logs for more information")
