"""
Context enhancement module for RAG system.
This module manages and enhances the context provided to LLMs with retrieved information.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from collections import defaultdict

from rag.retriever import BaseRetriever
from utils.text_processing import clean_text

# Setup logging
logger = logging.getLogger(__name__)

class ContextEnhancer:
    """Class for enhancing LLM context with retrieved information."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        max_context_length: int = 4000,
        relevance_threshold: float = 0.5,
        include_metadata: bool = True
    ):
        """
        Initialize the context enhancer.
        
        Args:
            retriever: Retriever for finding relevant context
            max_context_length: Maximum length of context in tokens
            relevance_threshold: Minimum relevance score for inclusion
            include_metadata: Whether to include document metadata in context
        """
        self.retriever = retriever
        self.max_context_length = max_context_length
        self.relevance_threshold = relevance_threshold
        self.include_metadata = include_metadata
        self.conversation_context = []
    
    def add_to_conversation(self, query: str, response: str):
        """
        Add query-response pair to conversation context.
        
        Args:
            query: User query
            response: System response
        """
        self.conversation_context.append({
            "role": "user",
            "content": query
        })
        self.conversation_context.append({
            "role": "assistant",
            "content": response
        })
        
        # If we have contextual retriever, update its history too
        if hasattr(self.retriever, "add_to_history"):
            self.retriever.add_to_history(query, response)
    
    def clear_conversation(self):
        """Clear conversation context."""
        self.conversation_context = []
        
        # If we have contextual retriever, clear its history too
        if hasattr(self.retriever, "conversation_history"):
            self.retriever.conversation_history = []
    
    def get_conversation_context(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        Get recent conversation context.
        
        Args:
            max_turns: Maximum number of conversation turns to include
            
        Returns:
            List of conversation messages
        """
        if not self.conversation_context:
            return []
        
        # Get last max_turns * 2 messages (each turn is a query-response pair)
        return self.conversation_context[-max_turns*2:]
    
    def format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context text.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context text
        """
        if not documents:
            return ""
        
        context = "Relevant Information:\n\n"
        
        for i, doc in enumerate(documents, 1):
            # Extract content and metadata
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Add document header with metadata
            context += f"--- Document {i} "
            if self.include_metadata and metadata:
                # Include key metadata fields
                if "table_name" in metadata:
                    context += f"| Table: {metadata['table_name']} "
                if "column_name" in metadata:
                    context += f"| Column: {metadata['column_name']} "
                if "content_type" in metadata:
                    context += f"| Type: {metadata['content_type']} "
            
            context += "---\n"
            
            # Add document content
            context += f"{content}\n\n"
        
        return context
    
    def organize_by_table(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize retrieved documents by table name.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Dictionary mapping table names to lists of documents
        """
        by_table = defaultdict(list)
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            table_name = metadata.get("table_name", "Unknown")
            by_table[table_name].append(doc)
        
        return by_table
    
    def format_structured_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into structured context organized by table.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Structured context text
        """
        if not documents:
            return ""
        
        # Organize documents by table
        by_table = self.organize_by_table(documents)
        
        context = "Relevant Database Information:\n\n"
        
        # Process each table
        for table_name, table_docs in by_table.items():
            context += f"### Table: {table_name}\n\n"
            
            # Group by content type
            table_definitions = [d for d in table_docs if d.get("metadata", {}).get("content_type") == "table_definition"]
            column_definitions = [d for d in table_docs if d.get("metadata", {}).get("content_type") == "column_definition"]
            schema_definitions = [d for d in table_docs if d.get("metadata", {}).get("content_type") == "schema"]
            other_docs = [d for d in table_docs if d.get("metadata", {}).get("content_type") not in ["table_definition", "column_definition", "schema"]]
            
            # Add table definitions
            if table_definitions:
                # Use the highest scoring table definition
                best_table_def = max(table_definitions, key=lambda d: d.get("score", 0))
                context += f"{best_table_def.get('content', '')}\n\n"
            
            # Add column definitions
            if column_definitions:
                context += "#### Columns:\n"
                # Sort by score
                sorted_cols = sorted(column_definitions, key=lambda d: d.get("score", 0), reverse=True)
                for doc in sorted_cols[:5]:  # Limit to top 5 columns
                    metadata = doc.get("metadata", {})
                    col_name = metadata.get("column_name", "Unknown")
                    data_type = metadata.get("data_type", "")
                    
                    if data_type:
                        context += f"- {col_name} ({data_type}): "
                    else:
                        context += f"- {col_name}: "
                    
                    # Extract description from content
                    content = doc.get("content", "")
                    desc_parts = content.split("Description:", 1)
                    if len(desc_parts) > 1:
                        context += desc_parts[1].strip()
                    else:
                        context += content
                    
                    context += "\n"
                
                context += "\n"
            
            # Add schema definitions if available
            if schema_definitions:
                context += "#### Schema Definition:\n"
                # Use the highest scoring schema definition
                best_schema = max(schema_definitions, key=lambda d: d.get("score", 0))
                context += f"```sql\n{best_schema.get('content', '')}\n```\n\n"
            
            # Add other documents
            if other_docs:
                for doc in other_docs:
                    context += f"{doc.get('content', '')}\n\n"
        
        return context
    
    def enhance_query_context(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        structured: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhance query with relevant context from retrieval.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            filter: Optional filter for retrieval
            structured: Whether to use structured formatting
            
        Returns:
            Tuple of (enhanced context, retrieved documents)
        """
        # Clean the query
        query = clean_text(query)
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=query,
            k=k,
            filter=filter
        )
        
        # Filter by relevance threshold
        if self.relevance_threshold > 0:
            documents = [doc for doc in documents if doc.get("score", 0) >= self.relevance_threshold]
        
        # Format context
        if structured:
            context = self.format_structured_context(documents)
        else:
            context = self.format_retrieved_documents(documents)
        
        return context, documents
    
    def build_enhanced_prompt(
        self,
        query: str,
        system_prompt: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_conversation: bool = True,
        max_conversation_turns: int = 3,
        structured_context: bool = True
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Build an enhanced prompt with retrieved context and conversation history.
        
        Args:
            query: User query
            system_prompt: Base system prompt
            k: Number of documents to retrieve
            filter: Optional filter for retrieval
            include_conversation: Whether to include conversation history
            max_conversation_turns: Maximum number of conversation turns to include
            structured_context: Whether to use structured context formatting
            
        Returns:
            Tuple of (enhanced prompt messages, retrieved documents)
        """
        # Get relevant context
        context, documents = self.enhance_query_context(
            query=query,
            k=k,
            filter=filter,
            structured=structured_context
        )
        
        # Build messages array
        messages = []
        
        # Add system message with enhanced context
        if context:
            enhanced_system_prompt = f"{system_prompt}\n\n{context}"
        else:
            enhanced_system_prompt = system_prompt
        
        messages.append({
            "role": "system",
            "content": enhanced_system_prompt
        })
        
        # Add conversation context if requested
        if include_conversation:
            conversation = self.get_conversation_context(max_turns=max_conversation_turns)
            messages.extend(conversation)
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages, documents
    
    def manage_context_window(self, prompt: str, max_length: int = None) -> str:
        """
        Manage context window to ensure it fits within token limits.
        
        Args:
            prompt: Full prompt text
            max_length: Maximum length in tokens (defaults to self.max_context_length)
            
        Returns:
            Trimmed prompt that fits in context window
        """
        max_length = max_length or self.max_context_length
        
        # Simple token estimation (words + punctuation)
        estimated_tokens = len(prompt.split()) + prompt.count(".") + prompt.count(",")
        
        if estimated_tokens <= max_length:
            return prompt
        
        # If too long, trim relevant information section
        sections = prompt.split("Relevant Information:", 1)
        
        if len(sections) == 1:
            # No "Relevant Information" section, just truncate
            words = prompt.split()
            return " ".join(words[:max_length])
        
        base_prompt = sections[0]
        relevant_info = "Relevant Information:" + sections[1]
        
        # Estimate base prompt tokens
        base_tokens = len(base_prompt.split()) + base_prompt.count(".") + base_prompt.count(",")
        
        # Calculate how many tokens we have left for relevant info
        remaining_tokens = max_length - base_tokens
        
        if remaining_tokens <= 50:  # If almost no room left
            return base_prompt
        
        # Truncate relevant information
        info_parts = relevant_info.split("--- Document ")
        header = info_parts[0]
        docs = ["--- Document " + part for part in info_parts[1:]]
        
        result = base_prompt + header
        tokens_used = base_tokens + len(header.split()) + header.count(".") + header.count(",")
        
        # Add documents until we reach the limit
        for doc in docs:
            doc_tokens = len(doc.split()) + doc.count(".") + doc.count(",")
            
            if tokens_used + doc_tokens > max_length:
                break
            
            result += doc
            tokens_used += doc_tokens
        
        return result
