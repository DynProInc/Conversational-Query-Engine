"""
Document processing module for RAG system.
This module handles processing and chunking of data dictionaries and schema definitions.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import os
import json
import csv
import pandas as pd
import logging
import re
from pathlib import Path

# Import utility functions
from utils.text_processing import (
    clean_text,
    split_text_by_tokens,
    split_text_by_semantic_units,
    chunk_schema_definition,
    extract_table_and_column_info
)
from utils.embedding_utils import EmbeddingGenerator
from rag.vector_store import VectorStore, VectorStoreFactory, get_default_embedding_function

# Setup logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class for processing documents into chunks for vector storage."""
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        use_semantic_chunking: bool = True
    ):
        """
        Initialize the document processor.
        
        Args:
            embedding_generator: Optional embedding generator
            chunk_size: Maximum size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            use_semantic_chunking: Whether to use semantic chunking
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a text document into chunks with embeddings.
        
        Args:
            text: Text to process
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of processed chunks with text, embeddings, and metadata
        """
        # Clean the text
        text = clean_text(text)
        
        # Split text into chunks
        if self.use_semantic_chunking:
            chunks = split_text_by_semantic_units(text, self.chunk_size)
        else:
            chunks = split_text_by_tokens(text, self.chunk_size, self.chunk_overlap)
        
        # Generate embeddings for chunks
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        # Create result documents with embeddings and metadata
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_schema_definition(self, schema_text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a schema definition into semantically meaningful chunks.
        
        Args:
            schema_text: Schema definition text
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of processed chunks with text, embeddings, and metadata
        """
        # Chunk the schema definition
        chunked_schema = chunk_schema_definition(schema_text)
        
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunked_schema]
        
        # Generate embeddings for chunks
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Create result documents with embeddings and metadata
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunked_schema, embeddings)):
            # Merge metadata from chunk and provided metadata
            combined_metadata = {
                **chunk["metadata"],
                "chunk_id": i,
                "total_chunks": len(chunked_schema),
                **(metadata or {})
            }
            
            doc = {
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": combined_metadata
            }
            documents.append(doc)
        
        return documents
    
    def process_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a list of documents for RAG.
        
        Args:
            documents: List of documents with 'text', 'metadata', and optional 'id'
            collection_name: Name of the document collection
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            metadata: Optional metadata to add to all documents
            
        Returns:
            Processing results with processed documents and statistics
        """
        processed_docs = []
        total_chunks = 0
        
        # Use provided chunk parameters or defaults
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # Temporarily update chunk parameters
        original_chunk_size = self.chunk_size
        original_chunk_overlap = self.chunk_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            for i, doc in enumerate(documents):
                # Extract text and metadata
                text = doc.get('text', '')
                doc_metadata = doc.get('metadata', {})
                doc_id = doc.get('id', f'doc_{i}')
                
                # Combine document metadata with global metadata
                combined_metadata = {
                    'document_id': doc_id,
                    'collection_name': collection_name,
                    **(metadata or {}),
                    **doc_metadata
                }
                
                # Process the document text
                chunks = self.process_text(text, combined_metadata)
                processed_docs.extend(chunks)
                total_chunks += len(chunks)
                
                logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")
            
            logger.info(f"Total documents processed: {len(documents)}, Total chunks: {total_chunks}")
            
            return {
                'success': True,
                'collection_name': collection_name,
                'documents_processed': len(documents),
                'total_chunks': total_chunks,
                'processed_documents': processed_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return {
                'success': False,
                'collection_name': collection_name,
                'error': str(e),
                'documents_processed': 0,
                'total_chunks': 0
            }
        finally:
            # Restore original chunk parameters
            self.chunk_size = original_chunk_size
            self.chunk_overlap = original_chunk_overlap
    
    def process_csv_data_dictionary(
        self, 
        file_path: str, 
        client_name: str,
        table_col: str = "Table", 
        column_col: str = "Column", 
        description_col: str = "Description"
    ) -> List[Dict[str, Any]]:
        """
        Process a CSV data dictionary file.
        
        Args:
            file_path: Path to CSV file
            client_name: Name of the client
            table_col: Column name for table names
            column_col: Column name for column names
            description_col: Column name for descriptions
            
        Returns:
            List of processed chunks with text, embeddings, and metadata
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns exist
            required_cols = [table_col, column_col, description_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV file is missing required columns: {missing_cols}")
            
            # Process each table-column pair
            documents = []
            
            # Group by table
            table_groups = df.groupby(table_col)
            
            for table_name, table_df in table_groups:
                # Create table-level document
                table_desc = f"Table {table_name}"
                if "TableDescription" in table_df.columns:
                    table_desc = table_df["TableDescription"].iloc[0]
                    if not isinstance(table_desc, str):
                        table_desc = f"Table {table_name}"
                
                table_text = f"Table: {table_name}\nDescription: {table_desc}\n"
                
                # Add columns overview
                table_text += "Columns:\n"
                for _, row in table_df.iterrows():
                    col_name = row[column_col]
                    col_desc = row[description_col]
                    data_type = row.get("DataType", "")
                    
                    if data_type:
                        table_text += f"- {col_name} ({data_type}): {col_desc}\n"
                    else:
                        table_text += f"- {col_name}: {col_desc}\n"
                
                # Process table text
                table_docs = self.process_text(table_text, {
                    "client_id": client_name,
                    "content_type": "table_definition",
                    "table_name": table_name
                })
                documents.extend(table_docs)
                
                # Process each column separately for more granular retrieval
                for _, row in table_df.iterrows():
                    col_name = row[column_col]
                    col_desc = row[description_col]
                    data_type = row.get("DataType", "")
                    
                    # Skip if no meaningful description
                    if not isinstance(col_desc, str) or not col_desc.strip():
                        continue
                    
                    # Create column text
                    col_text = f"Table: {table_name}\nColumn: {col_name}\n"
                    if data_type:
                        col_text += f"Data Type: {data_type}\n"
                    col_text += f"Description: {col_desc}"
                    
                    # Process column text
                    col_docs = self.process_text(col_text, {
                        "client_id": client_name,
                        "content_type": "column_definition",
                        "table_name": table_name,
                        "column_name": col_name,
                        "data_type": data_type
                    })
                    documents.extend(col_docs)
            
            return documents
        
        except Exception as e:
            logger.error(f"Error processing CSV data dictionary {file_path}: {e}")
            raise

class DataDictionaryIndexer:
    """Class for indexing data dictionaries into vector stores."""
    
    def __init__(
        self,
        vector_store_type: str = "chroma",
        embedding_generator: Optional[EmbeddingGenerator] = None,
        base_path: str = "vector_db",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
    ):
        """
        Initialize the data dictionary indexer.
        
        Args:
            vector_store_type: Type of vector store to use
            embedding_generator: Optional embedding generator
            base_path: Base path for vector stores
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Chunk overlap in tokens
        """
        self.vector_store_type = vector_store_type
        self.base_path = base_path
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Create document processor
        self.document_processor = DocumentProcessor(
            embedding_generator=self.embedding_generator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_chunking=True
        )
        
        # Create directory for vector stores if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        
        # Store client vector stores
        self.vector_stores = {}
    
    def get_vector_store(self, client_name: str) -> VectorStore:
        """
        Get or create a vector store for a client.
        
        Args:
            client_name: Name of the client
            
        Returns:
            VectorStore instance
        """
        if client_name in self.vector_stores:
            return self.vector_stores[client_name]
        
        # Create vector store for client
        if self.vector_store_type == "chroma":
            # For ChromaDB, we need an embedding function
            embedding_function = get_default_embedding_function()
            
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=self.vector_store_type,
                collection_name=client_name,
                persist_directory=os.path.join(self.base_path, client_name),
                embedding_function=embedding_function
            )
        else:
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=self.vector_store_type,
                collection_name=client_name,
                embedding_dim=self.embedding_generator.embedding_dimension,
                index_path=os.path.join(self.base_path, f"{client_name}.index"),
                metadata_path=os.path.join(self.base_path, f"{client_name}_metadata.json"),
                content_path=os.path.join(self.base_path, f"{client_name}_content.json")
            )
        
        self.vector_stores[client_name] = vector_store
        return vector_store
    
    def index_client_data_dictionaries(
        self, 
        client_name: str,
        data_dict_path: str
    ) -> int:
        """
        Index all data dictionaries for a client.
        
        Args:
            client_name: Name of the client
            data_dict_path: Path to data dictionary files
            
        Returns:
            Number of chunks indexed
        """
        data_dict_dir = Path(data_dict_path)
        
        # Check if directory exists
        if not data_dict_dir.exists() or not data_dict_dir.is_dir():
            logger.error(f"Data dictionary path does not exist: {data_dict_path}")
            return 0
        
        # Get vector store for client
        vector_store = self.get_vector_store(client_name)
        
        # Process all CSV files in directory
        total_indexed = 0
        
        for file_path in data_dict_dir.glob("*.csv"):
            try:
                logger.info(f"Processing data dictionary: {file_path}")
                
                # Process CSV data dictionary
                documents = self.document_processor.process_csv_data_dictionary(
                    file_path=str(file_path),
                    client_name=client_name
                )
                
                if not documents:
                    logger.warning(f"No documents extracted from {file_path}")
                    continue
                
                # Extract texts, embeddings, and metadata
                texts = [doc["text"] for doc in documents]
                embeddings = [doc["embedding"] for doc in documents]
                metadatas = [doc["metadata"] for doc in documents]
                
                # Add to vector store
                ids = vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                logger.info(f"Indexed {len(ids)} chunks from {file_path}")
                total_indexed += len(ids)
            
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
        
        logger.info(f"Indexed {total_indexed} chunks for client {client_name}")
        return total_indexed
    
    def index_all_clients_data_dictionaries(self, base_path: str = "config/clients/data_dictionaries") -> Dict[str, int]:
        """
        Index data dictionaries for all clients.
        
        Args:
            base_path: Base path for client data dictionaries
            
        Returns:
            Dictionary mapping client names to number of chunks indexed
        """
        base_dir = Path(base_path)
        
        if not base_dir.exists() or not base_dir.is_dir():
            logger.error(f"Base path does not exist: {base_path}")
            return {}
        
        # Find all subdirectories (each representing a client)
        results = {}
        
        for client_dir in base_dir.glob("*"):
            if client_dir.is_dir():
                client_name = client_dir.name
                logger.info(f"Indexing data dictionaries for client: {client_name}")
                
                # Index client data dictionaries
                count = self.index_client_data_dictionaries(
                    client_name=client_name,
                    data_dict_path=str(client_dir)
                )
                
                results[client_name] = count
        
        return results
