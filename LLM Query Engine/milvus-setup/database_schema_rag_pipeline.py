
#!/usr/bin/env python3
"""
Database Schema RAG System with Milvus
======================================

Complete implementation of Retrieval-Augmented Generation (RAG) system
for database schema retrieval using Milvus vector database.

This solves the problem of sending entire database schema to LLM every time,
instead retrieves only relevant schema information based on user queries.

Author: Database RAG Implementation
Date: 2025
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass

# Core dependencies
import openai
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    utility
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SchemaRecord:
    """Data class for database schema records"""
    db_schema: str
    table_name: str
    column_name: str
    data_type: str
    distinct_values: str
    description: str
    combined_text: str

class DatabaseSchemaRAG:
    """
    RAG system for efficient database schema retrieval using Milvus
    """

    def __init__(self, 
                 openai_api_key: str,
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 collection_name: str = "database_schema_collection"):

        self.openai_api_key = openai_api_key
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        # Initialize Milvus connection
        self._connect_to_milvus()

    def _connect_to_milvus(self):
        """Establish connection to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _prepare_schema_text(self, schema_record: Dict[str, Any]) -> str:
        """Prepare combined text representation of schema record for embedding"""
        combined_text = f"""Schema: {schema_record['DB_SCHEMA']}
Table: {schema_record['TABLE_NAME']}
Column: {schema_record['COLUMN_NAME']}
Data Type: {schema_record['DATA_TYPE']}
Description: {schema_record['DESCRIPTION']}
Distinct Values: {schema_record.get('DISTINCT_VALUES', 'N/A')}"""
        return combined_text

    def create_collection(self, drop_existing: bool = False):
        """Create Milvus collection for storing schema embeddings"""

        if utility.has_collection(self.collection_name):
            if drop_existing:
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                return Collection(self.collection_name)

        # Define collection schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Primary key"
            ),
            FieldSchema(
                name="schema_name",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="Database schema name"
            ),
            FieldSchema(
                name="table_name",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="Table name"
            ),
            FieldSchema(
                name="column_name",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="Column name"
            ),
            FieldSchema(
                name="data_type",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Column data type"
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=2000,
                description="Column description"
            ),
            FieldSchema(
                name="distinct_values",
                dtype=DataType.VARCHAR,
                max_length=1000,
                description="Distinct values in column"
            ),
            FieldSchema(
                name="combined_text",
                dtype=DataType.VARCHAR,
                max_length=3000,
                description="Combined text for context"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dimension,
                description="Text embedding vector"
            )
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Database schema information with embeddings for RAG"
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2
        )

        logger.info(f"Created collection: {self.collection_name}")
        return collection

    def create_index(self, collection: Collection):
        """Create vector index for efficient similarity search"""
        index_params = {
            "metric_type": "IP",  # Inner Product similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}  # Fine-grained clustering for better quality
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info("Created vector index")

    def load_and_embed_schema_data(self, schema_data: List[Dict[str, Any]]):
        """Load database schema data and create embeddings"""

        collection = self.create_collection(drop_existing=True)

        # Prepare data for insertion
        schema_names = []
        table_names = []
        column_names = []
        data_types = []
        descriptions = []
        distinct_values = []
        combined_texts = []
        embeddings = []

        logger.info(f"Processing {len(schema_data)} schema records...")

        for i, record in enumerate(schema_data):
            # Prepare combined text
            combined_text = self._prepare_schema_text(record)

            # Generate embedding
            embedding = self._generate_embedding(combined_text)

            # Collect data
            schema_names.append(record['DB_SCHEMA'])
            table_names.append(record['TABLE_NAME'])
            column_names.append(record['COLUMN_NAME'])
            data_types.append(record['DATA_TYPE'])
            descriptions.append(record['DESCRIPTION'])
            distinct_values.append(str(record.get('DISTINCT_VALUES', 'N/A')))
            combined_texts.append(combined_text)
            embeddings.append(embedding)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(schema_data)} records")

        # Insert data into collection
        entities = [
            schema_names,
            table_names, 
            column_names,
            data_types,
            descriptions,
            distinct_values,
            combined_texts,
            embeddings
        ]

        insert_result = collection.insert(entities)
        collection.flush()

        logger.info(f"Inserted {len(schema_data)} records into collection")

        # Create index and load collection
        self.create_index(collection)
        collection.load()

        return collection

    def retrieve_relevant_schema(self, 
                                query: str, 
                                top_k: int = 5,
                                similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve relevant database schema information for a given query

        Args:
            query: User's natural language query
            top_k: Number of top similar schema records to retrieve
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of relevant schema records with similarity scores
        """

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Get collection
        collection = Collection(self.collection_name)

        # Perform similarity search
        search_params = {
            "metric_type": "IP",  # Inner Product similarity
            "params": {"nprobe": 8}  # Fast but accurate for daily searches
        }

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=8,  # Perfect size for SQL context
            output_fields=[
                "schema_name", "table_name", "column_name", 
                "data_type", "description", "distinct_values", "combined_text"
            ]
        )

        # Process results
        relevant_schemas = []
        for hit in results[0]:
            if hit.score >= similarity_threshold:
                relevant_schemas.append({
                    "similarity_score": hit.score,
                    "schema_name": hit.entity.get("schema_name"),
                    "table_name": hit.entity.get("table_name"),
                    "column_name": hit.entity.get("column_name"),
                    "data_type": hit.entity.get("data_type"),
                    "description": hit.entity.get("description"),
                    "distinct_values": hit.entity.get("distinct_values"),
                    "combined_text": hit.entity.get("combined_text")
                })

        logger.info(f"Retrieved {len(relevant_schemas)} relevant schema records")
        return relevant_schemas

    def generate_rag_response(self, 
                            user_query: str, 
                            retrieved_schemas: List[Dict[str, Any]],
                            model: str = "gpt-3.5-turbo") -> str:
        """
        Generate SQL query and chart recommendations using retrieved schema context

        Args:
            user_query: User's natural language query
            retrieved_schemas: List of relevant schema records
            model: LLM model to use for generation

        Returns:
            Generated SQL query and recommendations
        """

        # Prepare context from retrieved schemas
        context = "\n\nRelevant Database Schema Information:\n"
        for i, schema in enumerate(retrieved_schemas, 1):
            context += f"{i}. Schema: {schema['schema_name']}\n"
            context += f"   Table: {schema['table_name']}\n"
            context += f"   Column: {schema['column_name']} ({schema['data_type']})\n"
            context += f"   Description: {schema['description']}\n"
            context += f"   Similarity Score: {schema['similarity_score']:.3f}\n\n"

        # Create prompt
        system_prompt = """You are an expert SQL developer and data analyst. 
        Based on the provided database schema information, generate:
        1. A precise SQL query to answer the user's question
        2. Chart/visualization recommendations
        3. Brief explanation of the approach

        Focus only on the relevant schema information provided. If information is insufficient, mention what additional schema details would be needed."""

        user_prompt = f"""User Query: {user_query}

        {context}

        Please provide:
        1. SQL Query
        2. Recommended Charts/Visualizations  
        3. Explanation"""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"

    def query_pipeline(self, user_query: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant schema and generate response

        Args:
            user_query: User's natural language query

        Returns:
            Dictionary containing retrieved schemas and generated response
        """

        logger.info(f"Processing query: {user_query}")

        # Step 1: Retrieve relevant schemas
        relevant_schemas = self.retrieve_relevant_schema(user_query, top_k=5)

        if not relevant_schemas:
            return {
                "user_query": user_query,
                "relevant_schemas": [],
                "generated_response": "No relevant schema information found for your query. Please try rephrasing or provide more specific details.",
                "token_savings": "100% (no schema sent to LLM)"
            }

        # Step 2: Generate response using retrieved context
        generated_response = self.generate_rag_response(user_query, relevant_schemas)

        # Calculate token savings (approximate)
        original_tokens = len(str(schema_data)) // 4  # Rough token estimate
        rag_tokens = sum(len(schema['combined_text']) for schema in relevant_schemas) // 4
        token_savings = ((original_tokens - rag_tokens) / original_tokens) * 100

        return {
            "user_query": user_query,
            "relevant_schemas": relevant_schemas,
            "generated_response": generated_response,
            "token_savings": f"{token_savings:.1f}%"
        }

def main():
    """
    Example usage of the Database Schema RAG system
    """

    # Sample database schema data (replace with your actual schema)
    sample_schema_data = [
        {
            "DB_SCHEMA": "PENGUIN_INTELLIGENCE_HUB.GOLD",
            "TABLE_NAME": "VW_KPI1_SALES_LEADER_BOARD",
            "COLUMN_NAME": "DEPARTMENT",
            "DATA_TYPE": "TEXT",
            "DISTINCT_VALUES": "['SALES', 'MARKETING', 'ENGINEERING', 'HR', 'FINANCE']",
            "DESCRIPTION": "The department column represents the specific division or unit within the organization responsible for a particular function or set of tasks related to technology solutions."
        },
        {
            "DB_SCHEMA": "PENGUIN_INTELLIGENCE_HUB.GOLD",
            "TABLE_NAME": "VW_KPI1_SALES_LEADER_BOARD",
            "COLUMN_NAME": "SALES_AMOUNT",
            "DATA_TYPE": "NUMERIC",
            "DISTINCT_VALUES": "NULL",
            "DESCRIPTION": "Total sales amount achieved by each department or individual sales representative."
        },
        # Add more schema records here...
    ]

    # Initialize RAG system
    rag_system = DatabaseSchemaRAG(
        openai_api_key="your-openai-api-key-here",
        milvus_host="localhost",
        milvus_port="19530"
    )

    # Load and embed schema data
    rag_system.load_and_embed_schema_data(sample_schema_data)

    # Example queries
    test_queries = [
        "Show me sales data by department",
        "Which columns contain sales information?",
        "What are the different departments in the organization?",
        "How can I analyze customer segments?"
    ]

    # Process queries
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print('='*80)

        result = rag_system.query_pipeline(query)

        print(f"\nToken Savings: {result['token_savings']}")
        print(f"\nRelevant Schema Records Found: {len(result['relevant_schemas'])}")

        for i, schema in enumerate(result['relevant_schemas'], 1):
            print(f"\n{i}. {schema['table_name']}.{schema['column_name']} (Score: {schema['similarity_score']:.3f})")

        print(f"\nGenerated Response:\n{result['generated_response']}")

if __name__ == "__main__":
    main()
