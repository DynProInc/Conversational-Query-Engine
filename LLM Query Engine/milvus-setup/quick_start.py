#!/usr/bin/env python3
"""
Quick Start Example for Database Schema RAG System
=================================================

This file demonstrates how to quickly get started with the RAG system.
Run this after setting up Milvus and installing dependencies.
"""

import os
from database_schema_rag_pipeline import DatabaseSchemaRAG

# Sample database schema data (replace with your actual data)
SAMPLE_SCHEMA_DATA = [
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
    {
        "DB_SCHEMA": "PENGUIN_INTELLIGENCE_HUB.GOLD", 
        "TABLE_NAME": "VW_KPI1_SALES_LEADER_BOARD",
        "COLUMN_NAME": "EMPLOYEE_ID",
        "DATA_TYPE": "INTEGER",
        "DISTINCT_VALUES": "NULL", 
        "DESCRIPTION": "Unique identifier for each employee in the sales leaderboard system."
    },
    {
        "DB_SCHEMA": "PENGUIN_INTELLIGENCE_HUB.GOLD",
        "TABLE_NAME": "VW_CUSTOMER_ANALYSIS", 
        "COLUMN_NAME": "CUSTOMER_SEGMENT",
        "DATA_TYPE": "TEXT",
        "DISTINCT_VALUES": "['ENTERPRISE', 'SMB', 'STARTUP', 'INDIVIDUAL']",
        "DESCRIPTION": "Customer segmentation based on company size and business type."
    },
    {
        "DB_SCHEMA": "PENGUIN_INTELLIGENCE_HUB.GOLD",
        "TABLE_NAME": "VW_CUSTOMER_ANALYSIS",
        "COLUMN_NAME": "REVENUE_POTENTIAL", 
        "DATA_TYPE": "NUMERIC",
        "DISTINCT_VALUES": "NULL",
        "DESCRIPTION": "Estimated revenue potential for each customer segment based on historical data and market analysis."
    }
]

def main():
    print("🚀 Database Schema RAG System - Quick Start")
    print("=" * 50)

    # Check for OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    try:
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag_system = DatabaseSchemaRAG(
            openai_api_key=openai_api_key,
            milvus_host="localhost", 
            milvus_port="19530",
            collection_name="quick_start_schema_collection"
        )

        # Load and embed schema data
        print("📊 Loading and embedding schema data...")
        rag_system.load_and_embed_schema_data(SAMPLE_SCHEMA_DATA)

        # Test queries
        test_queries = [
            "Show me sales data by department",
            "Which columns contain sales information?", 
            "What are the different customer segments?",
            "How can I find employee performance data?",
            "What revenue metrics are available?"
        ]

        print("\n🔍 Testing RAG system with sample queries...")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 40)

            # Process query through RAG pipeline
            result = rag_system.query_pipeline(query)

            print(f"💰 Token Savings: {result['token_savings']}")
            print(f"📋 Relevant Schemas Found: {len(result['relevant_schemas'])}")

            # Show retrieved schemas
            if result['relevant_schemas']:
                print("\n🎯 Retrieved Schema Information:")
                for j, schema in enumerate(result['relevant_schemas'][:2], 1):  # Show top 2
                    print(f"  {j}. {schema['table_name']}.{schema['column_name']}")
                    print(f"     Similarity: {schema['similarity_score']:.3f}")

            print(f"\n🤖 Generated Response:")
            print(result['generated_response'])

            if i < len(test_queries):
                input("\nPress Enter to continue to next query...")

        print("\n✅ Quick start completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Replace SAMPLE_SCHEMA_DATA with your actual database schema")
        print("2. Customize configuration in config_template.py")
        print("3. Deploy to production using milvus-cluster-k8s.yaml")
        print("4. Set up monitoring and alerts")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🛠️ Troubleshooting:")
        print("1. Check if Milvus is running: docker-compose ps")
        print("2. Check Milvus health: curl http://localhost:9091/healthz")
        print("3. Verify OpenAI API key is valid")
        print("4. Check logs: docker-compose logs milvus-standalone")

if __name__ == "__main__":
    main()
