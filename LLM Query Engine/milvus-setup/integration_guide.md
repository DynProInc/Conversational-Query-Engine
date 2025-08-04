# Multi-Client RAG Integration Guide

This guide explains how to integrate the enhanced schema processing with your multi-client RAG system.

## Enhanced Schema Processing

The new `schema_processor.py` file provides improved handling for database schemas, particularly for the combined `DB_SCHEMA` format (e.g., `PENGUIN_INTELLIGENCE_HUB.GOLD`) by:

1. **Properly splitting DATABASE.SCHEMA values** - Extracting database and schema components for better context
2. **Excluding empty fields** - Not including null values in embeddings to avoid misleading the model
3. **Maintaining strict client isolation** - No cross-client data sharing

## Integration Steps

### 1. Update the Schema Loading in `multi_client_rag.py`

```python
# Import the schema processor
from schema_processor import SchemaProcessor, SchemaRecord

# In the load_client_schema method:
def load_client_schema(self, client_id: str, data_dict_path: str) -> int:
    # ...existing code...
    
    # Load data dictionary
    df = pd.read_csv(data_dict_path)
    logger.info(f"Loaded {len(df)} schema records from {data_dict_path}")
    
    # Use schema processor to handle records
    schema_processor = SchemaProcessor()
    records = schema_processor.process_csv_data(client_id, df)
    
    # Process records for insertion
    combined_texts = []
    embeddings = []
    db_schemas = []
    table_names = []
    column_names = []
    data_types = []
    descriptions = []
    distinct_values = []
    
    for record in records:
        # Extract fields
        db_schemas.append(record.db_schema)
        table_names.append(record.table_name)
        column_names.append(record.column_name)
        data_types.append(record.data_type)
        descriptions.append(record.description)
        distinct_values.append(record.distinct_values if record.distinct_values else "")
        
        # Add combined text
        combined_texts.append(record.combined_text)
        
        # Generate embedding
        embedding = self._generate_embedding(record.combined_text)
        embeddings.append(embedding)
    
    # Rest of insertion code remains the same...
```

### 2. Update the `SchemaRecord` Class

Replace the current `SchemaRecord` class in `multi_client_rag.py` with the enhanced one from `schema_processor.py` or import it directly.

### 3. Enhance Retrieval Results

When retrieving schema for prompts, include the separate database and schema fields:

```python
def retrieve_relevant_schema(self, client_id: str, query: str, top_k: int = 5):
    # ...existing search code...
    
    # For each result, extract database and schema
    for hit in search_results:
        # Create SchemaRecord to leverage the extraction logic
        record = SchemaRecord(
            client_id=client_id,
            db_schema=hit.entity.get('db_schema'),
            table_name=hit.entity.get('table_name'),
            column_name=hit.entity.get('column_name'),
            data_type=hit.entity.get('data_type', ''),
            description=hit.entity.get('description', ''),
            distinct_values=hit.entity.get('distinct_values', '')
        )
        
        # Extract database and schema
        database, schema = record.extract_db_schema()
        
        # Include in result
        result = {
            'database': database,
            'schema': schema,
            'db_schema': hit.entity.get('db_schema'),
            'table_name': hit.entity.get('table_name'),
            'column_name': hit.entity.get('column_name'),
            'data_type': hit.entity.get('data_type', ''),
            'description': hit.entity.get('description', ''),
            'similarity': hit.score
        }
        
        results.append(result)
    
    # Return enhanced results
    return results
```

## Client Isolation Requirements

The enhanced schema processing maintains strict client isolation by:

1. **Processing each client's data separately** - No cross-client data sharing
2. **Using client-specific collections** - Each client has a dedicated Milvus collection
3. **No fallbacks to other clients** - If a client's data is missing or invalid, errors are properly reported

## Environment Variable Integration

The system continues to use client-specific environment variables from the client `.env` files, following your pattern:

```
CLIENT_{client_id}_VARIABLE_NAME
```

For example:
- `CLIENT_MTS_OPENAI_MODEL`
- `CLIENT_PENGUIN_OPENAI_MODEL`

No hardcoded model names or client-specific configurations are used, maintaining your requirement for fully dynamic client handling.

## Error Handling

Error handling has been enhanced to properly report issues with schema data without silent fallbacks:

- Missing required columns are logged with specific error messages
- Data processing errors are reported with client context
- Proper error propagation ensures issues are visible and not hidden
